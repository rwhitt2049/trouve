import numpy as np
from functools import wraps
import pandas as pd
# from memory_profiler import profile


def lazyproperty(func):
    name = '_lazy_' + func.__name__

    @property
    @wraps(func)
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value
    return lazy


def skip_check(*dargs):
    def wrapper(func):
        @wraps(func)
        def wrapped(self, *args):
            attrs = [getattr(self, darg) for darg in dargs]
            if any(attrs):
                func(self, *args)
        return wrapped
    return wrapper


class Events(object):
    def __init__(self, condition, sample_period=None,
                 activation_debounce=None, deactivation_debounce=None,
                 min_duration=None, max_duration=None,
                 start_offset=None, stop_offset=None):
        """

        Args:
            condition:
            sample_period:
            activation_debounce:
            deactivation_debounce:
            min_duration:
            max_duration:
            start_offset:
            stop_offset:
        """
        self._activation_debounce = activation_debounce
        self._deactivation_debounce = deactivation_debounce
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._starts = None
        self._stops = None

        if type(condition) is pd.core.series.Series:
            self.condition = condition.values
        else:
            self.condition = condition

        if not sample_period or sample_period <= 0:
            raise ValueError('sample_period must be a positive value of the time in seconds between two samples')
        else:
            self.sample_period = sample_period  # Assumes univariate time series

        if start_offset and start_offset > 0:
            raise ValueError('Currently only negative start offsets are supported')
        else:
            self._start_offset = start_offset

        if stop_offset and stop_offset < 0:
            raise ValueError('Currently only positive stop offsets are supported')
        else:
            self._stop_offset = stop_offset
        # TODO - work out strategy for multivariate data. Pass index

    @property
    def activation_debounce(self):
        try:
            return np.ceil(self._activation_debounce / self.sample_period)
        except TypeError:
            return 0

    @property
    def deactivation_debounce(self):
        try:
            return np.ceil(self._deactivation_debounce / self.sample_period)
        except TypeError:
            return 0

    @property
    def min_duration(self):
        try:
            return np.ceil(self._min_duration / self.sample_period)
        except TypeError:
            return 0

    @property
    def max_duration(self):
        try:
            return np.floor(self._max_duration / self.sample_period)
        except TypeError:
            return self.condition.size

    @property
    def start_offset(self):
        try:
            return np.ceil(self._start_offset * self.sample_period).astype('int32')
        except TypeError:
            return 0

    @property
    def stop_offset(self):
        try:
            return np.ceil(self._stop_offset * self.sample_period).astype('int32')
        except TypeError:
            return 0

    @property
    def starts(self):
        """Return a numpy.array() of start indexes."""
        if self._starts is None:
            self._apply_filters()
        return self._starts

    @property
    def stops(self):
        """Return a numpy.array() of start indexes."""
        if self._stops is None:
            self._apply_filters()
        return self._stops

    @lazyproperty
    def durations(self):
        """Return a numpy.array() of event durations in seconds."""
        return (self.stops - self.starts)/self.sample_period

    def as_array(self, false_values=0, true_values=1, dtype='float'):
        """
        Return the found events as a numpy array of 0's and 1'sample_period
        """
        try:
            from nimble.cyfunc.as_array import as_array
        except ImportError:
            from nimble.as_array import as_array

        output = np.ones(self.condition.size) * false_values
        output = as_array(self.starts, self.stops, output, true_values)
        return output.astype(dtype)

    def as_series(self, false_values=0, true_values=1, name='events'):
        index = pd.RangeIndex(self.condition.size, step=self.sample_period)
        data = self.as_array(false_values=false_values, true_values=true_values)
        return pd.Series(data=data, index=index, name=name)

    def _apply_filters(self):
        self.apply_condition_filter()
        self.apply_debounce_filter()
        self.apply_event_length_filter()
        self.apply_offsets()

    def apply_condition_filter(self):
        """
        Apply initial masking conditions
        """
        mask = (self.condition > 0).view('i1')
        slice_index = np.arange(mask.size + 1, dtype='int32')

        # Determine if condition is active at array start, set to_begin accordingly
        if mask[0] == 0:
            to_begin = np.array([0], dtype='i1')
        else:
            to_begin = np.array([1], dtype='i1')

        # Determine if condition is active at array end, set to_end accordingly
        if mask[-1] == 0:
            to_end = np.array([0], dtype='i1')
        else:
            to_end = np.array([-1], dtype='i1')

        deltas = np.ediff1d(mask, to_begin=to_begin, to_end=to_end)

        self._starts = np.ma.masked_where(deltas < 1, slice_index).compressed()
        self._stops = np.ma.masked_where(deltas > -1, slice_index).compressed()

    @skip_check('_activation_debounce', '_deactivation_debounce')
    def apply_debounce_filter(self):
        """ Apply debounce parameters"""
        try:
            from nimble.cyfunc.debounce import debounce
        except ImportError:
            from nimble.debounce import debounce

        self._starts, self._stops = debounce(self._starts, self._stops,
                                             self.activation_debounce, self.deactivation_debounce)

    @skip_check('_min_duration', '_max_duration')
    def apply_event_length_filter(self):
        event_lengths = self._stops - self._starts
        condition = ((event_lengths < self.min_duration) |
                     (event_lengths > self.max_duration))

        self._starts = np.ma.masked_where(condition, self._starts).compressed()
        self._stops = np.ma.masked_where(condition, self._stops).compressed()

    @skip_check('_start_offset', '_stop_offset')
    def apply_offsets(self):
        min_index = 0
        max_index = self.condition.size

        self._starts += self.start_offset
        self._stops += self.stop_offset

        np.clip(self._starts, min_index, max_index, out=self._starts)
        np.clip(self._stops, min_index, max_index, out=self._stops)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        try:
            self.start = self.starts[self.i]
            self.stop = self.stops[self.i]
            self.duration = (self.stop - self.start)/self.sample_period
            self.i += 1
            self.index = self.i-1
            return self
        except IndexError:
            raise StopIteration

    def __len__(self):
        return self.starts.size

    def __repr__(self):
        # TODO - due to the size of condition, this should take an optional path and serialize as pickle, yaml, or json
        return ('{__class__.__name__}(condition={condition!r}, sample_period={sample_period!r}, '
                'activation_debounce={_activation_debounce!r}, deactivation_debounce={_deactivation_debounce!r}, '
                'min_duration={_min_duration!r}, max_duration={_max_duration!r}, '
                'start_offset={_start_offset!r}, stop_offset={_stop_offset!r}').format(__class__=self.__class__,
                                                                                       **self.__dict__)

    def __str__(self):
        args = [len(self), np.min(self.durations), np.max(self.durations), np.mean(self.durations)]
        kwargs = {
            'sample_period': '{}Hz'.format(self.sample_period),
            '_activation_debounce': '{}s'.format(self._activation_debounce) if self._activation_debounce else None,
            '_deactivation_debounce': '{}s'.format(self.deactivation_debounce) if self.deactivation_debounce else None,
            '_min_duration': '{}s'.format(self._min_duration) if self._min_duration else None,
            '_max_duration': '{}s'.format(self._max_duration) if self._max_duration else None,
            '_start_offset': '{}s'.format(self._start_offset) if self._start_offset else None,
            '_stop_offset': '{}s'.format(self._stop_offset) if self._stop_offset else None
        }
        return ('Number of events: {0}'
                '\nMin, Max, Mean Duration: {1:.3f}s ,{2:.3f}s, {3:.3f}s'
                '\nsample rate: {sample_period}, '
                '\nactivation_debounce: {_activation_debounce} deactivation_debounce: {_deactivation_debounce}, '
                '\nmin_duration: {_min_duration}, max_duration: {_max_duration}, '
                '\nstart_offset: {_start_offset}, stop_offset: {_stop_offset}').format(*args, **kwargs)

    def __eq__(self, other):
        if (np.all(self.starts == other.starts) and np.all(self.stops == other.stops)
                and self.sample_period == other.sample_rate and self.condition.size == other.condition.size):
            return True
        else:
            return False

    def __hash__(self):
        """Numpy arrays aren't hashable. Researching solution that doesn't require something beyond standard lib."""
        return id(self)


def main():
    np.random.seed(15)
    mask = np.random.random_integers(0, 1, 1000000)
    events = Events(mask > 0,
                    activation_debounce=2,
                    min_duration=3,
                    start_offset=-1)
    
    starts = events.starts
    series = events.as_series()
    array = events.as_array()


if __name__ == '__main__':
    import sys
    sys.exit(main())
