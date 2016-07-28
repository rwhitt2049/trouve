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
    def __init__(self, condition, sample_rate=1,
                 entry_debounce=None, exit_debounce=None,
                 min_event_length=None, max_event_length=None,
                 start_offset=None, stop_offset=None):
        """

        Args:
            condition:
            sample_rate:
            entry_debounce:
            exit_debounce:
            min_event_length:
            max_event_length:
            start_offset:
            stop_offset:
        """
        if type(condition) is pd.core.series.Series:
            self.condition = condition.values
        else:
            self.condition = condition
        self._starts = None
        self._stops = None
        self.sample_rate = sample_rate  # Assumes univariate time series
        self._entry_debounce = entry_debounce
        self._exit_debounce = exit_debounce
        self._min_event_length = min_event_length
        self._max_event_length = max_event_length

        if start_offset is not None and start_offset > 0:
            raise ValueError('Currently only negative start offsets are supported')
        if stop_offset is not None and stop_offset < 0:
            raise ValueError('Currently only positive stop offsets are supported')

        self._start_offset = start_offset
        self._stop_offset = stop_offset

        # TODO - work out strategy for multivariate data. Pass index

    @property
    def entry_debounce(self):
        try:
            return np.ceil(self._entry_debounce * self.sample_rate)
        except TypeError:
            return 0

    @property
    def exit_debounce(self):
        try:
            return np.ceil(self._exit_debounce * self.sample_rate)
        except TypeError:
            return 0

    @property
    def min_event_length(self):
        try:
            return np.ceil(self._min_event_length * self.sample_rate)
        except TypeError:
            return 0

    @property
    def max_event_length(self):
        try:
            return np.floor(self._max_event_length * self.sample_rate)
        except TypeError:
            return self.condition.size

    @property
    def start_offset(self):
        try:
            return np.ceil(self._start_offset * self.sample_rate).astype('int32')
        except TypeError:
            return 0

    @property
    def stop_offset(self):
        try:
            return np.ceil(self._stop_offset * self.sample_rate).astype('int32')
        except TypeError:
            return 0

    @property
    def n_events(self):
        """Return the number of events found."""
        return self.starts.size

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
        return (self.stops - self.starts)/self.sample_rate

    def as_array(self, false_values=0, true_values=1, dtype='float'):
        """
        Return the found events as a numpy array of 0's and 1'sample_rate
        """
        try:
            from nimble.cyfunc.as_array import as_array
        except ImportError:
            from nimble.as_array import as_array

        output = np.ones(self.condition.size) * false_values
        output = as_array(self.starts, self.stops, output, true_values)
        return output.astype(dtype)

    def as_series(self, false_values=0, true_values=1, name='events'):
        index = pd.RangeIndex(self.condition.size, step=self.sample_rate)
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

    @skip_check('_entry_debounce', '_exit_debounce')
    def apply_debounce_filter(self):
        """ Apply debounce parameters"""
        try:
            from nimble.cyfunc.debounce import debounce
        except ImportError:
            from nimble.debounce import debounce

        self._starts, self._stops = debounce(self._starts, self._stops,
                                             self.entry_debounce, self.exit_debounce)

    @skip_check('_min_event_length', '_max_event_length')
    def apply_event_length_filter(self):
        event_lengths = self._stops - self._starts
        condition = ((event_lengths < self.min_event_length) |
                     (event_lengths > self.max_event_length))

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
            self.duration = (self.stop - self.start)/self.sample_rate
            self.i += 1
            self.index = self.i-1
            return self
        except IndexError:
            raise StopIteration

    def __len__(self):
        return self.starts.size

    def __repr__(self):
        # TODO - due to the size of condition, this should take an optional path and serialize as pickle, yaml, or json
        return ('{__class__.__name__}(condition={condition!r}, sample_rate={sample_rate!r}, '
                'entry_debounce={_entry_debounce!r}, exit_debounce={_exit_debounce!r}, '
                'min_event_length={_min_event_length!r}, max_event_length={_max_event_length!r}, '
                'start_offset={_start_offset!r}, stop_offset={_stop_offset!r}').format(__class__=self.__class__,
                                                                                       **self.__dict__)

    def __str__(self):
        args = [len(self), np.min(self.durations), np.max(self.durations), np.mean(self.durations)]
        kwargs = {
            'sample_rate': '{}Hz'.format(self.sample_rate),
            '_entry_debounce': '{}s'.format(self._entry_debounce) if self._entry_debounce else None,
            '_exit_debounce': '{}s'.format(self.exit_debounce) if self.exit_debounce else None,
            '_min_event_length': '{}s'.format(self._min_event_length) if self._min_event_length else None,
            '_max_event_length': '{}s'.format(self._max_event_length) if self._max_event_length else None,
            '_start_offset': '{}s'.format(self._start_offset) if self._start_offset else None,
            '_stop_offset': '{}s'.format(self._stop_offset) if self._stop_offset else None
        }
        return ('Number of events: {0}'
                '\nMin, Max, Mean Duration: {1:.3f}s ,{2:.3f}s, {3:.3f}s'
                '\nsample rate: {sample_rate}, '
                '\nentry_debounce: {_entry_debounce} exit_debounce: {_exit_debounce}, '
                '\nmin_event_length: {_min_event_length}, max_event_length: {_max_event_length}, '
                '\nstart_offset: {_start_offset}, stop_offset: {_stop_offset}').format(*args, **kwargs)

    def __eq__(self, other):
        if (np.all(self.starts == other.starts) and np.all(self.stops == other.stops)
                and self.sample_rate == other.sample_rate and self.condition.size == other.condition.size):
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
                    entry_debounce=2,
                    min_event_length=3,
                    start_offset=-1)
    
    starts = events.starts
    series = events.as_series()
    array = events.as_array()


if __name__ == '__main__':
    import sys
    sys.exit(main())
