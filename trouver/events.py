from collections import namedtuple
from functools import wraps

import numpy as np
import pandas as pd

# from memory_profiler import profile

# TODO update all docs

Occurrence = namedtuple('Occurrence', 'start stop slice duration')


def lazyproperty(func):
    """Cache a property as an attr"""
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


class Events(object):
    """Object to represent events in time series data

    Attributes:
        name (:obj: `str`): User provided name for events.
        _starts (:obj: `np.array` of int):
        _stops (:obj: `np.array` of int):
        _period (float):
        _condition_size (int):
    """
    def __init__(self, starts, stops, period, name, condition_size):
        self.name = name
        self._starts = starts
        self._stops = stops
        self._period = period
        self._condition_size = condition_size

    @lazyproperty
    def durations(self):
        """Return a numpy.array of event durations in seconds."""
        durations = (self._stops - self._starts) * self._period
        return durations

    # TODO force defaults to np.int8 datatype
    # specify false and true values as None, check if values and dtype is noe
    # and specify appropriately
    def as_array(self, false_values=0, true_values=1, dtype=np.float):
        """Returns a numpy.array that identifies events

        Useful for plotting or for creating a new mask.

        Parameters:
            false_values(float, optional): Default is 0. Value of array
                where events are not active.
            true_values (float, optional): Default is 1. Value of array
                where events are active.
            dtype (numpy.dtype, optional): Default is' numpy.float.
                Datatype of returned array.

        Returns:
            array: numpy.ndarray
                Ndarray of specified values that identify where events
                were found.

        """
        output = np.ones(self._condition_size, dtype=dtype) * false_values
        for start, stop in zip(self._starts, self._stops):
            output[start:stop] = 1 * true_values
        return output.astype(dtype)

    def as_series(self, false_values=0, true_values=1, name=None):
        """Returns a pandas.series that identifies events

        Useful for plotting, for creating a new mask, or using for
        group_by functionality in dataframes

        Parameters:
            false_values (float, optional): Default is 0. Value of
                array where events are not active.
            true_values (float, optional): Default is 1. Value of array
                where events are active.
            name: str, optional
                The name of the event. Default is 'events'

        Returns:
            series: pandas.Series
                Series of specified values that identify events
        """
        if name is None:
            name = self.name
        data = self.as_array(false_values=false_values, true_values=true_values)
        return pd.Series(data=data, name=name)

    def as_mask(self):
        """Returns an np.ndarray bool mask

        This method returns a numpy.ndarray of bools where values are
        False where the condition is met, and True where the condition
        are not met. Trouver treats conditionals opposite of how numpy
        treats them. That is to say that numpy will mask out values in
        an array that meet the condition, however trouver is by design
        more interested in finding and keeping events that meet the
        given condition. This method makes it more convenient to
        interact with the numpy masked array module.

        Examples:
            >>> from trouver import find_events
            >>> x = np.array([2, 2, 4, 5, 3, 2])
            >>> condition = x > 2
            >>> print(condition)
            [False False  True  True  True False]
            >>> events = find_events(condition, 1)
            >>> print(events.as_array())
            [ 0.  0.  1.  1.  1.  0.]
            >>> print(events.as_mask())
            [ True  True False False False  True]
            >>> print(np.ma.masked_where(events.as_mask(), x))
            [-- -- 4 5 3 --]

        Returns:
            np.ndarray of bool:
        """
        return self.as_array(1, 0, np.int8).view(bool)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        try:
            occurrence = Occurrence(
                start=self._starts[self._i],
                stop=self._stops[self._i] - 1,
                slice=slice(self._starts[self._i], self._stops[self._i]),
                duration=(self._stops[self._i] - self._starts[self._i]) * self._period
            )
            self._i += 1
            return occurrence
        except IndexError:
            raise StopIteration

    def __getitem__(self, item):
        occurrence = Occurrence(
            istart=self._starts[item],
            istop=self._stops[item]-1,
            slice=slice(self._starts[item], self._stops[item]),
            duration=(self._stops[item] - self._starts[item]) * self._period
        )
        return occurrence

    def __len__(self):
        return self._starts.size

    def __repr__(self):
        return (
            '{__class__.__name__}(_starts={_starts!r}, '
            '_stops={_stops!r}, '
            '_period={_period!r}, '
            'name={name!r}, '
            '_condition_size={_condition_size!r})'
        ).format(__class__=self.__class__, **self.__dict__)

    def __str__(self):
        args = [len(self),
                np.min(self.durations),
                np.max(self.durations),
                np.mean(self.durations)]

        kwargs = {'name': '{}'.format(self.name),
                  'period': '{}s'.format(self._period)}
        return (
            '{name}'
            '\nNumber of events: {0}'
            '\nMin, Max, Mean Duration: {1:.3f}s, {2:.3f}s, {3:.3f}s'
        ).format(*args, **kwargs)

    def __eq__(self, other):
        """Determine if two Events objects are identical

        Compares _starts, _stops, _period and condition.size to
        determine if two events are identical. Identical events objects
        can have different names and still be equal.
        """
        if (np.all(self._starts == other._starts)
                and np.all(self._stops == other._stops)
                and self._period == other._period
                and self._condition_size == other._condition_size):
            return True
        else:
            return False

    def __hash__(self):
        """Numpy arrays aren't hashable.

        Researching solution that doesn't require something beyond
        standard lib.
        """
        return id(self)


def main():
    import logging
    from trouver import debounce, offset_events, filter_durations
    logger = logging.getLogger('trouver')
    logger.setLevel(logging.DEBUG)
    np.random.seed(10)
    debounce = debounce(2, 2)
    offset_events = offset_events(-11, 2)
    filter_durations = filter_durations(3, 5)

    x = np.random.random_integers(0, 1, 20)
    y = np.random.random_integers(2, 4, 20)
    condition = (x > 0) & (y <= 3)

    events = find_events(condition, 1, debounce, filter_durations,
                         offset_events, name='example')
    print(repr(events))
    print(events.as_array())
    print(events.durations)
    events.as_array()
    print(events)


if __name__ == '__main__':
    import sys
    sys.exit(main())
