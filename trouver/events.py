from collections import namedtuple
from functools import wraps
from inspect import signature

import numpy as np
import pandas as pd
from toolz import pipe

from trouver.transformations import apply_condition

# from memory_profiler import profile

# TODO update all docs

Occurrence = namedtuple('Occurrence', 'istart istop slice duration')


def curry_func(func, period, condition_size):
    """Add _period and condition_size arguments if required

    Allows any transformation function that decorated with toolz.curry
    to have access to both the _period and condition_size arguments.
    If func requires neither _period or condition_size, then no
    operations are performed and the original function is returned.

    See Also: toolz.curry

    Args:
        func: A curried transformation function
        period (float): Sample _period from time series data
        condition_size (int): Size of condition numpy array

    Returns:
        callable: Curried function with period and/or condition_size

    """
    sig = signature(func)
    if 'period' in sig.parameters.keys():
        func = func(period=period)

    if 'condition_size' in sig.parameters.keys():
        func = func(condition_size=condition_size)

    return func


def find_events(condition, period, *transformations, name='events'):
    """Find events based off a condition

    Find events based off a bool conditional array and apply a sequence
    of transformation functions to them.

    See Also:
        trouver.events.Events

    Args:
        condition (:obj: `np.ndarray` or :obj: `pd.core.Series` bool):
            User supplied boolean conditional array.
        period (float): Time in seconds between each data point.
            Requires constant increment data that is uniform across
            all data. (1/Hz = s)
        *transformations (functions, optional): Sequence of
            transformation functions to apply to events derived from
            supplied condition. Supplied functions are applied via
            toolz.pipe()
        name (:obj: `str`, optional): Default is `'events'`.
            User provided name for events.

    Returns:
        trouver.events.Events: Returns events found from condition with
        any supplied transformation applied.

    Examples:
        >>> from trouver import find_events, debounce, offset_events, filter_durations
        >>> import numpy as np
        >>> np.random.seed(10)

        >>> debounce = debounce(2, 2)
        >>> offset_events = offset_events(-1,2)
        >>> filter_durations = filter_durations(3, 5)

        >>> x = np.random.random_integers(0, 1, 20)
        >>> y = np.random.random_integers(2, 4, 20)
        >>> condition = (x>0) & (y<=3)

        >>> events = find_events(condition, 1, debounce,
        ... filter_durations, offset_events, name='example')

        >>> events.durations
        array([7])

        >>> len(events)
        1

        >>> events.name
        'example'

        >>> events.as_array()
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> events.as_series()
        0     0.0
        1     0.0
        2     0.0
        3     0.0
        4     0.0
        5     0.0
        6     0.0
        7     1.0
        8     1.0
        9     1.0
        10    1.0
        11    1.0
        12    1.0
        13    1.0
        14    0.0
        15    0.0
        16    0.0
        17    0.0
        18    0.0
        19    0.0
        Name: example, dtype: float64

        >>> print(events)
        example
        Number of events: 1
        Min, Max, Mean Duration: 7.000s, 7.000s, 7.000s

        >>> string = 'Event {} was {}s in duration'
        >>> for i, event in enumerate(events):
        ...     print(string.format(i, event.duration))
        Event 0 was 7s in duration

        >>> string = ('Event {}, first y val is {}, last is {} and'
        ... ' slice is {}')
        >>> for i, event in enumerate(events):
        ...     print(string.format(i, y[event.istart],
        ...     y[event.istop], y[event.slice]))
        Event 0, first y val is 3, last is 3 and slice is [3 2 2 4 3 4 3]

        >>> events2 = find_events(condition, 1, debounce,
        ... filter_durations, offset_events, name='example')
        >>> events2 == events
        True
    """
    if type(condition) is pd.core.series.Series:
        condition = condition.values

    raw_events = apply_condition(condition)
    curried_funcs = [curry_func(func, period, condition.size) for func in transformations]
    transformed_events = pipe(raw_events, *curried_funcs)

    starts = transformed_events.starts
    stops = transformed_events.stops

    return Events(starts, stops, period, name, condition.size)


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
                Series of specified values that identify where
                events found.
        """
        if name is None:
            name = self.name
        data = self.as_array(false_values=false_values, true_values=true_values)
        return pd.Series(data=data, name=name)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        try:
            occurrence = Occurrence(
                istart=self._starts[self.i],
                istop=self._stops[self.i] - 1,
                slice=slice(self._starts[self.i], self._stops[self.i]),
                duration=(self._stops[self.i] - self._starts[self.i]) * self._period
            )
            self.i += 1
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

    events = find_events(condition, 1, debounce, filter_durations, offset_events, name = 'example')
    print(repr(events))
    print(events.as_array())
    print(events.durations)
    events.as_array()
    print(events)



if __name__ == '__main__':
    import sys
    sys.exit(main())
