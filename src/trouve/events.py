from collections import namedtuple

import numpy as np
import pandas as pd

Occurrence = namedtuple('Occurrence', 'start stop slice duration')


class Events(object):
    """Object to represent events found in time series data

    A representation of events based off a ``bool`` conditional array.

    Attributes:
        name (``str``): User provided name for events.
        _starts (``np.array`` of ``int``): The index for event starts
        _stops (``np.array`` of ``int``): The index for event stops
        _period (``float``): Time between each value of the original condition array
        _condition_size (``int``): The size of the original condition array

    """
    def __init__(self, starts, stops, period, name, condition_size):
        self.name = name
        self._starts = starts
        self._stops = stops
        self._period = period
        self._condition_size = condition_size

    @property
    def durations(self):
        """Return a ``numpy.ndarray`` of event durations in seconds.

        Examples:
            >>> import trouve as tr
            >>> x = np.array([2, 2, 4, 5, 3, 2])
            >>> condition = x == 2
            >>> events = tr.find_events(condition, period=1)
            >>> events.to_array()  # doctest: +SKIP
            array([1., 1., 0., 0., 0., 1.])
            >>> print(events.durations)
            [2 1]

        """
        durations = (self._stops - self._starts) * self._period
        return durations

    def to_array(self, inactive_value=0, active_value=1, dtype=None, order='C'):
        """Returns a ``numpy.ndarray`` identifying found events

        Useful for plotting or building another mask based on identified
        events.

        Parameters:
            inactive_value (``float``, optional): Default is 0.
                Value of array where events are not active.
            active_value (``float``, optional): Default is 1.
                Value of array where events are active.
            dtype (``numpy.dtype``, optional): Default is ``numpy.float64``.
                The datatype of returned array.
            order (``str``, optional): Default is 'C'. {'C', 'F'} whether to
                store multidimensional data in C- or Fortran-contiguous (row-
                or column-wise) order in memory.

        Returns:
            ``numpy.ndarray``: An array where values are coded to 
                identify when events are active or inactive.

        Examples:
            >>> import trouve as tr
            >>> x = np.array([2, 2, 4, 5, 3, 2])
            >>> condition = x > 2
            >>> print(condition)
            [False False  True  True  True False]
            >>> events = tr.find_events(condition, period=1)
            >>> events.to_array()  # doctest: +SKIP
            array([0., 0., 1., 1., 1., 0.])

        """
        if dtype is None and inactive_value == 0 and active_value == 1:
            dtype = np.int8

        output = np.full(self._condition_size, inactive_value, dtype, order)

        for event in self:
            output[event.slice] = active_value

        return output.astype(dtype)

    def to_series(self, inactive_value=0, active_value=1,
                  index=None, dtype=None, name=None):
        """Returns a ``pandas.Series`` identifying found events

        Useful for plotting and for filtering a ``pandas.DataFrame``

        Parameters:
            inactive_value(``float``, optional): Default is 0.
                Value of array where events are not active.
            active_value (``float``, optional): Default is 1.
                Value of array where events are active.
            index (``array-like`` or ``Index`` (1d)):Values must be
                hashable and have the same length as data. Non-unique
                index values are allowed. Will default to
                RangeIndex(len(data)) if not provided. If both a dict
                and index sequence are used, the index will override
                the keys found in the dict.
            dtype (``numpy.dtype`` or ``None``): If ``None``, ``dtype``
                will be inferred.
            name (``str``, optional): Default is :attr:`Events.name`.
                Name of series.

        Returns:
            ``pandas.Series``:
                A series where values are coded to identify when events are active
                or inactive.

        Examples:
            >>> import trouve as tr
            >>> x = np.array([2, 2, 4, 5, 3, 2])
            >>> condition = x > 2
            >>> print(condition)
            [False False  True  True  True False]
            >>> events = tr.find_events(condition, period=1)
            >>> events.to_series()
            0    0.0
            1    0.0
            2    1.0
            3    1.0
            4    1.0
            5    0.0
            Name: events, dtype: float64

        """
        if dtype is None and inactive_value == 0 and active_value == 1:
            dtype = np.int8
        if name is None:
            name = self.name
        data = self.to_array(inactive_value=inactive_value, active_value=active_value, dtype=dtype)
        return pd.Series(data=data, index=index, name=name)

    def __iter__(self):
        for start, stop in zip(self._starts, self._stops):
            occurrence = Occurrence(
                start=start,
                stop=stop - 1,
                slice=slice(start, stop),
                duration=(stop - start) * self._period
            )
            yield occurrence

    def __getitem__(self, item):
        """Get a specific :class:`.Occurrence`

        Examples:
            >>> import numpy as np
            >>> import trouve as tr
            >>> x = np.array([0, 1, 1, 0, 1, 0])
            >>> example = tr.find_events(x, period=1, name='example')
            >>> first_event = example[0]
            >>> print(first_event)
            Occurrence(start=1, stop=2, slice=slice(1, 3, None), duration=2)

        """
        if isinstance(item, slice):
            pass
        else:
            occurrence = Occurrence(
                start=self._starts[item],
                stop=self._stops[item] - 1,
                slice=slice(self._starts[item], self._stops[item]),
                duration=(self._stops[item] - self._starts[item]) * self._period
            )
            return occurrence

    def __len__(self):
        """Returns the number of events found

        Redirects to :any:`Events._starts` and returns ``Events._starts.size``

        Examples:
            >>> import numpy as np
            >>> import trouve as tr
            >>> x = np.array([0, 1, 1, 0, 1, 0])
            >>> example = tr.find_events(x, period=1, name='example')
            >>> len(example)
            2

        """
        return len(self._starts)

    def __repr__(self):
        msg = (
            '{__class__.__name__}(_starts={_starts!r}, '
            '_stops={_stops!r}, '
            '_period={_period!r}, '
            'name={name!r}, '
            '_condition_size={_condition_size!r})'
        ).format(__class__=self.__class__, **self.__dict__)

        return msg

    def __str__(self):
        """Prints a summary of the events

        Examples:
            >>> import numpy as np
            >>> import trouve as tr
            >>> x = np.array([0, 1, 1, 0, 1, 0])
            >>> example = tr.find_events(x, period=1, name='example')
            >>> print(example)
            example
            Number of events: 2
            Min, Max, Mean Duration: 1.000s, 2.000s, 1.500s

        """
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

        Compares :attr:`Events._starts`, :attr:`Events._stops`, :attr:`Events._period`
        and :attr:`Events.condition.size` to determine if equality of two events.
        Events objects can have different names and still be equal.

        Examples:
            >>> import numpy as np
            >>> import trouve as tr
            >>> x = np.array([0, 1, 1, 0, 1, 0])
            >>> example = tr.find_events(x, period=1, name='example')
            >>> other = tr.find_events(x, period=1, name='other')
            >>> id(example) # doctest: +SKIP
            2587452050568
            >>> id(other) # doctest: +SKIP
            2587452084352
            >>> example == other
            True
            >>> example != other
            False

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
