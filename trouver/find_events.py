import numpy as np
import pandas as pd
from toolz import pipe, curry

from trouver.events import Events


@curry
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

        >>> print(events.durations)
        [7]

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
        >>> for _i, event in enumerate(events):
        ...     print(string.format(_i, event.duration))
        Event 0 was 7s in duration

        >>> string = ('Event {}, first y val is {}, last is {} and'
        ... ' slice is {}')
        >>> for _i, event in enumerate(events):
        ...     print(string.format(_i, y[event.start],
        ...     y[event.stop], y[event.slice]))
        Event 0, first y val is 3, last is 3 and slice is [3 2 2 4 3 4 3]

        >>> events2 = find_events(condition, 1, debounce,
        ... filter_durations, offset_events, name='example')
        >>> events2 == events
        True
    """
    if type(condition) is pd.core.series.Series:
        condition = condition.values

    starts, stops = _apply_condition(condition)
    raw_events = Events(starts, stops, period, name, condition.size)

    transformed_events = pipe(raw_events, *transformations)

    return transformed_events


def _apply_condition(condition):
    """Convert an array of bool to starts and stops

    Convert a conditional array of bools into two numpy.ndarrays of
    integers where starts are the indexes where condition goes from
    False to True. Stops are the indexes where condition goes from
    True to False.

    Args:
        condition (:obj: `np.array` of bool):

    Returns:
        tuple(:obj:`numpy.ndarray`, :obj:`numpy.ndarray`):

    """
    if isinstance(condition, pd.core.series.Series):
        condition = condition.values

    mask = (condition > 0).view('i1')
    slice_index = np.arange(mask.size + 1, dtype=np.int32)

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

    starts = np.ma.masked_where(deltas < 1, slice_index).compressed()
    stops = np.ma.masked_where(deltas > -1, slice_index).compressed()

    return starts, stops
