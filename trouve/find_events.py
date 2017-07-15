import numpy as np
import pandas as pd
import warnings
from toolz import pipe


from trouve.events import Events


def find_events(condition, period, *transformations_, name='events', transformations=None):
    """Find events based off a condition

    Find events based off a ``bool`` conditional array and apply a sequence
    of transformation functions to them.

    Args:
        condition (``numpy.ndarray`` or ``pandas.Series`` of ``bool``):
            Boolean conditional array.
        period (``float``):
            Time in seconds between each data point. Requires constant
            increment data that is uniform across the array. (1/Hz = s)
        transformations (sequence of ``callable`` 's, optional):
            Ordered sequence of transformation functions to apply to
            events. Transformations are applied via ``toolz.pipe()``
        name (``str``, optional): Default is ``'events'``.
            User provided name for events.

    Returns:
        :class:`trouve.events.Events`:
            Returns events found from ``condition`` with any supplied
            ``*transformations`` applied.

    Examples:
        >>> import trouve as tr
        >>> import trouve.transformations as tt
        >>> import numpy as np
        >>> deb = tt.debounce(2, 2)
        >>> offsets = tt.offset_events(-1,2)
        >>> filt_dur = tt.filter_durations(3, 5)
        >>> x = np.array([4, 5, 1, 2, 3, 4, 5, 1, 3])
        >>> condition = (x > 2)
        >>> no_transforms = tr.find_events(condition, period=1)
        >>> events = tr.find_events(condition, period=1,
        ... transformations=[deb, filt_dur, offsets])
        >>> no_transforms.to_array()
        array([ 1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.])
        >>> events.to_array()
        array([ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.])

    """
    if type(condition) is pd.core.series.Series:
        condition = condition.values

    if transformations is None:
        transformations = []

    if transformations_:
        warnings.warn('Transformations should be specified with the '
                      'transformations keyword, not  as a *arg', DeprecationWarning)
        transformations = transformations_
        # TODO Deprecating in v0.5.x, removing in v0.6.x

    starts, stops = _apply_condition(condition)
    raw_events = Events(starts, stops, period, name, condition.size)

    transformed_events = pipe(raw_events, *transformations)

    return transformed_events


def _apply_condition(condition):
    """Distill an array of bool into start and stop indexes

    Convert a conditional array of bools into two numpy.ndarrays of
    integers where starts are the indexes where condition goes from
    False to True. Stops are the indexes where condition goes from
    True to False.

    Args:
        condition (numpy.array of bool):

    Returns:
        tuple(numpy.ndarray, numpy.ndarray):

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
