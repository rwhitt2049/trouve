from collections import namedtuple
from functools import partial
import logging
import numpy as np
import pandas as pd
from toolz import curry

RawEvents = namedtuple('RawEvents', 'starts stops')

logger = logging.getLogger('trouver.transformations')
logger.setLevel(logging.DEBUG)

def apply_condition(condition):
    """Convert an array of bool to starts and stops

    Convert a conditional array of bools into two numpy.ndarrays of
    integers where starts are the indexes where condition goes from
    False to True. Stops are the indexes where condition goes from
    True to False.

    Args:
        condition (:obj: `np.array` of bool):

    Returns:
        collections.namedtuple:
            trouver.transformations.RawEvents(starts, stops)

        Both entries in `Rawevents`, starts and stops, are both numpy
        arrays of integers. Starts is where conditions go from False to
        True. Stops is where conditions go from True to False.

    Examples:


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

    return RawEvents(starts, stops)


def debounce(entry_debounce=None, exit_debounce=None):
    """Debounce activation and deactivation of events

    Find an occurrence that is active for time >= adeb and activate
    event. Deactivate event only after an occurrence is found that
    is inactive for time >= to ddeb. Filter out all events that fall
    outside of these bounds. This function is used to prevent short
    lived occurrences from activating or deactivating longer events.
    See mechanical debounce in mechanical switches and relays for a
    similar concept.

    See Also:
        trouver.transfomations._debounce

    Args:
        adeb (float): Time in seconds. Default is None. An event
            must be active >= adeb to activate event.
        ddeb (float): Time in seconds. Default is None. An event
            must be inactive >= ddeb to deactivate event.

    Returns:
        callable: Partial function

    Examples:

    """
    return partial(_debounce, entry_debounce=entry_debounce, exit_debounce=exit_debounce)


@curry
def _debounce(events, period, entry_debounce, exit_debounce):
    """Debounce activate and deactivation of events

    See Also:
        trouver.transfomations.debounce

    Args:
        events(:obj: `collections.namedtuple` of int):
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops
        period (float): Time in seconds between each data point.
            Requires constant increment data that is uniform across
            all data. (1/Hz = s)
        entry_debounce (float): Time in seconds.
        exit_debounce (float): Time in seconds.

    Returns:
        collections.namedtuple:
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    # convert entry_debounce from time seconds to the number of elements
    if entry_debounce is None:
        _entry_debounce = 0
    elif entry_debounce >= 0:
        _entry_debounce = np.ceil(entry_debounce / period)
    else:
        raise TypeError('entry_debounce should be a value >= 0 or None')

    # convert entry_debounce from time seconds to the number of elements
    if exit_debounce is None:
        _exit_debounce = 0
    elif exit_debounce >= 0:
        _exit_debounce = np.ceil(exit_debounce / period)
    else:
        raise TypeError('exit_debounce should be a value >= 0 or None')

    # from here on out, all calculations are in number of points, not seconds

    start_mask = np.zeros(events.starts.size)
    stop_mask = np.zeros(events.stops.size)
    event_active = False

    for index in np.arange(events.starts.size):
        # get time of the event
        event_length = events.stops[index] - events.starts[index]

        # get time to the next event
        try:
            reset_length = events.starts[index + 1] - events.stops[index]
        except IndexError:
            reset_length = None

        # determine if event entry conditions are met
        if event_active:
            pass
        elif not event_active and event_length >= _entry_debounce:
            event_active = True
        elif not event_active and event_length < _entry_debounce:
            start_mask[index] = 1
            stop_mask[index] = 1
        else:
            raise ValueError

        # determine whether or not an active event stops being
        # active relative to reset_length
        if not event_active or reset_length is None:
            pass
        elif event_active and reset_length >= _exit_debounce:
            event_active = False
        elif event_active and reset_length < _exit_debounce:
            start_mask[index + 1] = 1
            stop_mask[index] = 1
        else:
            raise ValueError

    starts = np.ma.masked_where(start_mask > 0, events.starts).compressed()
    stops = np.ma.masked_where(stop_mask > 0, events.stops).compressed()

    return RawEvents(starts, stops)


def filter_durations(mindur=None, maxdur=None):
    """Filter out durations based on length of time active

    Filter out events that are < mindur and > maxdur (time in seconds).

    See Also:
        trouver.transfomations._filter_durations

    Args:
        mindur (float): Time in seconds. Default is None. Any
        occurrence whose time is < mindur is filtered out.
        maxdur (float): Time in seconds. Default is None. Any
            occurrence whose time is > maxdur is filtered out.

    Returns:
        callable: Partial function

    Examples:

    """
    return partial(_filter_durations, mindur=mindur, maxdur=maxdur)


@curry
def _filter_durations(events, period, mindur, maxdur):
    """Filter out events based  on duration

    See Also:
        trouver.transfomations.filter_durations

    Args:
        events (:obj: `collections.namedtuple` of int):
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops
        period (float): Time in seconds between each data point.
            Requires constant increment data that is uniform across
            all data. (1/Hz = s)
        mindur (float): Time in seconds. Any occurrence whose time is
            < mindur is filtered out.
        maxdur (float): Time in seconds. Any occurrence whose time is
            > maxdur is filtered out.

    Returns:
        collections.namedtuple:
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    # convert mindur from time seconds to the number of elements
    if mindur is None:
        _mindur = 0
    elif mindur >= 0:
        _mindur = np.ceil(mindur / period)
    else:
        raise TypeError('mindur should be a value >= 0 or None')

    # convert maxdur from time seconds to the number of elements
    if maxdur is None:
        _maxdur = None
    elif maxdur >= 0:
        _maxdur = np.floor(maxdur / period)
    else:
        raise TypeError('maxdur should be a value >= 0 or None')

    # from here on out, all calculations are in number of points, not seconds

    event_lengths = events.stops - events.starts
    if _maxdur is not None:
        condition = ((event_lengths < _mindur) |
                     (event_lengths > _maxdur))
    else:
        condition = (event_lengths < _mindur)

    starts = np.ma.masked_where(condition, events.starts).compressed()
    stops = np.ma.masked_where(condition, events.stops).compressed()

    return RawEvents(starts, stops)


def offset_events(start_offset=None, stop_offset=None):
    """Apply an offset to event start and stops

    Offset the starts and stops of events by the time in seconds
    specified by start_offset and stop_offset

    See Also:
        trouver.transfomations._offset_events

    Args:
        start_offset (float): Time in seconds. Default is None. Value
            must be <= 0
        stop_offset (float): Time in seconds. Default is None. Value
            must be >= 0

    Returns:
        callable: Partial function

    Examples:

    """
    return partial(_offset_events, start_offset=start_offset, stop_offset=stop_offset)


@curry
def _offset_events(events, period, condition_size, start_offset, stop_offset):
    """Apply an offset to event start and stops

    See Also:
        trouver.transfomations._offset_events

    Args:
        events (:obj: `collections.namedtuple` of int):
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops
        period (float): Time in seconds between each data point.
            Requires constant increment data that is uniform across
            all data. (1/Hz = s)
        condition_size (float):
        start_offset: Time in seconds.
        stop_offset: Time in seconds.

    Returns:
        collections.namedtuple:
            trouver.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    if start_offset is None:
        starts = events.starts
    elif start_offset <= 0:
        starts = events.starts + np.floor(start_offset/period).astype(np.int32)
    else:
        raise ValueError('start_offset must be None or <= 0')

    if stop_offset is None:
        stops = events.stops
    elif stop_offset >= 0:
        stops = events.stops + np.ceil(stop_offset/period).astype(np.int32)
    else:
        raise ValueError('stop_offset must be None or >= 0')

    np.clip(starts, 0, condition_size, out=starts)
    np.clip(stops, 0, condition_size, out=stops)

    return RawEvents(starts, stops)


def merge_overlap(events):
    _mask = np.array([False])
    mask = (events.starts[1:] < events.stops[:-1])
    starts = np.ma.masked_where(np.append(_mask, mask), events.starts).compressed()
    stops = np.ma.masked_where(np.append(mask, _mask), events.stops).compressed()

    return RawEvents(starts, stops)

def main():
    input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))
    #test_events = _debounce(input_events, period=1, entry_debounce=2, exit_debounce=0)
    test_events = _filter_durations(input_events, period=1, mindur=2, maxdur=3.1)

if __name__ == '__main__':
    main()
