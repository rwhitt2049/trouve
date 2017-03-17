from functools import partial

import numpy as np
from toolz import curry

__all__ = ['debounce', 'filter_durations', 'offset_events', 'merge_overlap']


def debounce(activate_debounce=None, deactivate_debounce=None):
    """Debounce activation and deactivation of events

    Find an occurrence that is active for time >= activate_debounce and
    activate event. Deactivate event only after an occurrence is found
    that is inactive for time >= to deactivate_debounce. Filter out all events
    that fall outside of these bounds. This function is used to prevent
    short duration occurrences from activating or deactivating longer
    events. See mechanical debounce in mechanical switches and relays
    for a similar concept.

    Args:
        activate_debounce (``float``): Default is ``None``.
            Default value does not apply an activate_debounce. Minimum time
            in seconds an occurrence must be active to activate an event.
            (>= activate_debounce)
        deactivate_debounce (``float``): Default is ``None``.
            Default value does not apply an deactivate_debounce. Maximum time
            in seconds an occurrence must be inactive to deactivate an event.
            (>= deactivate_debounce)

    Returns:
        ``callable``: Partial function

    Examples:
        >>> from trouve import find_events
        >>> from trouve.transformations import debounce
        >>> import numpy as np
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 3, 3])
        >>> condition = y > 2
        >>> test_events = find_events(condition, period=1)
        >>> deb = debounce(2, 2)
        >>> example_events = find_events(condition, deb, period=1)
        >>> test_events.as_array()
        array([ 0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.])
        >>> example_events.as_array()
        array([ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.])

    """
    return partial(_debounce, entry_debounce=activate_debounce, exit_debounce=deactivate_debounce)


def _debounce(events, entry_debounce, exit_debounce):
    """Debounce activate and deactivation of events

    See Also:
        trouve.transfomations.debounce

    Args:
        events(:obj: `collections.namedtuple` of int):
            trouve.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops
        period (float): Time in seconds between each data point.
            Requires constant increment data that is uniform across
            all data. (1/Hz = s)
        entry_debounce (float): Time in seconds.
        exit_debounce (float): Time in seconds.

    Returns:
        collections.namedtuple:
            trouve.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    # convert entry_debounce from time seconds to the number of elements
    if entry_debounce is None:
        _entry_debounce = 0
    elif entry_debounce >= 0:
        _entry_debounce = np.ceil(entry_debounce / events._period)
    else:
        raise TypeError('entry_debounce should be a value >= 0 or None')

    # convert entry_debounce from time seconds to the number of elements
    if exit_debounce is None:
        _exit_debounce = 0
    elif exit_debounce >= 0:
        _exit_debounce = np.ceil(exit_debounce / events._period)
    else:
        raise TypeError('exit_debounce should be a value >= 0 or None')

    # from here on out, all calculations are in number of points, not seconds

    start_mask = np.zeros(events._starts.size)
    stop_mask = np.zeros(events._stops.size)
    event_active = False

    for index in np.arange(events._starts.size):
        # get time of the event
        event_length = events._stops[index] - events._starts[index]

        # get time to the next event
        try:
            reset_length = events._starts[index + 1] - events._stops[index]
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

    events._starts = np.ma.masked_where(start_mask > 0, events._starts).compressed()
    events._stops = np.ma.masked_where(stop_mask > 0, events._stops).compressed()

    return events


def filter_durations(min_duration=None, max_duration=None):
    """Filter out durations based on length of time active

    Filter out events that are < min_duration and > max_duration
    (time in seconds).

    Args:
        min_duration (``float``): Default is ``None``.
            Default value does not apply a min_duration filter.
            Filter out events whose duration in seconds is < min_duration.
        max_duration (``float``): Default is ``None``.
            Default value does not apply a max_duration filter. Filter
            out events whose duration in seconds is > max_duration.

    Returns:
        ``callable``:
            Partial function

    Examples:
        >>> from trouve import find_events
        >>> from trouve.transformations import filter_durations
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 3, 3])
        >>> condition = y > 2
        >>> test_events = find_events(condition, period=1)
        >>> filt_dur = filter_durations(1.5, 2.5)
        >>> example_events = find_events(condition, filt_dur, period=1)
        >>> test_events.as_array()
        array([ 0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.])
        >>> example_events.as_array()
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.])

    """
    return partial(_filter_durations, mindur=min_duration, maxdur=max_duration)


def _filter_durations(events, mindur, maxdur):
    """Filter out events based  on duration

    See Also:
        trouve.transfomations.filter_durations

    Args:
        events (:obj: `collections.namedtuple` of int):
            trouve.transformations.RawEvents(starts, stops)
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
            trouve.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    # convert mindur from time seconds to the number of elements
    if mindur is None:
        _mindur = 0
    elif mindur >= 0:
        _mindur = np.ceil(mindur / events._period)
    else:
        raise TypeError('mindur should be a value >= 0 or None')

    # convert maxdur from time seconds to the number of elements
    if maxdur is None:
        _maxdur = None
    elif maxdur >= 0:
        _maxdur = np.floor(maxdur / events._period)
    else:
        raise TypeError('maxdur should be a value >= 0 or None')

    # from here on out, all calculations are in number of points, not seconds

    event_lengths = events._stops - events._starts
    if _maxdur is not None:
        condition = ((event_lengths < _mindur) |
                     (event_lengths > _maxdur))
    else:
        condition = (event_lengths < _mindur)

    events._starts = np.ma.masked_where(condition, events._starts).compressed()
    events._stops = np.ma.masked_where(condition, events._stops).compressed()

    return events


def offset_events(start_offset=None, stop_offset=None):
    """Apply an offset to event start and stops

    Offset the starts and stops of events by the time in seconds
    specified by start_offset and stop_offset.

    Args:
        start_offset (``float``): Default is ``None``.
            Time in seconds to offset event starts. Value must be <= 0.
        stop_offset (float): Default is ``None``.
            Time in seconds to offset event stops. Value must be >= 0.

    Returns:
        ``callable``:
            Partial function

    Examples:
        >>> from trouve import find_events
        >>> from trouve.transformations import offset_events
        >>> y = np.array([2, 2, 2, 3, 4, 5, 2, 2, 2])
        >>> condition = y > 2
        >>> test_events = find_events(condition, period=1)
        >>> offset = offset_events(-1, 1)
        >>> example_events = find_events(condition, offset, period=1)
        >>> test_events.as_array()
        array([ 0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.])
        >>> example_events.as_array()
        array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])

    """
    return partial(_offset_events, start_offset=start_offset, stop_offset=stop_offset)


def _offset_events(events, start_offset, stop_offset):
    """Apply an offset to event start and stops

    See Also:
        trouve.transfomations.offset_events

    Args:
        events (:obj: `collections.namedtuple` of int):
            trouve.transformations.RawEvents(starts, stops)
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
            trouve.transformations.RawEvents(starts, stops)
            starts (np.ndarry of int): Index values for event starts
            stops (np.ndarry of int): Index values for event stops

    """
    if start_offset is None:
        starts = events._starts
    elif start_offset <= 0:
        starts = events._starts + np.floor(start_offset/events._period).astype(np.int32)
    else:
        raise ValueError('start_offset must be None or <= 0')

    if stop_offset is None:
        stops = events._stops
    elif stop_offset >= 0:
        stops = events._stops + np.ceil(stop_offset/events._period).astype(np.int32)
    else:
        raise ValueError('stop_offset must be None or >= 0')

    np.clip(starts, 0, events._condition_size, out=events._starts)
    np.clip(stops, 0, events._condition_size, out=events._stops)

    return events


def merge_overlap(events):
    """Merge any events that overlap

    Some events such as offset_events can cause events to overlap. If
    this transformation is applied, any events that overlap will become
    one contiguous event.

    Args:
        events (:any:`Events`):

    Returns:
        :any:`Events`:

    Examples:
        >>> from trouve import find_events
        >>> from trouve.transformations import offset_events, merge_overlap
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 2, 2])
        >>> condition = y > 2
        >>> offset = offset_events(-1, 1)
        >>> test_events = find_events(condition, offset, period=1)
        >>> example_events = find_events(condition, offset,
        ... merge_overlap, period=1)
        >>> test_events.as_array()
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])
        >>> example_events.as_array()
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])
        >>> len(test_events)
        2
        >>> len(example_events)
        1

    """
    _mask = np.array([False])
    mask = (events._starts[1:] <= events._stops[:-1])
    events._starts = np.ma.masked_where(np.append(_mask, mask),
                                        events._starts).compressed()
    events._stops = np.ma.masked_where(np.append(mask, _mask),
                                       events._stops).compressed()

    return events


def main():
    input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))
    #test_events = _debounce(input_events, period=1, entry_debounce=2, exit_debounce=0)
    test_events = _filter_durations(input_events, period=1, mindur=2, maxdur=3.1)

if __name__ == '__main__':
    main()
