from functools import partial

import numpy as np


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
            (event active >= ``activate_debounce``)
        deactivate_debounce (``float``): Default is ``None``.
            Default value does not apply an deactivate_debounce. Maximum time
            in seconds an occurrence must be inactive to deactivate an event.
            (event inactive >= ``deactivate_debounce``)

    Returns:
        ``callable``: Partial function
    
    Examples:
        >>> import trouve as tr
        >>> import trouve.transformations as tt
        >>> import numpy as np
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 3, 3])
        >>> condition = y > 2
        >>> events = tr.find_events(condition, period=1)
        >>> deb = tt.debounce(2, 2)
        >>> trans_events = tr.find_events(condition, period=1, transformations=[deb])
        >>> events.to_array()  # doctest: +SKIP
        array([ 0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.])
        >>> trans_events.to_array()  # doctest: +SKIP
        array([ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.])
        
    Raises:
        ``ValueError``: If ``activate_debounce`` or ``deactivate_debounce`` < 0

    """
    a_deb = 0 if activate_debounce is None else activate_debounce
    d_deb = 0 if deactivate_debounce is None else deactivate_debounce

    if a_deb < 0:
        raise ValueError('activate_debounce must be a value >= 0 or None')

    if d_deb < 0:
        raise ValueError('deactivate_debounce must be a value >= 0 or None')

    return partial(_debounce, activate_debounce=a_deb, deactivate_debounce=d_deb)


def _debounce(events, activate_debounce, deactivate_debounce):
    """Debounce activate and deactivation of events

    See Also:
        :any:`trouve.transfomations.debounce`

    Args:
        events (:any:`trouve.events.Events`): Events to apply debounce to
        activate_debounce (``float``): Time in seconds.
        deactivate_debounce (``float``): Time in seconds.

    Returns:
        :any:`trouve.events.Events`: Events with debounces applied

    """
    # convert activate_debounce from time seconds to the number of elements
    activate_debounce_ = np.ceil(activate_debounce / events._period)
    deactivate_debounce_ = np.ceil(deactivate_debounce / events._period)

    start_mask = np.zeros(events._starts.size, dtype=np.bool)
    stop_mask = np.zeros(events._stops.size, dtype=np.bool)
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
        elif not event_active and event_length >= activate_debounce_:
            event_active = True
        elif not event_active and event_length < activate_debounce_:
            start_mask[index] = True
            stop_mask[index] = True
        else:
            raise ValueError

        # determine whether or not an active event stop is
        # active relative to reset_length
        if not event_active or reset_length is None:
            pass
        elif event_active and reset_length >= deactivate_debounce_:
            event_active = False
        elif event_active and reset_length < deactivate_debounce_:
            start_mask[index + 1] = True
            stop_mask[index] = True
        else:
            raise ValueError

    events._starts = np.ma.masked_where(start_mask, events._starts).compressed()
    events._stops = np.ma.masked_where(stop_mask, events._stops).compressed()

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
        ``callable``: Partial function
        
    Raises:
        ``ValueError``: If min_duration or max_duration is < 0

    Examples:
        >>> import trouve as tr
        >>> import trouve.transformations as tt
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 3, 3])
        >>> condition = y > 2
        >>> events = tr.find_events(condition, period=1)
        >>> filt_dur = filter_durations(1.5, 2.5)
        >>> trans_events = tr.find_events(condition, period=1, transformations=[filt_dur])
        >>> events.to_array()  # doctest: +SKIP
        array([ 0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.])
        >>> trans_events.to_array()  # doctest: +SKIP
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.])

    """
    min_dur = 0 if min_duration is None else min_duration
    max_dur = np.nan if max_duration is None else max_duration

    if min_dur < 0:
        raise ValueError('min_duration must be >= 0 or None')

    if max_dur < 0:
        raise ValueError('max_duration must be >= 0 or None')

    return partial(_filter_durations, min_duration=min_dur, max_duration=max_dur)


def _filter_durations(events, min_duration, max_duration):
    """Filter out events based  on duration

    See Also:
        :any:`trouve.transfomations.filter_durations`

    Args:
        events(:any:`trouve.events.Events`): Events to apply duration filter to.
        min_duration (float): Time in seconds. Any occurrence whose time is
            < mindur is filtered out.
        max_duration (float): Time in seconds. Any occurrence whose time is
            > maxdur is filtered out.

    Returns:
        :any:`trouve.events.Events`: Events with duration filters applied

    """
    # convert time in seconds to the number of elements
    min_duration_ = np.ceil(min_duration / events._period)
    max_duration_ = np.floor(max_duration / events._period)

    # from here on out, all calculations are in number of points, not seconds

    event_lengths = events._stops - events._starts
    if not np.isnan(max_duration_):
        condition = ((event_lengths < min_duration_) |
                     (event_lengths > max_duration_))
    else:
        condition = (event_lengths < min_duration_)

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
        ``callable``: Partial function
        
    Raises:
        ``ValueError``: If ``start_offset`` > 0 or ``stop_offset`` < 0

    Examples:
        >>> import trouve as tr
        >>> import trouve.transformations as tt
        >>> y = np.array([2, 2, 2, 3, 4, 5, 2, 2, 2])
        >>> condition = y > 2
        >>> events = tr.find_events(condition, period=1)
        >>> offset = tt.offset_events(-1, 1)
        >>> trans_events = tr.find_events(condition, period=1, transformations=[offset])
        >>> events.to_array()  # doctest: +SKIP
        array([ 0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.])
        >>> trans_events.to_array()  # doctest: +SKIP
        array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])

    """
    start_offset_ = 0 if start_offset is None else start_offset
    stop_offset_ = 0 if stop_offset is None else stop_offset

    if start_offset_ > 0:
        raise ValueError('start_offset must be None or <= 0')

    if stop_offset_ < 0:
        raise ValueError('stop_offset must be None or >= 0')

    return partial(_offset_events, start_offset=start_offset_, stop_offset=stop_offset_)


def _offset_events(events, start_offset, stop_offset):
    """Apply an offset to event start and stops

    See Also:
        :any:`trouve.transfomations.offset_events`

    Args:
        events (:any:`trouve.events.Events`): Events to apply offsets.
        start_offset: Time in seconds.
        stop_offset: Time in seconds.

    Returns:
        :any:`trouve.events.Events`: Events with offsets applied.

    """
    starts = events._starts + np.floor(start_offset / events._period).astype(np.int32)
    stops = events._stops + np.ceil(stop_offset / events._period).astype(np.int32)

    np.clip(starts, 0, events._condition_size, out=events._starts)
    np.clip(stops, 0, events._condition_size, out=events._stops)

    return events


def merge_overlap(events):
    """Merge any events that overlap

    Some events such as offset_events can cause events to overlap. If
    this transformation is applied, any events that overlap will become
    one contiguous event.

    Args:
        events (:any:`trouve.events.Events`):

    Returns:
        :any:`trouve.events.Events`: Any overlapping events merged into one event.

    Examples:
        >>> import trouve as tr
        >>> import trouve.transformations as tt
        >>> y = np.array([2, 3, 2, 3, 4, 5, 2, 2, 2])
        >>> condition = y > 2
        >>> offset = tt.offset_events(-1, 1)
        >>> events = tr.find_events(condition, period=1, transformations=[offset])
        >>> merged_events = tr.find_events(condition,  period=1,
        ... transformations=[offset, merge_overlap])
        >>> events.to_array()  # doctest: +SKIP
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])
        >>> merged_events.to_array()  # doctest: +SKIP
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.])
        >>> len(events)
        2
        >>> len(merged_events)
        1

    """
    if events:
        init_mask = np.array([False])
        mask = (events._starts[1:] <= events._stops[:-1])
        events._starts = np.ma.masked_where(np.append(init_mask, mask), events._starts).compressed()
        events._stops = np.ma.masked_where(np.append(mask, init_mask), events._stops).compressed()

    return events
