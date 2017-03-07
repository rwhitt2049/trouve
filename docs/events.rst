Events
======

The primary function of ``trouve`` is to find events in time-series data and apply
functional transformations in a specified order. The main function is :any:`find_events`.
This function takes in a condtional ``bool`` and then returns the class :any:`Events`.
The :any:`Events` class finds each distinct occurrence and records it's start and stop
index value. These values then allow a user to inspect each event in a Pythonic manner.

.. autofunction:: trouve.find_events.find_events

.. autoclass:: trouve.events.Events
    :members:
    :exclude-members: __init__, __hash__, __weakref__
    :member-order: bysource
    :special-members:

.. autoclass:: trouve.events.Occurrence

``trouve.events.Occurrence`` is a ``collections.namedtuple`` that is returned by both
:any:`Events.__getitem__` and :any:`Events.__next__`

    Parameters:
        * start (``int``): Index of the start of the occurrence
        * stop (``int``): Index of the stop of the occurrence
        * slice (``slice``): ``slice`` object for the entire occurrence
        * duration (``float``): Duration in seconds of the occurrence

    Examples:

    .. doctest:: python

        >>> import numpy as np
        >>> from trouve import find_events
        >>> x = np.array([0, 1, 1, 0, 1, 0])
        >>> example = find_events(x, 1, name='example')
        >>> first_event = example[0]
        >>> print(first_event)
        Occurrence(start=1, stop=2, slice=slice(1, 3, None), duration=2)
        >>> first_event.start
        1
        >>> x[first_event.slice]
        array([1, 1])
