Quickstart
==========

Setup
-----

.. testsetup:: *

    import numpy as np
    from trouver import find_events, debounce, merge_overlap, offset_events, filter_durations
    x = np.array([0, 1, 1, 0, 1, 0])
    example = find_events(x > 0, 1, name='example')
    example_2 = find_events(x > 0, 1, name='example_2')

Example events for quickstart.

.. code-block:: python

    >>> import numpy as np
    >>> from trouver import find_events
    >>> from trouver.transformation import *
    >>> x = np.array([0, 1, 1, 0, 1, 0])
    >>> example = find_events(x > 0, 1, name='example')

Finding Events
--------------

Find all occurrences where the ``numpy.array`` ``x`` is greater than zero. Assume the sample
period is one second.

.. doctest:: find_events

    >>> sample_period = 1 #second
    >>> example = find_events(x > 0, sample_period, name='example')
    >>> len(example)
    2

Applying Transformations
------------------------

Transformation functions are applied in the specified order. Each transformation alters
events inplace to avoid making unnecessary copies.

.. doctest:: transformations

    >>> deb = debounce(2, 1)
    >>> offset = offset_events(0, 1)
    >>> cond = x > 0
    >>> deb_first = find_events(cond, 1, deb, offset, name='example')
    >>> deb_first.as_array()
    array([ 0.,  1.,  1.,  1.,  0.,  0.])

.. note:: Order matters with transformations.

Observe how the events change if the offset is applied before debouncing.

.. doctest:: transformations

    >>> offset_first = find_events(cond, 1, offset, deb, name='example')
    >>> offset_first.as_array()
    array([ 0.,  1.,  1.,  1.,  1.,  1.])
    >>> offset_first == deb_first
    False


Array Methods
-------------
:any:`Events` provides several methods to produce array representations of events.

``numpy.ndarray`` s via :any:`Events.as_array` .

.. doctest:: arrays

    >>> example.as_array()
    array([ 0.,  1.,  1.,  0.,  1.,  0.])

``pandas.Series`` s via :any:`Events.as_series` .

.. doctest:: arrays

    >>> example.as_series()
    0    0.0
    1    1.0
    2    1.0
    3    0.0
    4    1.0
    5    0.0
    Name: example, dtype: float64

Boolean masks via :any:`Events.as_mask` for use with the ``numpy.ma.`` module.

.. doctest:: arrays

    >>> example.as_mask()
    array([ True, False, False,  True, False,  True], dtype=bool)
    >>> x > 0
    array([False,  True,  True, False,  True, False], dtype=bool)

.. note:: Identified occurrences return as ``False`` from ``Events.as_mask``. This is done as a convenience for working with the ``numpy.ma`` module.


Inspecting Events
-----------------

The ``trouver.Events`` class implements ``__getitem__`` which returns an
:any:`Occurrence` .

.. doctest:: inspection

    >>> first_event = example[0]
    >>> first_event.duration
    2
    >>> x[first_event.slice]
    array([1, 1])

``trouver.Events`` is also an iterable through implementation of both ``__iter__`` and
``__next__``. Every iteration returns an :any:`Occurrence` .

.. doctest:: inspection

    >>> for event in example:
    ...     print(event.duration)
    2
    1

Magic Methods
-------------

``Trouver`` implements several magic methods including:

``__len__`` for determining the number of events found using ``len``.

.. doctest:: magic

    >>> len(example)
    2

``__str__`` for printing a summary of the events with ``print``.

.. doctest:: magic

    >>> print(example)
    example
    Number of events: 2
    Min, Max, Mean Duration: 1.000s, 2.000s, 1.500s

``__eq__`` for determining if two events are equal.

.. doctest:: magic

    >>> example == example_2
    True

.. note:: Equality compares ``_starts``, ``_stops``, ``_period`` and ``_condition_size`` of both ``Event``s. The event ``name`` does **not** have to be the same for both events.

``__repr__`` for help with trouble-shooting using ``repr``.

.. doctest:: magic

    >>> repr(example)
    "Events(_starts=array([1, 4]), _stops=array([3, 5]), _period=1, name='example', _condition_size=6)"
