Quickstart
==========

Setup
-----

.. testsetup:: *

    import numpy as np
    import trouve as tr
    import trouve.transformations as tt
    x = np.array([0, 1, 1, 0, 1, 0])
    example = tr.find_events(x > 0, period=1, name='example')
    example_2 = tr.find_events(x > 0, period=1, name='example_2')

Example events for quickstart.

.. code-block:: python

    >>> import numpy as np
    >>> import trouve as tr
    >>> import trouve.transformations as tt
    >>> x = np.array([0, 1, 1, 0, 1, 0])
    >>> example = tr.find_events(x > 0, period=1, name='example')

Finding Events
--------------

Find all occurrences where the ``numpy.array`` ``x`` is greater than zero. Assume the sample
period is one second.

.. doctest:: find_events

    >>> sample_period = 1 #second
    >>> example = find_events(x > 0, period=sample_period, name='example')
    >>> len(example)
    2

Applying Transformations
------------------------

Transformation functions are applied in the specified order. Each transformation alters
events inplace to avoid making unnecessary copies.

.. doctest:: transformations

    >>> deb = tt.debounce(2, 1)
    >>> offset = tt.offset_events(0, 1)
    >>> cond = x > 0
    >>> deb_first = tr.find_events(cond, period=1,
    ... transformations=[deb, offset])
    >>> deb_first.to_array()
    array([ 0.,  1.,  1.,  1.,  0.,  0.])

.. note:: Order matters with transformations.

Observe how the events change if the offset is applied before debouncing.

.. doctest:: transformations

    >>> offset_first = find_events(cond, period=1, transformations=[offset, deb])
    >>> offset_first.to_array()
    array([ 0.,  1.,  1.,  1.,  1.,  1.])
    >>> offset_first == deb_first
    False


Array Methods
-------------
:any:`Events` provides several methods to produce array representations of events.

``numpy.ndarray`` s via :any:`Events.to_array` .

.. doctest:: arrays

    >>> example.to_array()
    array([ 0.,  1.,  1.,  0.,  1.,  0.])



    >>> example.to_array()
    array([ 0.,  1.,  1.,  0.,  1.,  0.])



    >>> example.to_array()
    array([ 0.,  1.,  1.,  0.,  1.,  0.])

``pandas.Series`` s via :any:`Events.to_series` .

.. doctest:: arrays

    >>> example.to_series()
    0    0.0
    1    1.0
    2    1.0
    3    0.0
    4    1.0
    5    0.0
    Name: example, dtype: float64

Boolean masks via

    >>> example.to_series()
    0    0.0
    1    1.0
    2    1.0
    3    0.0
    4    1.0
    5    0.0
    Name: example, dtype: float64

Boolean masks via

    >>> example.to_series()
    0    0.0
    1    1.0
    2    1.0
    3    0.0
    4    1.0
    5    0.0
    Name: example, dtype: float64

Boolean masks via :any:`Events.to_array` for use with the ``numpy.ma`` module.

.. doctest:: arrays

    >>> example.to_array(1, 0, dtype=np.bool)
    array([ True, False, False,  True, False,  True], dtype=bool)
    >>> x > 0
    array([False,  True,  True, False,  True, False], dtype=bool)

Inspecting Events
-----------------

The ``trouve.Events`` class implements ``__getitem__`` which returns an
:any:`Occurrence` .

.. doctest:: inspection

    >>> first_event = example[0]
    >>> first_event.duration
    2
    >>> x[first_event.slice]
    array([1, 1])

``trouve.Events`` is also an iterable through implementation of both ``__iter__`` and
``__next__``. Every iteration returns an :any:`Occurrence` .

.. doctest:: inspection

    >>> for event in example:
    ...     print(event.duration)
    2
    1

Magic Methods
-------------

``Trouve`` implements several magic methods including:

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
