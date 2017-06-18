Tips and Tricks
===============

Here are some recipes to effectively use Trouve to it's full potential.


.. testsetup:: *

    import numpy as np
    import pandas as pd
    from trouve import find_events
    from trouve.transformations import *

Specify Sample Period with `functools.partial`
----------------------------------------------

If you're looking for multiple events in the same data set, then one shortcut is
to use ``functools.partial`` and add the period to the partial function.
You can then call that function instead of ``find_events`` .

.. doctest:: partial_period

    >>> from functools import partial
    >>> x = np.array([1, 1, 2, 0, 2])
    >>> period = 1
    >>> pfind_events = partial(find_events, period=period)
    >>> events_1 = pfind_events(x == 1)
    >>> events_1.as_array()
    array([ 1.,  1.,  0.,  0.,  0.])
    >>> events_2 = pfind_events(x == 2)
    >>> events_2.to_array()
    array([ 0.,  0.,  1.,  0.,  1.])



    >>> from functools import partial
    >>> x = np.array([1, 1, 2, 0, 2])
    >>> period = 1
    >>> pfind_events = partial(find_events, period=period)
    >>> events_1 = pfind_events(x == 1)
    >>> events_1.as_array()
    array([ 1.,  1.,  0.,  0.,  0.])
    >>> events_2 = pfind_events(x == 2)
    >>> events_2.to_array()
    array([ 0.,  0.,  1.,  0.,  1.])



    >>> from functools import partial
    >>> x = np.array([1, 1, 2, 0, 2])
    >>> period = 1
    >>> pfind_events = partial(find_events, period=period)
    >>> events_1 = pfind_events(x == 1)
    >>> events_1.to_array()
    array([ 1.,  1.,  0.,  0.,  0.])
    >>> events_2 = pfind_events(x == 2)
    >>> events_2.as_array()
    array([ 0.,  0.,  1.,  0.,  1.])



    >>> from functools import partial
    >>> x = np.array([1, 1, 2, 0, 2])
    >>> period = 1
    >>> pfind_events = partial(find_events, period=period)
    >>> events_1 = pfind_events(x == 1)
    >>> events_1.to_array()
    array([ 1.,  1.,  0.,  0.,  0.])
    >>> events_2 = pfind_events(x == 2)
    >>> events_2.as_array()
    array([ 0.,  0.,  1.,  0.,  1.])



    >>> from functools import partial
    >>> x = np.array([1, 1, 2, 0, 2])
    >>> period = 1
    >>> pfind_events = partial(find_events, period=period)
    >>> events_1 = pfind_events(x == 1)
    >>> events_1.as_array()
    array([ 1.,  1.,  0.,  0.,  0.])
    >>> events_2 = pfind_events(x == 2)
    >>> events_2.as_array()
    array([ 0.,  0.,  1.,  0.,  1.])

Multi-parameter Conditional Array
---------------------------------

The condition can be as complicated as necessary. Using multiple inputs and the
ampersand (``&``) or the pipe (``|``). The following example find events where x > 0 and
y == 2, or z <= 1. ``((x > 0) & (y == 2)) | (z <= 1)``

.. note:: When using more than one parameter, you must put each expression in its own parenthesis

.. doctest::

    >>> x = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    >>> y = np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 2])
        >>> z = np.array([2, 2, 2, 3, 3, 0, 3, 3, 3, 3])
        >>> cond = ((x > 0) & (y == 2)) | (z <= 1)
        >>> events = find_events(cond, period=1)
        >>> events.to_array()
        array([ 1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.])


    >>> z = np.array([2, 2, 2, 3, 3, 0, 3, 3, 3, 3])
    >>> cond = ((x > 0) & (y == 2)) | (z <= 1)
    >>> events = find_events(cond, period=1)
    >>> events.as_array()
    array([ 1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.])

Events and the ``numpy.ma`` Module
----------------------------------

The :any:`Events.as_mask` method was developed to integrate directly with ``numpy.ma.MaskedArray``
and ``numpy.ma.masked_where`` . The ``numpy.ma`` module makes things like summing or finding
min/max of arrays based on your condition.

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> mask = events.as_mask()
    >>> np.ma.masked_where(mask, x)
    masked_array(data = [-- 1 -- -- 1 1 -- 1 -- 1],
                 mask = [ True False  True  True False False  True False  True False],
           fill_value = 999999)
    <BLANKLINE>
    >>> masked_x = np.ma.MaskedArray(x, mask)
    >>> masked_x.sum()
    5
    >>> x.sum()
    0


Getting Events into a ``pandas.DataFrame``
------------------------------------------

The ``pandas.DataFrame`` data structure and ``trouve`` fit nicely together. You can loop through
each occurrence and append a statistical description to the dataframe. This is helpful you
your trying to pull features out of time-series data for a machine learning algorithm,
or you want to describe all events found in a data set and then use ``pandas`` idioms to
further process them.

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    >>> y = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> columns = ['duration', 'ave_y_value', 'y_value_at_event_start']
    >>> df = pd.DataFrame(index=pd.RangeIndex(len(events)), columns=columns)
    >>> for i, occurrence in enumerate(events):
    ...     df.iloc[i] = dict(
    ...         duration=occurrence.duration,
    ...         ave_y_value= y[occurrence.slice].mean(),
    ...         y_value_at_event_start=y[occurrence.start]
    ...     )
    >>> df
      duration ave_y_value y_value_at_event_start
    0        1           2                      2
    1        2         4.5                      5
    2        1           2                      2
    3        1           0                      0

Finding Inverse Events
----------------------

If you're interested in when events aren't active, then you can use the inverse of the condition.
This would be helpful if you wanted to know the average, min, or max time between events.

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        >>> cond = x == 1
        >>> events = find_events(cond, period=1)
        >>> inv_events = find_events(~cond, period=1)
        >>> events.as_array()
        array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
        >>> inv_events.to_array()
        array([ 1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.])


        >>> cond = x == 1
        >>> events = find_events(cond, period=1)
        >>> inv_events = find_events(~cond, period=1)
        >>> events.as_array()
        array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
        >>> inv_events.to_array()
        array([ 1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.])


    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> inv_events = find_events(~cond, period=1)
    >>> events.as_array()
    array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    >>> inv_events.as_array()
    array([ 1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.])

``Events.durations`` Tips
-------------------------

Total time in seconds events are active.

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> events.durations.sum()
    5

Occurrence rate: Occurrences/second

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> len(events) / events.durations.sum() # doctest: +SKIP
    0.8

Creating a histogram of event lengths

.. doctest::

    >>> x = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    >>> cond = x == 1
    >>> events = find_events(cond, period=1)
    >>> np.histogram(events.durations, [0, 0.5, 1, 1.5, 2, 2.5])
    (array([0, 0, 3, 0, 1], dtype=int64), array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5]))
