Trouve
======

A package to search for events in time-series data that match a boolean condition. Various
transformation functions are built in to filter and alter events.

See Trouve's documentation at https://trouve.readthedocs.io

Install
-------
``trouve`` is on the Python Package Index (PyPI):

::

   pip install trouve

Dependencies
------------

* ``numpy``

* ``pandas``

* ``toolz``

Example
-------

This finds events in a short sample of 1Hz, time-series data and filters out events based
on their duration

.. code-block:: python

   >>> import numpy as np
   >>> from trouve import find_events
   >>> import trouve.transformations as tt
   >>> x = np.array([1, 2, 2, 2, 0, 1, 2, 0, 2, 2])
   >>> period = 1 # period = 1 / sample_rate
   >>> duration_filter = tt.filter_durations(2, 3)
   >>> events = find_events(x == 2, duration_filter, period=1)
   >>> len(events)
   2
   >>> events.as_array()
   array([ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.])
