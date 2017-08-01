Change Log
==========

0.6.0
-----

* Apply ``toolz.curry`` to :any:``trouve.find_events``

0.5.1
-----

* Fixed issue where deprecated methods in 0.5.0 didn't issue deprecation warnings

0.5.0
-----

**Events** methods

* Deprecate ``Events.as_array``, use :any:`Events.to_array`
* Deprecate ``Events.as_series``, use :any:`Events.to_series`
* Deprecate ``Events.as_mask``, use :any:`Events.to_array` with ``inactive_value=1``, ``ative_values=`` and ``dtype=np.bool``

**Transformations**

* Deprecate passing transformation functions as *args to :any:`trouve.find_events`. Pass them to the explicit transformations keyword arguments