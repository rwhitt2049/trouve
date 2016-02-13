import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def as_array(np.ndarray[long, ndim=1] starts, np.ndarray[long, ndim=1] stops,
             np.ndarray[np.float64_t, ndim=1] mask_array, long true_values=1):

    cdef Py_ssize_t index
    cdef double start, stop
    for index in range(starts.shape[0]):
        start = starts[index]
        stop = stops[index]
        mask_array[start:stop] = true_values
    return mask_array