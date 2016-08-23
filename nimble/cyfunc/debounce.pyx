import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def debounce(np.ndarray[np.int32_t, ndim=1] starts,
             np.ndarray[np.int32_t, ndim=1] stops,
             double entrydb, double exitdb):

        cdef Py_ssize_t index
        cdef double event_length, reset_length, next_index
        cdef np.ndarray[np.int_t] start_mask = np.zeros(starts.shape[0], dtype=np.int)
        cdef np.ndarray[np.int_t] stop_mask = np.zeros(stops.shape[0], dtype=np.int)
        event_active = False

        for index in range(starts.shape[0]):
            event_length = stops[index] - starts[index]
            
            if index < (starts.shape[0]-1):
                reset_length = starts[index + 1] - stops[index]
            else: 
                reset_length = 0
        
            if event_active:
                pass
            elif not event_active and event_length >= entrydb:
                event_active = True
            elif not event_active and event_length < entrydb:
                start_mask[index] = 1
                stop_mask[index] = 1
            else:
                raise ValueError

            if not event_active or reset_length == 0:
                pass
            elif event_active and reset_length >= exitdb:
                event_active = False
            elif event_active and reset_length < exitdb:
                start_mask[index + 1] = 1
                stop_mask[index] = 1
            else:
                raise ValueError

        starts = np.ma.masked_where(start_mask > 0, starts).compressed()
        stops = np.ma.masked_where(stop_mask > 0, stops).compressed()

        return starts, stops
