import numpy as np


class Event(object):
    def __init__(self, condition, entry_debounce=0, exit_debounce=0,
                 sample_rate=1):
        self.condition = condition
        self.entry_debounce = entry_debounce
        self.exit_debounce = exit_debounce
        self.sample_rate = sample_rate  # Assumes univariate time series
        #TODO - work out strategy for multivariate data
        self.starts, self.stops = self._apply_parameters()

    @property
    def size(self):
        """
        Return the number of events found.
        """
        return len(self.starts)

    @property
    def as_array(self):
        """
        Return the found events as a numpy array of 0's and 1'sample_rate
        """
        #TODO - Cache this value?
        #import pdb; pdb.set_trace()
        output = np.zeros(self.condition.size, dtype='i1')
        for start, stop in zip(self.starts, self.stops):
            output[start:stop] = 1
        return output

    def _apply_parameters(self):
        starts, stops = self._apply_condition()

        return starts, stops

    def _apply_condition(self):
        """
        Apply initial masking conditions
        """
        mask = (self.condition > 0).view('i1')
        slice_index = np.arange(mask.size+1)

        if mask[0] == 0:
            to_begin = np.array([0])
        else:
            to_begin = np.array([1])

        if mask[-1] == 0:
            to_end = np.array([0])
        else:
            to_end = np.array([-1])

        deltas = np.ediff1d(mask, to_begin=to_begin, to_end=to_end)
        #import pdb; pdb.set_trace()
        starts = np.ma.masked_where(deltas < 1, slice_index).compressed()
        stops = np.ma.masked_where(deltas > -1, slice_index).compressed()

        return starts, stops