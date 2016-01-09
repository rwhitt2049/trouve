import numpy as np


class Event(object):
    def __init__(self, condition, sample_rate=1,
                 entry_debounce=0, exit_debounce=0):

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

        output = np.zeros(self.condition.size, dtype='i1')
        for start, stop in zip(self.starts, self.stops):
            output[start:stop] = 1
        return output

    def _apply_parameters(self):
        starts, stops = self._apply_condition()
        
        if len(starts) and (self.entry_debounce or self.exit_debounce):
            starts, stops = self._apply_debounce(starts, stops)

        return starts, stops

    def _apply_condition(self):
        """
        Apply initial masking conditions
        """
        mask = (self.condition > 0).view('i1')
        slice_index = np.arange(mask.size+1, dtype='int32')

        if mask[0] == 0:
            to_begin = np.array([0], dtype='i1')
        else:
            to_begin = np.array([1], dtype='i1')

        if mask[-1] == 0:
            to_end = np.array([0], dtype='i1')
        else:
            to_end = np.array([-1], dtype='i1')

        deltas = np.ediff1d(mask, to_begin=to_begin, to_end=to_end)

        starts = np.ma.masked_where(deltas < 1, slice_index).compressed()
        stops = np.ma.masked_where(deltas > -1, slice_index).compressed()

        return starts, stops
        
    def _apply_debounce(self, starts, stops):
        #n_events = len(starts)
        start_mask = np.zeros(starts.size)
        stop_mask = np.zeros(stops.size)
        event_started = False

        for id in np.arange(starts.size):
            event_length = stops[id] - starts[id]

            try:
                reset_length = starts[id+1] - stops[id]
            except IndexError:
                reset_length = None

            if event_length >= self.entry_debounce and not event_started:
                event_started = True
            elif not event_started and event_length < self.entry_debounce:
                start_mask[id] = 1
                stop_mask[id] = 1
            elif event_started:
                pass
            else:
                raise ValueError

            if not event_started or reset_length is None:
                pass
            elif event_started and reset_length >= self.exit_debounce:
                event_started = False
            elif event_started and reset_length < self.exit_debounce:
                start_mask[id + 1] = 1
                stop_mask[id] = 1
            else:
                raise ValueError

        starts = np.ma.masked_where(start_mask > 0, starts).compressed()
        stops = np.ma.masked_where(stop_mask > 0, stops).compressed()

        return starts, stops