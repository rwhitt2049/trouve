import numpy as np


class Event(object):
    def __init__(self, condition, sample_rate=1,
                 entry_debounce=0, exit_debounce=0,
                 min_event_length=0, max_event_length=None):

        self.condition = condition
        self.sample_rate = sample_rate  # Assumes univariate time series
        self.entry_debounce = entry_debounce
        self.exit_debounce = exit_debounce
        self.min_event_length = min_event_length
        self.max_event_length = max_event_length

        self.starts, self.stops = self._apply_parameters()
        # TODO - work out strategy for multivariate data
        # TODO - potentially just return a tuple of (start, stop) values
        # TODO Replace len()'s with x.size, these are all numpy arrays

    @property
    def size(self):
        """
        Return the number of events found.
        """
        # TODO  return self.starts.size instead
        return len(self.starts)

    @property
    def as_array(self):
        """
        Return the found events as a numpy array of 0's and 1'sample_rate
        """
        # TODO - Cache this value? or make it a method (better option)

        output = np.zeros(self.condition.size, dtype='i1')
        for start, stop in zip(self.starts, self.stops):
            output[start:stop] = 1
        return output

    def _apply_parameters(self):
        starts, stops = self._apply_condition()

        if len(starts) and (self.entry_debounce or self.exit_debounce):
            starts, stops = self._apply_debounce(starts, stops)

        if len(starts) and (self.min_event_length or self.max_event_length):
            starts, stops = self._apply_event_length_filter(starts, stops)

        return starts, stops

    def _apply_condition(self):
        """
        Apply initial masking conditions
        """
        mask = (self.condition > 0).view('i1')
        slice_index = np.arange(mask.size + 1, dtype='int32')

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
        """ Apply debounce paramaters"""
        start_mask = np.zeros(starts.size)
        stop_mask = np.zeros(stops.size)
        event_active = False

        for index in np.arange(starts.size):
            event_length = stops[index] - starts[index]

            try:
                reset_length = starts[index + 1] - stops[index]
            except IndexError:
                reset_length = None

            if event_active:
                pass
            elif not event_active and event_length >= self.entry_debounce:
                event_active = True
            elif not event_active and event_length < self.entry_debounce:
                start_mask[index] = 1
                stop_mask[index] = 1
            else:
                raise ValueError

            if not event_active or reset_length is None:
                pass
            elif event_active and reset_length >= self.exit_debounce:
                event_active = False
            elif event_active and reset_length < self.exit_debounce:
                start_mask[index + 1] = 1
                stop_mask[index] = 1
            else:
                raise ValueError

        starts = np.ma.masked_where(start_mask > 0, starts).compressed()
        stops = np.ma.masked_where(stop_mask > 0, stops).compressed()

        return starts, stops

    def _apply_event_length_filter(self, starts, stops):
        event_lengths = stops - starts

        if not self.max_event_length:
            condition = (event_lengths < self.min_event_length)
        elif self.min_event_length >= 0 and self.max_event_length > 0:
            condition = ((event_lengths < self.min_event_length) |
                         (event_lengths > self.max_event_length))
        else:
            raise ValueError

        starts = np.ma.masked_where(condition, starts).compressed()
        stops = np.ma.masked_where(condition, stops).compressed()

        return starts, stops
