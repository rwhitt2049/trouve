import numpy as np


def debounce(starts, stops, entry_debounce, exit_debounce):
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
        elif not event_active and event_length >= entry_debounce:
            event_active = True
        elif not event_active and event_length < entry_debounce:
            start_mask[index] = 1
            stop_mask[index] = 1
        else:
            raise ValueError

        if not event_active or reset_length is None:
            pass
        elif event_active and reset_length >= exit_debounce:
            event_active = False
        elif event_active and reset_length < exit_debounce:
            start_mask[index + 1] = 1
            stop_mask[index] = 1
        else:
            raise ValueError

    starts = np.ma.masked_where(start_mask > 0, starts).compressed()
    stops = np.ma.masked_where(stop_mask > 0, stops).compressed()

    return starts, stops