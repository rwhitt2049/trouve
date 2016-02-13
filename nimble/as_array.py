def as_array(starts, stops, mask_array, true_values=1):
    for start, stop in zip(starts, stops):
        mask_array[start:stop] = 1 * true_values

    return mask_array
