from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from trouve.events import Events


class TestClassIterable(TestCase):
    def setUp(self):
        array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        starts = np.array([1, 7, 10])
        stops = np.array([4, 9, 12])
        period = 1
        self.events = Events(starts, stops, period, 'control', array.size)

    def test_start_stop_loc(self):
        test_starts = []
        test_stops = []

        for event in self.events:
            test_starts.append(event.start)
            test_stops.append(event.stop)

        npt.assert_array_equal([1, 7, 10], test_starts)
        npt.assert_array_equal([3, 8, 11], test_stops)

    def test_durations(self):
        validation_durations = [3, 2, 2]
        test_durations = []
        for event in self.events:
            test_durations.append(event.duration)

        npt.assert_array_equal(validation_durations, test_durations)


if __name__ == '__main__':
    main()
