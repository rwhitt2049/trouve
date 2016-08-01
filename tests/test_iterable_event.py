from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from nimble import Events


class TestClassIterable(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition, sample_period=0.5).find()

    def test_start_stop_loc(self):
        test_starts = []
        test_stops = []

        for event in self.events:
            test_starts.append(event.istart)
            test_stops.append(event.istop)

        npt.assert_array_equal([1, 7, 10], test_starts)
        npt.assert_array_equal([3, 8, 11], test_stops)

    def test_durations(self):
        validation_durations = [1.5, 1, 1]
        test_durations = []
        for event in self.events:
            test_durations.append(event.iduration)

        npt.assert_array_equal(validation_durations, test_durations)

    def test_index(self):
        validation_index = [0, 1, 2]
        test_index = []
        for event in self.events:
            test_index.append(event.i)

        npt.assert_array_equal(validation_index, test_index)

if __name__ == '__main__':
    main()
