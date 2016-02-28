from unittest import TestCase

import numpy as np
import numpy.testing as npt

from nimble import Events


class TestClassIterable(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition)

    def test_start_stop_loc(self):
        test_starts = []
        test_stops = []

        for event in self.events:
            test_starts.append(event.start)
            test_stops.append(event.stop)

        npt.assert_array_equal(self.events.starts, test_starts)
        npt.assert_array_equal(self.events.stops, test_stops)

    def test_durations(self):
        test_durations = []
        for event in self.events:
            test_durations.append(event.duration)

        npt.assert_array_equal(self.events.durations, test_durations)

