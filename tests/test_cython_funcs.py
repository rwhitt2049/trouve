from unittest import TestCase
import numpy as np
import numpy.testing as npt
from nimble import Events
import nimble.cyfunc.debounce as cy


class TestAsArrayMethod(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
        events = Events((conditional_array > 0))
        self.starts, self.stops = events.starts, events.stops

    def test_event_entry_debounce_cython(self):
        starts, stops = cy.debounce(self.starts, self.stops, 2, 0)
        starts_validation = [2, 6]
        stops_validation = [4, 9]

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_event_exit_debounce(self):
        starts, stops = cy.debounce(self.starts, self.stops, 0, 2)
        starts_validation = [2, 6]
        stops_validation = [4, 11]

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_entry_and_exit_debounce(self):
        starts = self.starts
        stops = self.stops
        starts, stops = cy.debounce(starts, stops, 2, 2)
        starts_validation = [2, 6]
        stops_validation = [4, 11]

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)
