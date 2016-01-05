import numpy as np
import numpy.testing as npt
from nimble.core.event_detection import Event
from unittest import TestCase


class TestEventDetection(TestCase):
    def test_default_parameters(self):
        """
        Test event detection with only a supplied condtion
        """
        np.random.seed(10)
        validation_array = np.random.random_integers(0, 1, 100)
        condition = (validation_array > 0)
        events = Event(condition)
        npt.assert_aray_equal(validation_array, events.array)
    #def test_no_events_found(self):
        """
        Test arrays that have no active events
        """

    #def test_event_always_active(self):
        """
        Test arrays that has an event active throughout entire array
        """

    #def test_array_with_event_active_at_start(self):
        """
        Test arrays that have events active at the start
        """

    #def test_array_with_event_active_at_end(self):
        """
        Test arrays that have events active at the end
        """

    #def test_event_entry_debounce(self):
    #def test_event_exit_debounce(self):
    #def test_entry_and_exit_debounce(self):
    #def test_min_event_window_length(self):
    #def test_max_event_window_length(self):