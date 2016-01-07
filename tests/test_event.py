import numpy as np
import numpy.testing as npt
from nimble.core.event_detection import Event
from unittest import TestCase


class TestEventDetection(TestCase):
    def test_default_parameters(self):
        """
        Test event detection with only a supplied condition
        """
        np.random.seed(10)
        validation_array = np.random.random_integers(0, 1, 100)
        condition = (validation_array > 0)
        events = Event(condition)
        
        npt.assert_array_equal(validation_array, events.as_array)
        
    def test_no_events_found(self):
        """
        Test arrays that have no active events
        """
        conditional_array = np.ones(10, dtype='i1') * 5
        validation_array = np.zeros(10, dtype='i1') 
        condition = (conditional_array > 5)
        events = Event(condition)
        
        npt.assert_array_equal(validation_array, events.as_array)

    def test_event_always_active(self):
        """
        Test arrays that has an event active throughout entire array
        """
        validation_array = np.ones(10, dtype='i1')
        condition = (validation_array > 0)
        events = Event(condition)
        
        npt.assert_array_equal(validation_array, events.as_array)

    def test_array_with_event_active_at_start(self):
        """
        Test arrays that have events active at the start
        """
        validation_array = np.zeros(10, dtype='i1')
        validation_array[:1] = 1
        condition = (validation_array > 0)
        
        events = Event(condition)
        npt.assert_array_equal(validation_array, events.as_array)

    def test_array_with_event_active_at_end(self):
        """
        Test arrays that have events active at the end
        """
        validation_array = np.zeros(10, dtype='i1')
        validation_array[-1:] = 1
        condition = (validation_array > 0)
        
        events = Event(condition)
        npt.assert_array_equal(validation_array, events.as_array)

    #def test_event_entry_debounce(self):
    #def test_event_exit_debounce(self):
    #def test_entry_and_exit_debounce(self):
    #def test_min_event_window_length(self):
    #def test_max_event_window_length(self):