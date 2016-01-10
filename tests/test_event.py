import numpy as np
import numpy.testing as npt
from nimble.core.event_detection import Events
from unittest import TestCase


class TestAsArrayMethod(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        self.events = Events(conditional_array)

    def test_default_parameters(self):
        """Test as_array() with default settings"""
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array())

    def test_as_array_low_value(self):
        """Test as_array() with low value"""
        validation_array = np.array([-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array(
            false_values=-1))

    def test_as_array_high_value(self):
        """Test as_array() with high value"""
        validation_array = np.array([0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5])
        npt.assert_array_equal(validation_array, self. events.as_array(
            true_values=5))

    def test_as_array_low_and_high_value(self):
        """Test as_array() with low and high values"""
        validation_array = np.array([-1, 5, 5, 5, -1, -1, -1, 5, 5, -1, 5, 5])
        npt.assert_array_equal(validation_array, self.events.as_array(
            false_values=-1,
            true_values=5))


class TestEventDetection(TestCase):
    def test_default_parameters(self):
        """Test event detection with only a supplied condition"""
        np.random.seed(10)
        validation_array = np.random.random_integers(0, 1, 100)
        condition = (validation_array > 0)
        events = Events(condition)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_no_events_found(self):
        """Test arrays that have no active events"""
        conditional_array = np.ones(10, dtype='i1') * 5
        validation_array = np.zeros(10, dtype='i1')
        condition = (conditional_array > 5)
        events = Events(condition)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_event_always_active(self):
        """Test arrays that has an event active throughout entire array"""
        validation_array = np.ones(10, dtype='i1')
        condition = (validation_array > 0)
        events = Events(condition)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_array_with_event_active_at_start(self):
        """Test arrays that have events active at the start"""
        validation_array = np.zeros(10, dtype='i1')
        validation_array[:1] = 1
        condition = (validation_array > 0)

        events = Events(condition)
        npt.assert_array_equal(validation_array, events.as_array())

    def test_array_with_event_active_at_end(self):
        """Test arrays that have events active at the end"""
        validation_array = np.zeros(10, dtype='i1')
        validation_array[-1:] = 1
        condition = (validation_array > 0)

        events = Events(condition)
        npt.assert_array_equal(validation_array, events.as_array())

    def test_event_entry_debounce(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        condition = (conditional_array > 0)
        events = Events(condition, entry_debounce=3)
        npt.assert_array_equal(validation_array, events.as_array())

    def test_event_exit_debounce(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

        condition = (conditional_array > 0)
        events = Events(condition, exit_debounce=2)
        npt.assert_array_equal(validation_array, events.as_array())

    def test_entry_and_exit_debounce(self):
        conditional_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
        validation_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])

        condition = (conditional_array > 0)
        events = Events(condition, entry_debounce=2, exit_debounce=2)
        npt.assert_array_equal(validation_array, events.as_array())

    def test_min_event_window_length(self):
        condition_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        validation_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

        condition = (condition_array > 0)
        events = Events(condition, min_event_length=3)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_max_event_window_length(self):
        condition_array = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        validation_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

        condition = (condition_array > 0)
        events = Events(condition, max_event_length=3)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_min_event_window_length(self):
        condition_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        validation_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

        condition = (condition_array > 0)
        events = Events(condition, min_event_length=3)

        npt.assert_array_equal(validation_array, events.as_array())

    def test_max_and_min_event_window_length(self):
        condition_array = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1])
        validation_array = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        condition = (condition_array > 0)
        events = Events(condition, min_event_length=2, max_event_length=3)

        npt.assert_array_equal(validation_array, events.as_array())
