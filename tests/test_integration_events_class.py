from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from nimble import Events


class TestAsArrayMethod(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition).find()

    def test_len_special_method(self):
        self.assertEqual(3, len(self.events))


class TestAsArrayMethod(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition, period=1).find()

    def test_default_parameters(self):
        """Test as_array() with default settings"""
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array())

    def test_as_array_false_value(self):
        """Test as_array() with low value"""
        validation_array = np.array([-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array(
            false_values=-1))

    def test_as_array_true_value(self):
        """Test as_array() with high value"""
        validation_array = np.array([0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5])
        npt.assert_array_equal(validation_array, self. events.as_array(
            true_values=5))

    def test_as_array_false_and_true_value(self):
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
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_no_events_found(self):
        """Test arrays that have no active events"""
        conditional_array = np.ones(10, dtype='i1')
        validation_array = np.zeros(10, dtype='i1')
        condition = (conditional_array > 5)
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_event_always_active(self):
        """Test arrays that has an event active throughout entire array"""
        validation_array = np.ones(10, dtype='i1')
        condition = (validation_array > 0)
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_array_with_event_active_at_start(self):
        """Test arrays that have events active at the start"""
        validation_array = np.zeros(10, dtype='i1')
        validation_array[:1] = 1
        condition = (validation_array > 0)
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_array_with_event_active_at_end(self):
        """Test arrays that have events active at the end"""
        validation_array = np.zeros(10, dtype='i1')
        validation_array[-1:] = 1
        condition = (validation_array > 0)
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_multi_input_condition_event(self):
        """Test arrays that have multi-input conditions"""
        x = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])

        validation_array = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        condition = ((x > 0) & (y > 0))
        events = Events(condition, period=1).find()

        npt.assert_array_equal(validation_array, events.as_array())


class TestEventDebounce(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
        self.condition = (conditional_array > 0)

    def test_event_entry_debounce(self):
        validation_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        events = Events(self.condition, period=1, adeb=2).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_event_exit_debounce(self):
        validation_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        events = Events(self.condition, period=1, ddeb=2).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_entry_and_exit_debounce(self):
        validation_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        events = Events(self.condition, period=1, adeb=2, ddeb=2).find()

        npt.assert_array_equal(validation_array, events.as_array())


class TestEventLengthFilter(TestCase):
    def setUp(self):
        condition_array = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.condition = (condition_array > 0)

    def test_min_event_window_length(self):
        validation_array = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        events = Events(self.condition, period=1, mindur=2).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_max_event_window_length(self):
        validation_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
        events = Events(self.condition, period=1, maxdur=3).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_max_and_min_event_window_length(self):
        validation_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        events = Events(self.condition, period=1, mindur=2, maxdur=3).find()

        npt.assert_array_equal(validation_array, events.as_array())


class TestEventOffsets(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.condition = (condition_array > 0)

    def test_start_index_offset(self):
        validation_array = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
        validation_starts_index = np.array([0, 2, 7, 11])
        events = Events(self.condition, period=1, startoffset=-1).find()

        npt.assert_array_equal(validation_array, events.as_array())
        npt.assert_array_equal(validation_starts_index, events._starts)

    def test_stop_index_offset(self):
        validation_array = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1])
        validation_stops_index = np.array([2, 7, 11, 13])
        events = Events(self.condition, period=1, stopoffset=1).find()

        npt.assert_array_equal(validation_array, events.as_array())
        npt.assert_array_equal(validation_stops_index, events._stops)

    def test_start_and_stop_index_offset(self):
        validation_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        validation_starts_index = np.array([0, 2, 7, 11])
        validation_stops_index = np.array([2, 7, 11, 13])
        events = Events(self.condition, period=1, startoffset=-1, stopoffset=1).find()

        npt.assert_array_equal(validation_array, events.as_array())
        npt.assert_array_equal(validation_stops_index, events._stops)
        npt.assert_array_equal(validation_starts_index, events._starts)


class TestSampleRates(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
                                    0, 0, 0, 1, 0, 0])
        self.condition = (condition_array > 0)

    def test_entry_and_exit_debounce(self):
        validation_array = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0])

        events = Events(self.condition, period=1 / 3,
                        adeb=0.5, ddeb=1).find()

        npt.assert_array_equal(validation_array, events.as_array())

    def test_max_and_min_event_window_length(self):
        validation_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0])
        events = Events(self.condition, period=1 / 3,
                        mindur=0.5, maxdur=1).find()

        npt.assert_array_equal(validation_array, events.as_array())


class TestDurationArray(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
                                    0, 0, 0, 1, 0, 0])

        condition = (condition_array > 0)
        self.events = Events(condition, period=3,
                             adeb=0.5, ddeb=1).find()

    def test_durations(self):
        # validation_array = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        validation_durations = [(8/3)]
        npt.assert_array_equal(validation_durations, self.events.durations)


class TestDurationArray(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        condition = (condition_array > 0)
        self.events = Events(condition, period=1).find()

    def test__len__(self):
        self.assertEquals(4, len(self.events))

if __name__ == '__main__':
    main()
