from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd

from trouve.events import Events


class EventTestCase(TestCase):
    def setUp(self):
        array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        starts = np.array([1, 7, 10])
        stops = np.array([4, 9, 12])
        period = 1
        self.events = Events(starts, stops, period, 'control', array.size)


class TestAsArray(EventTestCase):
    def test_default_parameters(self):
        """Test to_array() with default settings"""
        control = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        test = self.events.to_array()
        npt.assert_array_equal(control, test)

    def test_as_array_false_value(self):
        """Test to_array() with low value"""
        control = np.array([-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1])
        test = self.events.to_array(inactive_value=-1)
        npt.assert_array_equal(control, test)

    def test_as_array_true_value(self):
        """Test to_array() with high value"""
        control = np.array([0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5])
        test = self.events.to_array(active_value=5)
        npt.assert_array_equal(control, test)

    def test_as_array_false_and_true_value(self):
        """Test to_array() with low and high values"""
        control = np.array([-1, 5, 5, 5, -1, -1, -1, 5, 5, -1, 5, 5])
        test = self.events.to_array(inactive_value=-1, active_value=5)
        npt.assert_array_equal(control, test)

    def test_type(self):
        test = self.events.to_array()
        self.assertIsInstance(test, np.ndarray)


class TestAsMask(EventTestCase):
    def test_as_mask(self):
        test = self.events.as_mask()
        control = np.array([True, False, False, False, True, True,
                            True, False, False, True, False, False])
        npt.assert_array_equal(control, test)


class TestAsSeries(EventTestCase):
    def test_default_parameters(self):
        """Test to_array() with default settings"""
        control = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        test = self.events.to_series()
        npt.assert_array_equal(control, test)

    def test_as_array_false_value(self):
        """Test to_array() with low value"""
        control = np.array([-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1])
        test = self.events.to_series(inactive_value=-1)
        npt.assert_array_equal(control, test)

    def test_as_array_true_value(self):
        """Test to_array() with high value"""
        control = np.array([0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5])
        test = self.events.to_series(active_value=5)
        npt.assert_array_equal(control, test)

    def test_as_array_false_and_true_value(self):
        """Test to_array() with low and high values"""
        control = np.array([-1, 5, 5, 5, -1, -1, -1, 5, 5, -1, 5, 5])
        test = self.events.to_series(inactive_value=-1, active_value=5)
        npt.assert_array_equal(control, test)

    def test_type(self):
        test = self.events.to_series()
        self.assertIsInstance(test, pd.core.series.Series)


class TestDurations(TestCase):
    def setUp(self):
        array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        starts = np.array([1, 7, 11])
        stops = np.array([4, 9, 13])
        period = 1/3
        self.events = Events(starts, stops, period, 'control', array.size)

    def test_durations(self):
        control = self.events.durations
        test = np.array([3, 2, 2])/3
        npt.assert_array_equal(control, test)


class TestSpecialMethods(EventTestCase):
    def test_len(self):
        self.assertEquals(3, len(self.events))

    def test_getitem_duration(self):
        pass

    def test_getitem_istart(self):
        pass

    def test_getitem_istop(self):
        pass

    def test_getitem_slice(self):
        pass

    def test_eq(self):
        array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        starts = np.array([1, 7, 10])
        stops = np.array([4, 9, 12])
        period = 1
        other = Events(starts, stops, period, 'other', array.size)
        self.assertEqual(self.events, other)

    def test_print(self):
        test = self.events.__str__()
        control = ('control\n'
                   'Number of events: 3\n'
                   'Min, Max, Mean Duration: 2.000s, 3.000s, 2.333s')
        self.assertEqual(control, test)

    def test_repr(self):
        control = ("Events(_starts=array([ 1,  7, 10]), "
                   "_stops=array([ 4,  9, 12]), _period=1, "
                   "name='control', _condition_size=12)")
        test = repr(self.events)
        self.assertEqual(control, test)


if __name__ == '__main__':
    main()
