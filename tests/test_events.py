import numpy as np
import numpy.testing as npt
import pandas as pd
from unittest import TestCase, main
from trouver import Events


class EvTestCase(TestCase):
    @staticmethod
    def assertStartStops(events, vstarts, vstops):
        npt.assert_array_equal(events._starts, vstarts)
        npt.assert_array_equal(events._stops, vstops)


class TestDebouncing(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_adeb(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 10])
        events = Events(self.cond, period=1, adeb=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_ddeb(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=1, ddeb=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_adeb_ddeb(self):
        vstarts = np.array([2])
        vstops = np.array([12])
        events = Events(self.cond, period=1, adeb=2, ddeb=3.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_nonint_deb(self):
        vstarts = np.array([2, 7, 11])
        vstops = np.array([4, 10, 12])
        events = Events(self.cond, period=1, adeb=float(0.00000001),
                        ddeb=float(0.99999999)).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=0.1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=0.12, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 6])
        vstops = np.array([2, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, adeb=2, ddeb=2).find()
        self.assertStartStops(events, vstarts, vstops)


class TestDurationFilter(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_mindur(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 10])
        events = Events(self.cond, period=1, mindur=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_maxdur(self):
        vstarts = np.array([2, 11])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=1, maxdur=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_mindur_maxdur(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=1, mindur=2, maxdur=2.5).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_nonint_durs(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=1, mindur=float(1.00000001),
                        maxdur=float(2.99999999)).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=0.1, mindur=0.15, maxdur=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=0.12, mindur=0.15, maxdur=0.35).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, mindur=0.15, maxdur=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, mindur=0.15, maxdur=20).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 6])
        vstops = np.array([2, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, mindur=2, maxdur=2).find()
        self.assertStartStops(events, vstarts, vstops)


class TestEventOffset(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_startoffset(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([4, 10, 12])
        events = Events(self.cond, period=1, startoffset=-1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_stopoffset(self):
        vstarts = np.array([2, 7, 11])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_startoffset_stopoffset(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=0.1, startoffset=-0.1, stopoffset=0.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=0.12, startoffset=-0.1, stopoffset=0.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 5])
        vstops = np.array([3, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)


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
        npt.assert_array_equal(validation_array, self.events.as_array(
            true_values=5))

    def test_as_array_false_and_true_value(self):
        """Test as_array() with low and high values"""
        validation_array = np.array([-1, 5, 5, 5, -1, -1, -1, 5, 5, -1, 5, 5])
        npt.assert_array_equal(validation_array, self.events.as_array(
            false_values=-1,
            true_values=5))

    def test_type(self):
        typ = type(self.events.as_array(false_values=-1, true_values=5))
        self.assertEqual(typ, np.ndarray)


class TestAsSeries(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition, period=1).find()

    def test_default_parameters(self):
        """Test as_array() with default settings"""
        validation_series = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        npt.assert_array_equal(validation_series, self.events.as_series())

    def test_as_array_false_value(self):
        """Test as_array() with low value"""
        validation_series = np.array([-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1])
        npt.assert_array_equal(validation_series, self.events.as_series(
            false_values=-1))

    def test_as_array_true_value(self):
        """Test as_array() with high value"""
        validation_series = np.array([0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5])
        npt.assert_array_equal(validation_series, self.events.as_series(
            true_values=5))

    def test_as_array_false_and_true_value(self):
        """Test as_array() with low and high values"""
        validation_series = np.array([-1, 5, 5, 5, -1, -1, -1, 5, 5, -1, 5, 5])
        npt.assert_array_equal(validation_series, self.events.as_series(
            false_values=-1,
            true_values=5))

    def test_type(self):
        typ = type(self.events.as_series(false_values=-1, true_values=5))
        self.assertEqual(typ,  pd.core.series.Series)


class TestDurations(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
                                    0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

        condition = (condition_array > 0)
        self.events = Events(condition, period=1/3,
                             adeb=0.5, ddeb=1).find()

    def test_durations(self):
        # validation_array = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        validation_durations = [(8 / 3)]
        npt.assert_array_equal(validation_durations, self.events.durations)


class TestEventDetection(TestCase):
    def test_default_parameters(self):
        """Test event detection with only a supplied condition"""
        np.random.seed(10)
        validation_array = np.random.random_integers(0, 1, 100)
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


class TestSpecialMethods(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.condition = (condition_array > 0)
        self.events = Events(self.condition, period=1).find()

    def test__len__(self):
        self.assertEquals(4, len(self.events))

    def test__eq__(self):
        other = Events(self.condition, period=1).find()
        self.assertEqual(self.events, other)


class TestAttributes(TestCase):
    def setUp(self):
        condition_array = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.condition = (condition_array > 0)

    def test_period(self):
        self.assertRaises(ValueError, Events, self.condition, period=0)

    def test_startoffset(self):
        self.assertRaises(ValueError, Events, self.condition,
                          period=1, startoffset=1)

    def test_stopoffset(self):
        self.assertRaises(ValueError, Events, self.condition, period=0, stopoffset=-1)


class TestProperties(TestCase):
    def setUp(self):
        self.events = Events(np.array([False, False]), period=0.12,
                             adeb=1, ddeb=1,
                             mindur=1, maxdur=1,
                             startoffset=-1, stopoffset=1)

    def test_adeb(self):
        self.assertEqual(self.events._adeb, 9)

    def test_ddeb(self):
        self.assertEqual(self.events._adeb, 9)

    def test_mindur(self):
        self.assertEqual(self.events._mindur, 9)

    def test_maxdur(self):
        self.assertEqual(self.events._maxdur, 8)

    def test_startoffset(self):
        self.assertEqual(self.events._startoffset, -9)

    def test_stopoffset(self):
        self.assertEqual(self.events._stopoffset, 9)

if __name__ == '__main__':
    main()
