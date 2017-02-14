from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd

from trouver.transformations import (_filter_durations, _offset_events,
                                     _debounce, RawEvents, filter_durations,
                                     offset_events, debounce, apply_condition)


class FilterTestCase(TestCase):
    @staticmethod
    def assertEvents(one, two):
        npt.assert_array_equal(one.starts, two.starts)
        npt.assert_array_equal(one.stops, two.stops)


class TestApplyCondition(FilterTestCase):
    def test_apply_condition_pd_series(self):
        condition = pd.Series([True, False, True, True])
        test_events = apply_condition(condition)
        control_events = RawEvents(np.array([0, 2]), np.array([1, 4]))
        self.assertEvents(control_events, test_events)

    def test_apply_condition_ndarray(self):
        condition = np.array([True, False, True, True])
        test_events = apply_condition(condition)
        control_events = RawEvents(np.array([0, 2]), np.array([1, 4]))
        self.assertEvents(control_events, test_events)


class TestDebounce(FilterTestCase):
    def setUp(self):
        self.condition = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))

    def test_entry_debounce(self):
        test_events = _debounce(self.input_events, period=1,
                                entry_debounce=2, exit_debounce=0)

        control_events = RawEvents(np.array([2, 7]), np.array([4, 10]))
        self.assertEvents(control_events, test_events)

    def test_exit_debounce(self):
        test_events = _debounce(self.input_events, period=1,
                                entry_debounce=0, exit_debounce=2)

        control_events = RawEvents(np.array([2, 7]), np.array([4, 12]))
        self.assertEvents(control_events, test_events)

    def test_entry_and_exit_debounce(self):
        test_events = _debounce(self.input_events, period=1,
                                entry_debounce=2, exit_debounce=3.1)

        control_events = RawEvents(np.array([2]), np.array([12]))
        self.assertEvents(control_events, test_events)

    def test_non_int_debounces(self):
        test_events = _debounce(self.input_events, period=1,
                                entry_debounce=float(0.00000001),
                                exit_debounce=float(0.99999999))

        control_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))
        self.assertEvents(control_events, test_events)

    def test_period_100ms(self):
        test_events = _debounce(self.input_events, period=0.1,
                                entry_debounce=0.15, exit_debounce=0.2)

        control_events = RawEvents(np.array([2, 7]), np.array([4, 12]))
        self.assertEvents(control_events, test_events)

    def test_period_120ms(self):
        test_events = _debounce(self.input_events, period=0.12,
                                entry_debounce=0.15, exit_debounce=0.2)

        control_events = RawEvents(np.array([2, 7]), np.array([4, 12]))
        self.assertEvents(control_events, test_events)


class TestDurationFilter(FilterTestCase):
    def setUp(self):
        # condition array: np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))

    def test_mindur(self):
        test_events = _filter_durations(self.input_events, period=1,
                                        mindur=2, maxdur=None)
        control_events = RawEvents(np.array([2, 7]), np.array([4, 10]))
        self.assertEvents(control_events, test_events)

    def test_maxdur(self):
        test_events = _filter_durations(self.input_events, period=1,
                                        mindur=0, maxdur=2)
        control_events = RawEvents(np.array([2, 11]), np.array([4, 12]))
        self.assertEvents(control_events, test_events)

    def test_mindur_maxdur(self):
        test_events = _filter_durations(self.input_events, period=1,
                                        mindur=2, maxdur=3.1)
        control_events = RawEvents(np.array([2, 7]), np.array([4, 10]))
        self.assertEvents(control_events, test_events)

    def test_nonint_durs(self):
        test_events = _filter_durations(self.input_events, period=1,
                                        mindur=float(1.00000001),
                                        maxdur=float(2.99999999))
        control_events = RawEvents(np.array([2]), np.array([4]))
        self.assertEvents(control_events, test_events)

    def test_period_100ms(self):
        test_events = _filter_durations(self.input_events, period=0.1,
                                        mindur=0.15, maxdur=0.2)
        control_events = RawEvents(np.array([2]), np.array([4]))
        self.assertEvents(control_events, test_events)

    def test_period_120ms(self):
        test_events = _filter_durations(self.input_events, period=0.12,
                                        mindur=0.15, maxdur=0.35)
        control_events = RawEvents(np.array([2]), np.array([4]))
        self.assertEvents(control_events, test_events)


class TestOffsets(FilterTestCase):
    def setUp(self):
        self.condition = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))

    def test_start_offset(self):
        control_events = RawEvents(np.array([1, 6, 10]), np.array([4, 10, 12]))
        test_events = _offset_events(self.input_events, period=1,
                                     condition_size=self.condition.size,
                                     start_offset=-1, stop_offset=0)
        self.assertEvents(control_events, test_events)

    def test_stop_offset(self):
        control_events = RawEvents(np.array([2, 7, 11]), np.array([5, 11, 12]))
        test_events = _offset_events(self.input_events, period=1,
                                     condition_size=self.condition.size,
                                     start_offset=0, stop_offset=1)
        self.assertEvents(control_events, test_events)

    def test_start_stop_offset(self):
        control_events = RawEvents(np.array([1, 6, 10]), np.array([5, 11, 12]))
        test_events = _offset_events(self.input_events, period=1,
                                     condition_size=self.condition.size,
                                     start_offset=-1, stop_offset=1)
        self.assertEvents(control_events, test_events)

    def test_period_100ms_start_stop_offset(self):
        control_events = RawEvents(np.array([1, 6, 10]), np.array([5, 11, 12]))
        test_events = _offset_events(self.input_events, period=0.1,
                                     condition_size=self.condition.size,
                                     start_offset=-0.1, stop_offset=0.1)
        self.assertEvents(control_events, test_events)

    def test_period_120ms_start_stop_offset(self):
        control_events = RawEvents(np.array([1, 6, 10]), np.array([5, 11, 12]))
        test_events = _offset_events(self.input_events, period=1,
                                     condition_size=self.condition.size,
                                     start_offset=-0.1, stop_offset=0.1)
        self.assertEvents(control_events, test_events)


class TestNoEvents(FilterTestCase):
    def test_debounce_no_events(self):
        input_events = RawEvents(np.array([]), np.array([]))
        test_events = _debounce(input_events, period=0.12,
                                entry_debounce=0.15, exit_debounce=0.2)
        control_events = RawEvents(np.array([]), np.array([]))
        self.assertEvents(control_events, test_events)

    def test_duration_filter_no_events(self):
        input_events = RawEvents(np.array([]), np.array([]))
        test_events = _filter_durations(input_events, period=0.12,
                                        mindur=0.15, maxdur=0.2)
        control_events = RawEvents(np.array([]), np.array([]))
        self.assertEvents(control_events, test_events)

    def test_offsets_no_events(self):
        input_events = RawEvents(np.array([]), np.array([]))
        test_events = _offset_events(input_events, period=0.12, condition_size=10,
                                     start_offset=-0.15, stop_offset=0.2)
        control_events = RawEvents(np.array([]), np.array([]))
        self.assertEvents(control_events, test_events)


class TestDefaultArguments(FilterTestCase):
    """Test default arguments of all functions

    The input of these functions should match the output exactly.
    Default arguments are expected to be no-ops.
    """
    def setUp(self):
        self.condition = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.input_events = RawEvents(np.array([2, 7, 11]), np.array([4, 10, 12]))

    def test_debounce_defaults(self):
        test_func = debounce()
        test_events = test_func(self.input_events, 1)
        self.assertEvents(self.input_events, test_events)

    def test_duration_filter_defaults(self):
        test_func = filter_durations()
        test_events = test_func(self.input_events, 1)
        self.assertEvents(self.input_events, test_events)

    def test_offset_defaults(self):
        test_func = offset_events()
        test_func = test_func(condition_size=self.condition.size)
        test_events = test_func(self.input_events, 1)
        self.assertEvents(self.input_events, test_events)


if __name__ == '__main__':
    main()
