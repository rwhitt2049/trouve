from unittest import TestCase

import numpy as np
import numpy.testing as npt
import pandas as pd

import trouve
from trouve.find_events import find_events, _apply_condition


class TestApplyCondition(TestCase):
    @staticmethod
    def condition_type_fixture(condition):
        test_starts, test_stops = _apply_condition(condition)

        control_starts = np.array([0, 2])
        control_stops = np.array([1, 4])

        npt.assert_array_equal(control_starts, test_starts)
        npt.assert_array_equal(control_stops, test_stops)

    def test_apply_condition_pd_series(self):
        condition = pd.Series([True, False, True, True])
        self.condition_type_fixture(condition)

    def test_apply_condition_ndarray(self):
        condition = np.array([True, False, True, True])
        self.condition_type_fixture(condition)


class TestFindEvents(TestCase):
    def setUp(self):
        x = np.arange(10)
        self.condition = x > 5

    def test_return_type(self):
        events = find_events(self.condition, period=1)
        self.assertIsInstance(events, trouve.events.Events)


def test_find_events_empty_condition():
    events = find_events(np.array([], dtype=bool), period=1)
    assert len(events) == 0
    npt.assert_array_equal(events._starts, [])
    npt.assert_array_equal(events._stops, [])
    assert events.to_array().size == 0
