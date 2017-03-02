from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd

import trouver
from trouver.find_events import find_events, _apply_condition


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
        events = find_events(self.condition, 1)
        self.assertIsInstance(events, trouver.events.Events)


if __name__ == '__main__':
    main()
