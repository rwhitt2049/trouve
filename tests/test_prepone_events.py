from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from trouve.events import Events
from trouve.transformations import prepone_events


import pytest


@pytest.fixture
def raw_event(period=1):
    def _raw_event():
        condition = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        starts = np.array([2, 7, 11])
        stops = np.array([4, 10, 12])

        input_events = Events(starts, stops, period, 'input',
                              condition.size)
        return input_events
    return _raw_event


def assert_events_equal(expected_starts, actual_starts,
                        expected_stops, actual_stops):
    npt.assert_equal(actual_starts, expected_starts)
    npt.assert_equal(actual_stops, expected_stops)


def test_start_offset(raw_event):
    expected_starts = np.array([1, 6, 10])
    expected_stops = np.array([4, 10, 12])
    
    transformation = prepone_events(1, 0)
    actual_events = transformation(raw_event())
    assert_events_equal(expected_starts, actual_events._starts,
                        expected_stops, actual_events._stops)

def test_stop_offset(raw_event):
    expected_starts = np.array([2, 7])
    expected_stops = np.array([3, 9])

    transformation = prepone_events(0, 1)
    actual_events = transformation(raw_event())
    assert_events_equal(expected_starts, actual_events._starts,
                        expected_stops, actual_events._stops)


def test_start_stop_offset(raw_event):
    expected_starts = np.array([1, 6, 10])
    expected_stops = np.array([3, 9, 11])

    transformation = prepone_events(1, 1)
    actual_events = transformation(raw_event())
    assert_events_equal(expected_starts, actual_events._starts,
                        expected_stops, actual_events._stops)

#
# def test_period_100ms_start_stop_offset(self):
#     period = 0.1
#     expected_starts = np.array([1, 6, 10])
#     expected_stops = np.array([5, 11, 12])
#     self.fixture(expected_starts, expected_stops, period, start=0.1, stop=0.1)
#
# def test_period_120ms_start_stop_offset(self):
#     period = 0.12
#     expected_starts = np.array([1, 6, 10])
#     expected_stops = np.array([5, 11, 12])
#     self.fixture(expected_starts, expected_stops, period, start=0.1, stop=0.1)
#
#
# class TransformationTestFixture(TestCase):
#     def setUp(self):
#         raise NotImplementedError
#
#     def fixture(self, expected_starts, expected_stops, period, **kwargs):
#         input_events = Events(self.starts, self.stops, period, 'input', self.condition.size)
#
#         test_events = self.transformation(input_events, **kwargs)
#         test_starts, test_stops = test_events._starts, test_events._stops
#
#         npt.assert_array_equal(expected_starts, test_starts)
#         npt.assert_array_equal(expected_stops, test_stops)
#
#
# class TestPrePone(TransformationTestFixture):
#     def setUp(self):
#         self.condition = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
#         self.starts = np.array([2, 7, 11])
#         self.stops = np.array([4, 10, 12])
#         self.transformation = _pre_pone_events
#
#     def test_start_offset(self):
#         period = 1
#         expected_starts = np.array([1, 6, 10])
#         expected_stops = np.array([4, 10, 12])
#         self.fixture(expected_starts, expected_stops, period, start=1, stop=0)
#
#     def test_stop_offset(self):
#         period = 1
#         expected_starts = np.array([2, 7])
#         expected_stops = np.array([4, 10])
#         self.fixture(expected_starts, expected_stops, period, start=0, stop=1)
#
#     def test_start_stop_offset(self):
#         period = 1
#         expected_starts = np.array([1, 6])
#         expected_stops = np.array([3, 9])
#         self.fixture(expected_starts, expected_stops, period, start=1, stop=1)
#
#     def test_period_100ms_start_stop_offset(self):
#         period = 0.1
#         expected_starts = np.array([1, 6, 10])
#         expected_stops = np.array([5, 11, 12])
#         self.fixture(expected_starts, expected_stops, period, start=0.1, stop=0.1)
#
#     def test_period_120ms_start_stop_offset(self):
#         period = 0.12
#         expected_starts = np.array([1, 6, 10])
#         expected_stops = np.array([5, 11, 12])
#         self.fixture(expected_starts, expected_stops, period, start=0.1, stop=0.1)


if __name__ == '__main__':
    pytest.main()