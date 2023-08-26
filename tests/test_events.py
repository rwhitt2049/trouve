import numpy as np
import numpy.testing as npt
import pytest

import trouve
from trouve.events import Events


@pytest.fixture
def reference_array():
    return np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])


@pytest.fixture
def reference_events(reference_array):
    return trouve.find_events(reference_array, period=1, name="reference")


def test_reference_events(reference_events, reference_array):
    expected = Events(
        starts=np.array([1, 7, 10]),
        stops=np.array([4, 9, 12]),
        period=1,
        name="test",
        condition_size=len(reference_array)
    )

    assert reference_events == expected


def test_events_to_array(reference_events, reference_array):
    actual = reference_events.to_array()
    expected = reference_array
    npt.assert_equal(actual, expected, err_msg="Converting events to array don't match expected")

    expected = reference_array.copy()
    expected[expected == 0] = 5
    expected[expected == 1] = 3

    npt.assert_equal(reference_events.to_array(5, 3), expected, err_msg="Converting events to array with options don't match expected")
    assert isinstance(reference_events.to_array(), np.ndarray)


def test_events_durations(reference_events):
    expected = np.array([3, 2, 2])
    actual = reference_events.durations
    npt.assert_equal(actual, expected)


if __name__ == '__main__':
    pytest.main()
