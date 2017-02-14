from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

import trouver.debounce as py


class TestPyDebounceFunction(TestCase):
    # condition = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    def test_entry_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 9])
        starts, stops = py.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    2, 0)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_exit_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 11])
        starts, stops = py.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    0, 2)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_entry_and_exit_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 11])
        starts, stops = py.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    2, 2)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_small_deb(self):
        starts_validation = np.array([6])
        stops_validation = np.array([9])
        starts, stops = py.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    2.000001, 0.999999)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)


if __name__ == '__main__':
    main()
