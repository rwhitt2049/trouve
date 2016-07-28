from unittest import TestCase, main
import numpy as np
import numpy.testing as npt
import nimble.debounce as py
import nimble.cyfunc.debounce as cy
from nimble import Events


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


class TestCyDebounceFunction(TestCase):
    # condition = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    def test_entry_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 9])
        starts, stops = cy.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    2,  0)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_exit_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 11])
        starts, stops = cy.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
                                    0, 2)

        npt.assert_array_equal(starts_validation, starts)
        npt.assert_array_equal(stops_validation, stops)

    def test_entry_and_exit_debounce(self):
        starts_validation = np.array([2, 6])
        stops_validation = np.array([4, 11])
        starts, stops = cy.debounce(np.array([2, 6, 10]), np.array([4, 9, 11]),
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


class TestCyvPy(TestCase):
    def setUp(self):
        np.random.seed(10)
        x = np.random.random_integers(0, 1, 300000)
        events = Events(x > 0)
        self.starts = events.starts
        self.stops = events.stops

    def test_large_array_compare(self):
        py_starts, py_stops = py.debounce(self.starts, self.stops, 2, 2)
        cy_starts, cy_stops = cy.debounce(self.starts, self.stops, 2, 2)

        npt.assert_array_equal(py_starts, cy_starts)
        npt.assert_array_equal(py_stops, cy_stops)

if __name__ == '__main__':
    main()
