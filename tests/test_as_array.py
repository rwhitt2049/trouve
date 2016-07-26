from unittest import TestCase, main
import numpy as np
import numpy.testing as npt
import nimble.as_array as py
import nimble.cyfunc.as_array as cy
from nimble import Events


class TestPyAsArrayFunction(TestCase):
    def setUp(self):
        # output = as_array(self.starts, self.stops, output, true_values)
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        events = Events(condition)
        self.starts = events.starts
        self.stops = events.stops
        self.mask = np.zeros(condition.size)

    def test_python_as_array(self):
        """Test as_array() with default settings"""
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        test_array = py.as_array(self.starts, self.stops, self.mask, 1)
        npt.assert_array_equal(validation_array, test_array)


class TestCyAsArrayFunction(TestCase):
    def setUp(self):
        # output = as_array(self.starts, self.stops, output, true_values)
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        events = Events(condition)
        self.starts = events.starts
        self.stops = events.stops
        self.mask = np.zeros(condition.size)
        self.validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])

    def test_c_as_array(self):
        """Test as_array() with default settings"""
        test_array = cy.as_array(self.starts, self.stops, self.mask, 1)
        npt.assert_array_equal(self.validation_array, test_array)


class TestCyvPy(TestCase):
    def setUp(self):
        np.random.seed(10)
        x = np.random.random_integers(0, 1, 300000)
        events = Events(x > 0)
        self.mask = np.zeros(events.n_events)
        self.starts = events.starts
        self.stops = events.stops

    def test_large_array_compare(self):
        py_arr = py.as_array(self.starts, self.stops, self.mask)
        cy_arr = cy.as_array(self.starts, self.stops, self.mask)

        npt.assert_array_equal(py_arr, cy_arr)

if __name__ == '__main__':
    main()
