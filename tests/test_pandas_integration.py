from unittest import TestCase

import numpy as np
import pandas as pd
import numpy.testing as npt

from nimble import Events


class TestAsPandasCondition(TestCase):
    def setUp(self):
        conditional_series = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_series > 0)
        self.events = Events(condition)

    def test_as_series(self):
        validation_series = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        test_series = self.events.as_series()
        test_series.equals(validation_series)

    def test_as_array(self):
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array())


class TestAsNpArrCondition(TestCase):
    def setUp(self):
        conditional_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        condition = (conditional_array > 0)
        self.events = Events(condition)

    def test_as_series(self):
        validation_series = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        test_series = self.events.as_series()
        test_series.equals(validation_series)

    def test_as_array(self):
        validation_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        npt.assert_array_equal(validation_array, self.events.as_array())
