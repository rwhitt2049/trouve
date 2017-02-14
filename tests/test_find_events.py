from unittest import TestCase, main

import numpy as np
from toolz import curry

import trouver
from trouver.events import find_events, curry_func


def func_no_sample_period():
    return


@curry
def func_with_sample_period(sample_period, an_arg):
    return


@curry
def func_with_condition_size(condition_size, an_arg):
    return


class TestCurrySamplePeriod(TestCase):
    def test_without_without_sample_period(self):
        test = curry_func(func_no_sample_period, 1, 5)
        self.assertTrue(callable(test))

    def test_function_with_sample_period(self):
        test = curry_func(func_with_sample_period, 1, 5)
        self.assertTrue(callable(test))

    def test_function_with_condition_size(self):
        test = curry_func(func_with_condition_size, 1, 5)
        self.assertTrue(callable(test))


class TestFindEvents(TestCase):
    def setUp(self):
        x = np.arange(10)
        self.condition = x > 5

    def test_return_type(self):
        events = find_events(self.condition, 1)
        self.assertIsInstance(events, trouver.events.Events)


if __name__ == '__main__':
    main()
