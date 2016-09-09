import numpy as np
import numpy.testing as npt
from unittest import TestCase
from nimble import Events


class EvTestCase(TestCase):
    def assertStartStops(self, events, vstarts, vstops):
        npt.assert_array_equal(events._starts, vstarts)
        npt.assert_array_equal(events._stops, vstops)


class TestDebouncing(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_adeb(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 10])
        events = Events(self.cond, period=1, adeb=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_ddeb(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=1, ddeb=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_adeb_ddeb(self):
        vstarts = np.array([2])
        vstops = np.array([12])
        events = Events(self.cond, period=1, adeb=2, ddeb=3.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_nonint_deb(self):
        vstarts = np.array([2, 7, 11])
        vstops = np.array([4, 10, 12])
        events = Events(self.cond, period=1, adeb=float(0.00000001),
                        ddeb=float(0.99999999)).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=0.1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=0.12, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

# Test debounce
        # adeb
        # ddeb
        # both
        # event active at start
        # event active at stop
        # different sample periods: 1, .1, .12

# Test durations
        # min
        # max
        # both
        # event active at start
        # event active at stop
        # different sample periods: 1, .1, .12

# Test offsets
        # start
        # stop
        # both
        # event active at start
        # event active at stop
        # different sample periods: 1, .1, .12

# Test constructor
    #property tests