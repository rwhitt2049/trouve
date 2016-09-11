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

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, adeb=0.15, ddeb=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 6])
        vstops = np.array([2, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, adeb=2, ddeb=2).find()
        self.assertStartStops(events, vstarts, vstops)


class TestDurationFilter(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_mindur(self):
        vstarts = np.array([2, 7])
        vstops = np.array([4, 10])
        events = Events(self.cond, period=1, mindur=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_maxdur(self):
        vstarts = np.array([2, 11])
        vstops = np.array([4, 12])
        events = Events(self.cond, period=1, maxdur=2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_mindur_maxdur(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=1, mindur=2, maxdur=2.5).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_nonint_durs(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=1, mindur=float(1.00000001),
                        maxdur=float(2.99999999)).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=0.1, mindur=0.15, maxdur=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([2])
        vstops = np.array([4])
        events = Events(self.cond, period=0.12, mindur=0.15, maxdur=0.35).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, mindur=0.15, maxdur=0.2).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, mindur=0.15, maxdur=20).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 6])
        vstops = np.array([2, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, mindur=2, maxdur=2).find()
        self.assertStartStops(events, vstarts, vstops)


class TestEventOffset(EvTestCase):
    def setUp(self):
        condarr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        self.cond = condarr > 0

    def test_startoffset(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([4, 10, 12])
        events = Events(self.cond, period=1, startoffset=-1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_stopoffset(self):
        vstarts = np.array([2, 7, 11])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_startoffset_stopoffset(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_100ms(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=0.1, startoffset=-0.1, stopoffset=0.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_period_120ms(self):
        vstarts = np.array([1, 6, 10])
        vstops = np.array([5, 11, 12])
        events = Events(self.cond, period=0.12, startoffset=-0.1, stopoffset=0.1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_no_events_found(self):
        vstarts = np.array([])
        vstops = np.array([])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x > 0, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_event_always_active(self):
        vstarts = np.array([0])
        vstops = np.array([8])
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        events = Events(x == 0, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)

    def test_end_conditions(self):
        vstarts = np.array([0, 5])
        vstops = np.array([3, 8])
        x = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        events = Events(x == 1, period=1, startoffset=-1, stopoffset=1).find()
        self.assertStartStops(events, vstarts, vstops)


# Test constructor
    #property tests