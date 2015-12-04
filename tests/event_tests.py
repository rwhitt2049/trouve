import numpy as np
from nimble.core.event_detection import Event
from unittest import TestCase

class TestEventDetection(TestCase):
    def test_default_parameters(self):
        """
        Test event detection with only a supplied condtion and default
        parameters.
        """
        
