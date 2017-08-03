from gtable import Table
import numpy as np
import pandas as pd

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.small = np.arange(100)
        self.t = Table()
        self.tlarge = self.t
        self.tlarge.a = np.arange(1E6)
        self.tlarge.b = np.arange(1E6)

    def time_creation(self):
        t1 = Table()

    def time_setattr(self):
        self.tlarge.b = self.tlarge.a 

    def time_setattr_small(self):
        self.t.a = self.small

    def time_mul_setattr(self):
        self.tlarge.c = self.tlarge.a + self.tlarge.b
