from gtable import Table
import numpy as np
import pandas as pd
import cProfile

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.small = np.arange(100)
        self.t = Table()
        self.t.a = np.arange(100)
        self.t.b = np.arange(100)
        self.tlarge = Table()
        self.tlarge.a = np.arange(1E6)
        self.tlarge.b = np.arange(1E6)

    def time_base(self):
        pass

    def time_creation(self):
        t1 = Table()

    def time_setattr(self):
        self.tlarge.b = self.tlarge.a

    def time_setattr_small(self):
        self.t.d = self.small

    def time_mul_setattr_small(self):
        self.t.c = self.t.a * self.t.b

    def time_mul_setattr(self):
        self.tlarge.c = self.tlarge.a + self.tlarge.b

if __name__ == '__main__':
    t = TimeSuite()
    t.setup()
    pr = cProfile.Profile()
    pr.enable()
    t.time_setattr()
    t.time_creation()
    t.time_setattr_small()
    t.time_mul_setattr()
    pr.disable()
    pr.print_stats(sort='time')
    
