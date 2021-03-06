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
        self.t.a = np.arange(100)
        self.t.b = np.arange(100)
        self.tlarge = Table()
        self.tlarge.a = np.arange(1E6)
        self.tlarge.b = np.arange(1E6)
        self.tvsmall = Table()
        self.tvsmall.a = np.arange(10)
        self.tvsmall.b = np.arange(10)

        self.df = pd.DataFrame({'a': np.arange(1E6), 'b': np.arange(1E6)})
        self.df_small = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100)})
        self.df_vsmall = pd.DataFrame({'a': np.arange(10), 'b': np.arange(10)})


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

    def time_mul_setattr_vsmall(self):
        self.tvsmall.c = self.tvsmall.a * self.tvsmall.b

    def time_setattr_own(self):
        self.t.a = self.t.a

    def time_mul_setattr(self):
        self.tlarge.c = self.tlarge.a + self.tlarge.b

    def time_pandas_mul_setattr(self):
        self.df['c'] = self.df.a + self.df.b

    def time_pandas_setattr_own(self):
        self.df['a'] = self.df.a

    def time_pandas_mul_setattr_small(self):
        self.df_small['c'] = self.df_small.a + self.df_small.b

    def time_pandas_mul_setattr_vsmall(self):
        self.df_vsmall['c'] = self.df_vsmall.a + self.df_vsmall.b

        
if __name__ == '__main__':
    import timeit
    import inspect
    import sys
    import os
    from statistics import mean, stdev
    t = TimeSuite()
    t.setup()
    print('Getting runtimes')

    for method in inspect.getmembers(t, predicate=inspect.ismethod):
        if method[0].startswith('time_'):
            stats = timeit.repeat(
                "t.{}()".format(method[0]),
                globals=globals(),
                number=100,
                repeat=10)

            mean_t = mean(stats[1:]) * 10
            stdev_t = stdev(stats[1:]) * 10
            sys.stdout.write(method[0] + ': ')
            sys.stdout.write(str(mean_t) + ' ± (std) ' +
                             str(stdev_t) + ' [ms]')
            sys.stdout.write(os.linesep)

