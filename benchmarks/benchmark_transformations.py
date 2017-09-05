from gtable import Table
import numpy as np


class TimeSuite:
    def setup(self):
        t = Table()
        t.keys = ['a', 'b']
        t.data = [
            np.arange(10),
            np.array([1, 2])]
        t.index = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)
        self.b = t.b

    def time_fillna(self):
        self.b.fillna()
        
    def time_fillna_fillvalue(self):
        self.b.fillna(fillvalue=-1)
        
    def time_fillna_reverse(self):
        self.b.fillna(reverse=True)
        
    def time_fillna_reverse_fillvalue(self):
        self.b.fillna(reverse=True, fillvalue=-1)


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
