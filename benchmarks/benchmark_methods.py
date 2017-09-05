from gtable.lib import merge_table, sort_table
from gtable import Table
import numpy as np


class TimeSuite:
    def setup(self):
        self.a = np.arange(10, dtype=np.double)
        self.b = np.arange(5, 15, dtype=np.double)
        self.table_a = Table({'a': self.a, 'b': self.a})
        self.table_b = Table({'b': self.b})

    def time_concatenate(self):
        np.concatenate((self.a, self.b))
        
    # def time_concatenate_fast(self):
    #     concatenate_float(self.a, self.b)

    def time_merge(self):
        merge_table(self.table_a, self.table_b, 'b')
        
    def time_sort(self):
        sort_table(self.table_a, 'a')


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