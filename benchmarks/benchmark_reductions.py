import gtable as gt
import gtable.reductions


class TimeSuite:
    def setup(self):
        self.t = gt.Table()
        self.t.add_column('a', [1, 2, 2, 3, 4, 5])
        self.t.add_column('b', [1, 2, 3])
        self.t.add_column('c', [3, 4, 5], align='bottom')

    def time_reduce_sum_1(self):
        gtable.reduce_by_key(self.t, 'a', 'sum')

    def time_reduce_sum_2(self):
        gtable.reduce_by_key(self.t, 'a', 'sum')

    def time_reduce_prod_1(self):
        gtable.reduce_by_key(self.t, 'a', 'prod')

    def time_reduce_method(self):
        self.t.reduce_by_key('a').sum()

    def time_reduce_mean(self):
        self.t.reduce_by_key('a').mean()

    def time_reduce_std(self):
        self.t.reduce_by_key('a').std()

