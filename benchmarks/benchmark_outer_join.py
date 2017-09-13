import gtable as gt
import numpy as np
import pandas as pd


class TimeSuite:
    def setup(self):
        self.df1_s = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                                   'B': ['B0', 'B1', 'B2', 'B3'],
                                   'C': ['C0', 'C1', 'C2', 'C3'],
                                   'D': ['D0', 'D1', 'D2', 'D3']},
                                  index=[0, 1, 2, 3])

        self.df2_s = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                                   'B': ['B4', 'B5', 'B6', 'B7'],
                                   'C': ['C4', 'C5', 'C6', 'C7'],
                                   'D': ['D4', 'D5', 'D6', 'D7']},
                                  index=[4, 5, 6, 7])

        self.t1_s = gt.Table({'A': ['A0', 'A1', 'A2', 'A3'],
                              'B': ['B0', 'B1', 'B2', 'B3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3'],
                              'idx': [1, 2, 3, 4]})

        self.t2_s = gt.Table({'A': ['A4', 'A5', 'A6', 'A7'],
                              'B': ['B4', 'B5', 'B6', 'B7'],
                              'C': ['C4', 'C5', 'C6', 'C7'],
                              'D': ['D4', 'D5', 'D6', 'D7'],
                              'idx': [5, 6, 7, 8]})

        self.df1_m = pd.DataFrame({'A': np.random.rand(100),
                                   'B': np.random.rand(100),
                                   'C': np.random.rand(100),
                                   'D': np.random.rand(100),
                                   'E': np.random.rand(100),
                                   'F': np.random.rand(100),
                                   'G': np.random.rand(100)},
                                  index=np.arange(100))

        self.df2_m = pd.DataFrame({'A': np.random.rand(100),
                                   'G': np.random.rand(100)},
                                  index=np.arange(100, 200))

        self.t1_m = gt.Table({'A': np.random.rand(100),
                              'B': np.random.rand(100),
                              'C': np.random.rand(100),
                              'D': np.random.rand(100),
                              'E': np.random.rand(100),
                              'F': np.random.rand(100),
                              'G': np.random.rand(100),
                              'idx': np.arange(100)})

        self.t2_m = gt.Table({'A': np.random.rand(100),
                              'G': np.random.rand(100),
                              'idx': np.arange(100, 200)})

        self.df1_l = pd.DataFrame({'A': np.random.rand(100000),
                                   'B': np.random.rand(100000),
                                   'C': np.random.rand(100000)},
                                  index=np.arange(100000))

        self.df2_l = pd.DataFrame({'A': np.random.rand(100000),
                                   'B': np.random.rand(100000),
                                   'C': np.random.rand(100000)},
                                  index=np.arange(100000, 200000))

        self.t1_l = gt.Table({'A': np.random.rand(100000),
                              'B': np.random.rand(100000),
                              'C': np.random.rand(100000),
                              'idx': np.arange(100000)})

        self.t2_l = gt.Table({'A': np.random.rand(100000),
                              'B': np.random.rand(100000),
                              'C': np.random.rand(100000),
                              'idx': np.arange(100000, 200000)})

    def time_pandas_outer_join_small(self):
        pd.concat([self.df1_s, self.df2_s])

    def time_gtable_outer_join_small(self):
        gt.full_outer_join(self.t1_s, self.t2_s, 'idx')

    def time_pandas_outer_join_med(self):
        pd.concat([self.df1_m, self.df2_m])

    def time_gtable_outer_join_med(self):
        gt.full_outer_join(self.t1_m, self.t2_m, 'idx')

    def time_pandas_outer_join_large(self):
        pd.concat([self.df1_l, self.df2_l])

    def time_gtable_outer_join_large(self):
        gt.full_outer_join(self.t1_l, self.t2_l, 'idx')
