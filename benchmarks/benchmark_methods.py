from gtable.fast import concatenate_float
from gtable.methods import merge
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
        
    def time_concatenate_fast(self):
        concatenate_float(self.a, self.b)

    def time_merge(self):
        merge(self.table_a, self.table_b, 'b')
        

