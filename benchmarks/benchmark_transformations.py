from gtable import Table
import numpy as np


class TimeSuite:
    def setup(self):
        t = Table()
        t._keys=['a', 'b']
        t._data=[
            np.arange(10),
            np.array([1, 2])]
        t._index = np.array(
            [[1,1,1,1,1,1,1,1,1,1],
             [0,0,1,0,0,0,0,1,0,0]], dtype=np.uint8)
    
        self.b = t.b

    def time_fillna(self):
        self.b.fillna()
        
    def time_fillna_fillvalue(self):
        self.b.fillna(fillvalue=-1)
        
    def time_fillna_reverse(self):
        self.b.fillna(reverse=True)
        
    def time_fillna_reverse_fillvalue(self):
        self.b.fillna(reverse=True, fillvalue=-1)
        
        
