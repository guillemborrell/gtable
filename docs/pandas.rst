Gtable and Pandas
=================

Let's start by creating a table

.. code:: python

    from gtable import Table
    import numpy as np
    import pandas as pd

.. code:: python

    t = Table()

.. code:: python

    t.a = np.random.rand(10)
    t.b = pd.date_range('2000-01-01', freq='M', periods=10)
    t.c = np.array([1,2])
    t.add_column('d', np.array([1, 2]), align='bottom')

You can create a column by assignment to an attribute. You can also use
the ``add_column`` method if the default alignment is not the one you
want. The usual representation of the table gives information about the
actual length of each column and its type.

.. code:: python

    t




.. parsed-literal::

    <Table[ a[10] <float64>, b[10] <object>, c[2] <int64>, d[2] <int64> ] object at 0x7f4f0eae68d0>



You can translate the table to a Pandas dataframe by just calling the
``to_pandas`` method, and leverage the great notebook visualization of
the Dataframe

.. code:: python

    df = t.to_pandas()
    df




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>d</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.772970</td>
          <td>2000-01-31</td>
          <td>1.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.863153</td>
          <td>2000-02-29</td>
          <td>2.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.112185</td>
          <td>2000-03-31</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.319948</td>
          <td>2000-04-30</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.657329</td>
          <td>2000-05-31</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.367910</td>
          <td>2000-06-30</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.264345</td>
          <td>2000-07-31</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.172011</td>
          <td>2000-08-31</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.007853</td>
          <td>2000-09-30</td>
          <td>NaN</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.705190</td>
          <td>2000-10-31</td>
          <td>NaN</td>
          <td>2.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Now that we have the same data stored as a Table and as a Dataframe,
let's see some of the differences between them. The first one is that
while the DataFrame has an index (an integer just keeps the order in
this case), the Table is just a table trivially indexed by the order of
the records

.. code:: python

    df.index.values




.. parsed-literal::

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



Another important difference how data is stored in each container.

.. code:: python

    df.c.values




.. parsed-literal::

    array([  1.,   2.,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan])



.. code:: python

    t.c.values




.. parsed-literal::

    array([1, 2])



While Pandas relies on NaN to store empty values, the Table uses a
bitmap index to differentiate between a missing element and a NaN

.. code:: python

    t.index




.. parsed-literal::

    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=uint8)



The mechanism for tracking NAs is the bitmap index. Of course, a bitmap
index has pros and cons. One of the interesting pros is that
computations with sparse data are significantly faster, while keeping
data indexed.

.. code:: python

    df.c.values




.. parsed-literal::

    array([  1.,   2.,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan])



.. code:: python

    t.c.values




.. parsed-literal::

    array([1, 2])



The main benefit of the Table class is that both assignment and
computation with sparse data is significantly faster. It operates with
less data, and it does not have to deal with the index

.. code:: python

    %%timeit
    2*t['c']


.. parsed-literal::

    1.63 µs ± 200 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


.. code:: python

    %%timeit
    2*df['c']


.. parsed-literal::

    73.6 µs ± 5.77 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


The amount of features of the Dataframe dwarfs the ones present in the
Table. But that does not mean that the Table is completely feature-less,
or that the features are slow. Table allows to filter the data in a
similar fashon to the Dataframe with slightly better performance.

.. code:: python

    %%timeit
    df[df.c>0]


.. parsed-literal::

    474 µs ± 89.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


.. code:: python

    df[df.c>0]




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>d</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.772970</td>
          <td>2000-01-31</td>
          <td>1.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.863153</td>
          <td>2000-02-29</td>
          <td>2.0</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    %%timeit
    t.filter(t.c > 0)


.. parsed-literal::

    131 µs ± 2.15 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


.. code:: python

    t.filter(t.c > 0).to_pandas()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.772970</td>
          <td>2000-01-31</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.863153</td>
          <td>2000-02-29</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



See that, as Table sees that there have not been results for the fourth
column, the generated dataframe omits that column.

One of the consequences of the Table's mechanism of indexing is that
data cannot be accessed through the index, and there is no such thing as
the Dataframe's iloc. If we extract the data of the column and we assign
a value to one of its items, we may get the result we want.

.. code:: python

    t['c'][1] = 3
    t.filter(t.c > 0).to_pandas()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.772970</td>
          <td>2000-01-31</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.863153</td>
          <td>2000-02-29</td>
          <td>3</td>
        </tr>
      </tbody>
    </table>
    </div>



But we cannot assign an element that does not exist

.. code:: python

    #t['c'][9]

Since the data of that column only has two elements

.. code:: python

    t['c']




.. parsed-literal::

    array([1, 3])



Up to this point we have created the Dataframe from the table, but we
can make the conversion the other way round

.. code:: python

    t1 = Table.from_pandas(df)
    t1




.. parsed-literal::

    <Table[ idx[10] <int64>, a[10] <float64>, b[10] <datetime64[ns]>, c[10] <float64>, d[10] <float64> ] object at 0x7f4ee2ae1c18>



See that some datatypes have changed, and the sparsity of the table is
lost, since Pandas cannot distinguish between NA and NaN. Note also that
another column has been added with the index information. If we already
know that all NaN are in fact NA, we can recover the sparse structure
with

.. code:: python

    t1.dropnan()

.. code:: python

    t1




.. parsed-literal::

    <Table[ idx[10] <int64>, a[10] <float64>, b[10] <datetime64[ns]>, c[2] <float64>, d[2] <float64> ] object at 0x7f4ee2ae1c18>



We can recover the types casting the columns, that are numpy arrays. To
restore the original columns we can also delete the index

.. code:: python

    t1['c'] = t1['c'].astype(np.int)
    t1['d'] = t1['d'].astype(np.int)
    t1.del_column('idx')

.. code:: python

    t1




.. parsed-literal::

    <Table[ a[10] <float64>, b[10] <datetime64[ns]>, c[2] <int64>, d[2] <int64> ] object at 0x7f4ee2ae1c18>



