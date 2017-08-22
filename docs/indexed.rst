Indexed or unindexed columns
============================

One important feature of the Table container is the ability to compute
arithmetic operations with and without taking the index into account.
Using the indexed or the unindexed operation is left as a choice for the
user. This notebook tries to explain the differences, and the
consequences of using each option. We'll start by creating a sparse
Table.

.. code:: python

    from gtable import Table
    t = Table()
    t.add_column('a', [1,2,3,4,5,6])
    t.add_column('b', [1,2,3], align="bottom")

If we access the column by attribute, we'll get a type called
``Column``, that includes information about the index

.. code:: python

    t.a




.. parsed-literal::

    <Column[ int64 ] object at 0x7f5348736c50>



Accessing by key is a shortcut to the data stored within the ``Table``,
and has no information about how the table is indexed.

.. code:: python

    t['a']




.. parsed-literal::

    array([1, 2, 3, 4, 5, 6])



The column ``a`` is not a particularly good example, but ``b`` is. The
data stored in the latter column has only three elements. Where those
elements are actually placed within the table is stored in the index.

.. code:: python

    t['b']




.. parsed-literal::

    array([1, 2, 3])



The easiest and safest way to operate with columns is to take the index
into account

.. code:: python

    c = t.a + t.b

.. code:: python

    c.values




.. parsed-literal::

    array([ 2.,  3.,  4.])



See that, since the ``b`` column only had three elements, the result of
the addition with the ``a`` column has only three elements. There are no
NaNs or NAs. However, using the index to perform arithmetic operations
has some cost, particularly in the case of large dense columns. Assume
that we want to scale the ``a`` column by the last element of ``b``. We
can do that either accessing the full column or by accessing the raw
data

.. code:: python

    c = t.a * t.b[-1]

.. code:: python

    c.values




.. parsed-literal::

    array([ 3,  6,  9, 12, 15, 18])



Using columns is more convenient, since in many cases arithmetic
operations do what they are supposed to do, but they have an important
caveat: performance:

.. code:: python

    %%timeit
    t.a * t.b[-1]


.. parsed-literal::

    26.2 µs ± 224 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


.. code:: python

    %%timeit
    t['a'] * t['b'][-1]


.. parsed-literal::

    6.19 µs ± 88.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


But since the data of each column has a different length, using the raw
data or the colum will have different outcomes

.. code:: python

    t.a + t.b




.. parsed-literal::

    <Column[ float64 ] object at 0x7f531fc411d0>



.. code:: python

    t['a'] + t['b']


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-12-2fb36be086ed> in <module>()
    ----> 1 t['a'] + t['b']
    

    ValueError: operands could not be broadcast together with shapes (6,) (3,) 


A caveat of columns is that they are designed to perform fast operations
using the column as a whole, and in consequence, accessing individual
item of a column is O(N).

Another important difference is that we can create new columns by
attribute, but not by index

.. code:: python

    t.c = t.a + t.b

.. code:: python

    t




.. parsed-literal::

    <Table[ a[6] <int64>, b[3] <int64>, c[3] <float64> ] object at 0x7f5348736fd0>



.. code:: python

    t['d'] = t['a']


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-16-46490dc9bd03> in <module>()
    ----> 1 t['d'] = t['a']
    

    ~/projects/gtable/gtable/table.py in __setitem__(self, key, value)
        158 
        159     def __setitem__(self, key, value):
    --> 160         self.data[self.keys.index(key)] = value
        161 
        162     def __delitem__(self, key):


    ValueError: 'd' is not in list

