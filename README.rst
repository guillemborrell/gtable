Gtable
======

Gtable is a container for tabular or tabular-like data designed with speed in
mind. It is heavily based on `pandas <http://pandas.pydata.org>`_, and it relies
on many of its capabilities. It tries to improve one some aspects of
using pandas datatyypes Series and DataFrame as containers for simple
computations:

* It tries to reduce the overhead for column access, creation, and
  concatenation.

* It supports sparse data with bitmap indexing.

* It truly handles NaNs, making a difference between a NaN and a NA in its
  internal representation.

* It provides fast transformations (filling NA values, filtering, joining...)

It also relies heavily on `numpy <http://www.numpy.org>`_. You can consider
gtable as a thin layer over numpy arrays.

Gtable is an open-source project released under a BSD 3-Clause license.