.. gtable documentation master file, created by
   sphinx-quickstart on Wed Aug 16 10:50:15 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gtable's documentation!
==================================

Gtable is a container for tabular or tabular-like data designed with speed in
mind. It is heavily based on `pandas <http://pandas.pydata.org>`_, and it relies
on many of its capabilities. It tries to improve one particular aspect of
pandas datatypes, namely the overhead of :py:class:`pandas.Series` and
:py:class:`pandas.DataFrame` for simple computations. It also relies heavily on
`numpy <http://www.numpy.org>`_. You can consider gtable as a thin layer over
numpy arrays.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   motivation
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
