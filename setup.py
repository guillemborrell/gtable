#!/usr/bin/env python

from setuptools import setup

with open('gtable/__init__.py') as f:
    exec(f.read())

setup(name='gtable',
      version=__version__,
      description='A fast table-like container for data analytics',
      packages=['gtable'],
      author='Guillem Borrell',
      author_email='guillemborrell@gmail.com',
      install_requires=['numpy', 'pandas', 'numba'])
