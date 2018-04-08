#!/usr/bin/env python

from setuptools import setup

with open('gtable/version.py') as f:
    exec(f.read())

with open('README.rst') as f:
    long_description = f.read()

setup(name='gtable',
      version=__version__,
      description='A fast table-like container for data analytics',
      packages=['gtable'],
      long_description=long_description,
      license='Revised BSD-3 Clause',
      author='Guillem Borrell',
      author_email='guillemborrell@gmail.com',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3 :: Only'
      ],
      install_requires=['numpy', 'pandas', 'numba'])
