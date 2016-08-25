#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path
from setuptools import find_packages, setup, Extension
import numpy
from nimble import __version__

USE_CYTHON = False
base = path.abspath(path.dirname(__file__))

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension('nimble.cyfunc.debounce', ['nimble/cyfunc/debounce'+ext]),
              Extension('nimble.cyfunc.as_array', ['nimble/cyfunc/as_array'+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


def install_requires():
    with open(path.join(base, 'requirements.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def dev_requires():
    with open(path.join(base, 'dev_requirements.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def long_description():
    with open(path.join(base, 'README.md'), encoding='utf-8') as file:
        return file.read()

extras = {
    'Cython': ['Cython']
}

kwargs = dict(
    name='nimble',
    version=__version__,
    description=long_description()[0],
    long_description=long_description(),
    author='Ry Whittington',
    author_email='rwhitt2049@gmail.com',
    license='MIT',
    url='https://github.com/rwhitt2049/nimble',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
    ],
    keywords='time_series, timeseries, iot, sensor',
    packages=find_packages(exclude=['contrib', 'documentation', 'tests*']),
    install_requires=install_requires(),
    package_data={},
    data_files=[],
    entry_points={},
    test_suite='tests',
    tests_require=dev_requires()
)

try:
    setup(ext_modules=extensions,
          include_dirs=[numpy.get_include()],
          extras_require=extras,
          **kwargs)
except SystemExit:
    setup(**kwargs)