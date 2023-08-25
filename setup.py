#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path

from setuptools import find_packages, setup

base = path.abspath(path.dirname(__file__))

def get_version():
    with open(path.join(base, "trouve/__init__.py")) as fp:

        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


def install_requires():
    with open(path.join(base, 'requirements.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def dev_requires():
    with open(path.join(base, 'requirements-dev.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def long_description():
    with open(path.join(base, 'README.rst'), encoding='utf-8') as file:
        return file.read()


kwargs = dict(
    name='trouve',
    version=get_version(),
    description='Event detection and transformation for time-series data',
    long_description=long_description(),
    author='Ry Whittington',
    author_email='rwhitt2049@gmail.com',
    license='MIT',
    url='https://github.com/rwhitt2049/trouve',
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
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
    keywords='time_series, timeseries, iot, sensor',
    packages=find_packages(exclude=['contrib', 'documentation', 'tests*']),
    install_requires=install_requires(),
    package_data={},
    data_files=[],
    extras_require={"develop": dev_requires()},
    entry_points={},
    test_suite='tests',
    tests_require=dev_requires()
)

setup(**kwargs)
