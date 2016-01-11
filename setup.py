#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path

from setuptools import find_packages, setup

from nimble import __version__

base = path.abspath(path.dirname(__file__))


def install_requires():
    with open(path.join(base, 'requirements.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def dev_requires():
    with open(path.join(base, 'dev_requirements.txt'), encoding='utf-8') as file:
        return file.read().splitlines()


def long_description():
    with open(path.join(base, 'README.rst'), encoding='utf-8') as file:
        return file.read()


setup(
    name='nimble',
    version=__version__,
    description=long_description()[0],
    long_description=long_description(),
    author='Ry Whittington',
    author_email='rwhitt2049@gmail.om',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='time_series, timeseries, iot, sensor',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=install_requires(),
    extras_require={},
    package_data={},
    data_files=[],
    entry_points={},
    test_suite='tests',
    tests_require=dev_requires(),
)