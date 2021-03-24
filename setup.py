#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Sun 17 Nov 2019 03:37:47 PM CST

# File Name: setup.py
# Description:

"""
import pathlib
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='scale_v2',
      version=get_version(HERE / "scale/__init__.py"),
      packages=find_packages(),
      description='Single-cell integrative Analysis via Latent feature Extraction',
      long_description=README,

      author='Lei Xiong',
      author_email='jsxlei@gmail.com',
      url='https://github.com/jsxlei/scale_v2',
      scripts=['SCALE.py'],
      install_requires=requirements,
      python_requires='>3.6.0',
      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )


