## interaction / setup.py
'''
Setup script to compile cython files. To compile, use:
'python setup.py build_ext --inplace'

Author: Bernard Shieh (bshieh@gatech.edu)
'''
# from distutils.core import setup, Extension
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np


ext_modules = [
    Extension(name = 'interaction3.bem.core.fma_functions',
              sources = ['interaction3/bem/core/fma_functions.pyx'],
              include_dirs = [np.get_include()],
              language='c++'
    )
]


setup(
    name = 'interaction3',
    ext_modules = cythonize(ext_modules)
)

