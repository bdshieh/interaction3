## interaction / setup.py
'''
Setup script to compile cython files. To compile, use:
'python setup.py build_ext --inplace'

Author: Bernard Shieh (bshieh@gatech.edu)
'''
from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np




ext_modules = [
    Extension(name = 'bem.core.fma_functions',
              sources = ['bem/core/fma_functions.pyx'],
              include_dirs = [np.get_include()]
    )
]

setup(
    name = 'interaction3',
    ext_modules = cythonize(ext_modules)
)

