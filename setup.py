## interaction / setup.py
'''
Setup script to compile cython files. To compile, use:
'python setup.py build_ext --inplace'

Author: Bernard Shieh (bshieh@gatech.edu)
'''
# from distutils.core import setup, Extension
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import os

# set language argument based on OS
if os.name == 'nt':
    _LANGUAGE = 'c++' # for MSVC++
else:
    _LANGUAGE = 'c' # for gcc

ext_modules = [
    Extension(name='interaction3.bem.core.fma_functions',
              sources=['interaction3/bem/core/fma_functions.pyx'],
              include_dirs=[np.get_include()],
              language=_LANGUAGE
    ),
    Extension(name='interaction3.beamform.engines_cy',
              sources=['interaction3/beamform/engines_cy.pyx'],
              include_dirs=[np.get_include()],
              language=_LANGUAGE
    )
]


setup(
    name='interaction3',
    version='0.5',
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    package_data={
        'interaction3.abstract': ['*.json'],
        'interaction3.mfield.core': ['*.m', '*.mat', '*.pdf', '*.mexw64', '*.mexa64']
    },
    entry_points={
      'console_scripts': [
          'interaction3 = interaction3.cli:main'
      ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'cython',
        'pandas',
        'sympy',
        'attrs',
        'tqdm',
        'namedlist'
    ]
)

