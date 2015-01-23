from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = 'DP',
      ext_modules = cythonize(["dp_*.pyx"]),
)
