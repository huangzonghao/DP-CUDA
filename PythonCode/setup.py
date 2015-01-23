from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'DP',
    ext_modules = cythonize(["dp_*.pyx"]),
    include_dirs=[numpy.get_include()],
)
