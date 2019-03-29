from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [Extension(
    "ct",
    ["_tomograph.pyx"],
    extra_compile_args=['-O3',"-fopenmp"],
    extra_link_args=['-fopenmp'],
)]

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(ext_modules)
)
