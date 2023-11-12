from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    ...
]

setup(
    name="syphon",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "embedsignature": True,
            "language_leve": 3
        }
    ),
    include_dirs=[
        numpy.get_include()
    ]
)
