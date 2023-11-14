from setuptools import setup, Extension
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "syphon.ensemble._hist_gradient_boosting.histogram",
        sources=["src/syphon/ensemble/_hist_gradient_boosting/histogram.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    name="syphon",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "embedsignature": True,
            "language_level": 3
        }
    ),
    include_dirs=[
        numpy.get_include()
    ]
)
