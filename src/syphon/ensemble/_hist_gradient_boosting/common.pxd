cimport numpy as cnp

ctypedef cnp.uint8_t UINT8_DTYPE_C
ctypedef cnp.float32_t FLOAT32_DTYPE_C
ctypedef cnp.float64_t FLOAT64_DTYPE_C

cdef packed struct hist_struct:
    # C implementation of histogram dtype for views in cython
    FLOAT64_DTYPE_C sum_gradients
    FLOAT64_DTYPE_C sum_hessians
    unsigned int count