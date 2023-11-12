import numpy as np
cimport numpy as cnp

UINT8_DTYPE = np.uint8
UINT32_DTYPE = np.uint8
FLOAT32_DTYPE = np.float32
FLOAT64_DTYPE = np.float64
ALMOST_INF = 1e300

HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', FLOAT64_DTYPE),
    ('count', UINT32_DTYPE),
])