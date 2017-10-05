"""
Vector-dot product of two vectors stored as three-dimensional arrays.
"""

# Specific Python modules
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Standard Python modules
from pyns.standard import *

# =============================================================================
def vec_vec(x, y, alloc1=None, alloc2=None, gpu=False):
# -----------------------------------------------------------------------------
    """
    Args:
      x: Three-dimensional array holding vector for multiplication.
      y: Three-dimensional array holding vector for multiplication.
    gpu: Bool to indicate whether to run CUDA accelerated version.

    Returns:
      Result of the vector-dot product.

    Note:
      Try to find a better way to summ the elements of a matrix than
      sum(sum(sum()))
    """

    # if gpu == True then run on GPU
    if gpu:
        return (gpuarray.dot( alloc1.set(x), alloc2.set(y) )).get()

    return sum( sum( sum( multiply(x, y) ) ) )  # end of function
