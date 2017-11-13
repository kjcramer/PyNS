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
def vec_vec(x, y, gpu_arr1=None, gpu_arr2=None, gpu=False):
# -----------------------------------------------------------------------------
    """
    Args:
        x: Three-dimensional array holding vector for multiplication.
        y: Three-dimensional array holding vector for multiplication.
     ptr1: Address of the memory pre-allocated to hold x on device
     ptr2: Address of the memory pre-allocated to hold d on device
      gpu: Bool to indicate whether to run CUDA accelerated version.

    Returns:
      Result of the vector-dot product.

    Note:
      Try to find a better way to summ the elements of a matrix than
      sum(sum(sum()))
    """

    # if gpu == True then run on GPU
    if gpu:
        # populate the GPUArray with the values from x and y
        gpu_arr1.set(x)
        gpu_arr2.set(y)

        # reduce on gpu
        return gpuarray.dot( gpu_arr1, gpu_arr2).get()
    
    return sum( sum( sum( multiply(x, y) ) ) )  # end of function
