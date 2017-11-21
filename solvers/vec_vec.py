"""
Vector-dot product of two vectors stored as three-dimensional arrays.
"""

# Standard Python modules
from pyns.standard import *

# =============================================================================
def vec_vec(x, y, gpu=False):
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
	
	# Specific Python modules
	import pycuda.driver as cuda
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
        
	return gpuarray.dot( gpuarray.to_gpu(x), gpuarray.to_gpu(y))
    
    return sum( sum( sum( multiply(x, y) ) ) )  # end of function
