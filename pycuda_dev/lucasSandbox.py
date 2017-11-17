"""
Testing performance of set vs to_gpu, save data
"""

# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.curandom as curandom

# Utils
import time

# Standard Python modules
import numpy as np

# ==============================================================================
def copy_sandbox():
# ------------------------------------------------------------------------------
    """
    Check the behavior of gpuarray.copy()
    """
    a = gpuarray.arange(1, 5, 1, dtype=np.float32)
    b = a.copy()
    print(b)
    print(type(b))



# ==============================================================================
def elementwise_ops():
# ------------------------------------------------------------------------------
    """
    Check whether elementwise ops work in a predictable way (they do)
    """
    a = gpuarray.arange(1, 5, 1, dtype=np.float32)
    b = a*2
    c = b/a
    print(c)
    print(type(c))


# ==============================================================================
def slicing_test():
# ------------------------------------------------------------------------------
    """
    Check whether we can:
    + slice (we should, as of PyCuda 2013.1) -- manipulate gpuarrays as Rvalues
    + address and manipulate slices -- manipulate gpuarrays as Lvalues
    """
    a = curandom.rand((3, 3, 3))
    print(a)
    b = a[2,:,:]*2/2
    print(b)
    print(type(b))
    b[0,0] = gpuarray.to_gpu( np.asarray(42).astype(np.float32) )
    print(b)
    print(type(b))


# ==============================================================================
def gpu_norm():
# ------------------------------------------------------------------------------
    """
    A GPU version of Bojan's norm()
    """
    a = curandom.rand((5))
    print(a)
    norm_on_gpu = cumath.sqrt(gpuarray.dot(a, a))
    norm_on_cpu = np.sqrt(sum(a.get()**2))
    print(norm_on_cpu)
    print(norm_on_gpu)

# ==============================================================================
# ------------------------------------------------------------------------------

# ==============================================================================
def comparisons():
# ------------------------------------------------------------------------------
    """
    Comparing values that live in different places
    """
    a = gpuarray.to_gpu( np.asarray(11).astype(np.float32) )
    b_gpu = gpuarray.to_gpu( np.asarray(1000).astype(np.float32) )
    b_cpu = np.asarray(1000)
    print(a < b_gpu)
    print(a < b_cpu)
    


copy_sandbox()
# elementwise_ops()
# slicing_test()
# gpu_norm()
# comparisons()
