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
def scalar_array_multiplication():
# ------------------------------------------------------------------------------
    """
    How to multiply a scalar with an array
    """
    a_array = curandom.rand((2, 2, 2))
    b = gpuarray.to_gpu(np.asarray(10.0).astype(np.float32))
    c_array = curandom.rand((2, 2, 2))
    d_array = curandom.rand((2, 2, 2))
    e_array = b.get() * b.get() * d_array + b.get() * (c_array - b.get() * a_array)
    print(a_array)
    print(c_array)
    print(e_array)
    print(np.shape(e_array.get()))

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

    print("--- the original matrix ---")
    print(b)
    print(type(b))

    print("--- replacing one value ---")
    b[0,0] = gpuarray.to_gpu( np.asarray(42).astype(np.float32) )
    print(b)

    print("--- replacing more than one value ---")
    b[:,0] = gpuarray.to_gpu( (np.ones((3)) * 43).astype(np.float32) )
    print(b)

    print("--- replacing more than one value, more elegantly ---")
    b[:-1,:] = gpuarray.to_gpu( (np.ones(( np.shape(b[:-1,:]) )) * 44).astype(np.float32) )
    print(b)

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
    
scalar_array_multiplication()
# slicing_test()
# gpu_norm()
# comparisons()