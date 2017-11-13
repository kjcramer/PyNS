"""
A sandbox for understanding memory allocation
"""

# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Utils
import time

# Standard Python modules
import numpy as np

# an array of float64: 8 byte * 3**3 = ?? MByte
a = np.random.randn(3, 3, 3)

# =======================================================================
# [X] pushing an array to device, reading it
# -----------------------------------------------------------------------
alloc2=cuda.to_device(a)
#print( cuda.from_device(alloc2, a.shape, a.dtype))

# =======================================================================
# [X] allocating mem on device, using it as a gpuarray object
# -----------------------------------------------------------------------
addr1=cuda.mem_alloc(a.nbytes)

# allocator=None is a default
a_gpu = gpuarray.empty(a.shape, a.dtype, gpudata=addr1, allocator=None)
print(a_gpu.get())
print( cuda.from_device(addr1, a.shape, a.dtype))
