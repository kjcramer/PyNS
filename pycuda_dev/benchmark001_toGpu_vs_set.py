"""
Testing performance of set vs to_gpu
"""

# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Utils
import time

# Standard Python modules
import numpy as np

# array size
sz = 16

# number of iterations
iters = 8000

# generate random 3x3x3 arrays on host
a = np.random.randn(sz, sz, sz)
b = np.random.randn(sz, sz, sz)

# preallocate space for the result on host
c = np.empty(1, a.dtype)

# gpuarray.to_gpu -- push arrays to device, perform reduction
starttime = time.time()
for ii in range(0, iters):
    (gpuarray.dot( gpuarray.to_gpu(a), gpuarray.to_gpu(b))).get(c)

endtime = time.time()
print("gpuarray.to_gpu -- %d iterations in %2.3e s" %(ii+1, endtime - starttime))

# gpuarray.set -- preallocate on device, push arrays to device, perform reduction
starttime = time.time()

a_gpu = gpuarray.empty( (sz, sz, sz), a.dtype)
b_gpu = gpuarray.empty_like(a_gpu)

for ii in range(0, iters):
    a_gpu.set(a)
    b_gpu.set(b)
    (gpuarray.dot( a_gpu, b_gpu )).get(c)

endtime = time.time()
print("gpuarray.set -- %d iterations in %2.3e s" %(ii+1, endtime - starttime))




