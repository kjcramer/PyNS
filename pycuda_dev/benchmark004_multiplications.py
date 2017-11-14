"""
Matrix element-wise multiplication, on cpu vs gpu

Inspired by the ElementWise PyCuda example in the docs
"""

import time
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand
import numpy


# ==============================================================================
# GPU CODE
# ------------------------------------------------------------------------------

# generate random matrices on gpu
a_gpu = curand((12000, 100, 100))
b_gpu = curand((12000, 100, 100))

# define kernel for multiplications
gpu_mult = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] * y[i]",
    "elementwise_multiplication")

# preallocate on gpu
c_gpu = gpuarray.empty_like(a_gpu)

# perform multiplication on gpu
starttime_gpu = time.time()
#gpu_mult(a_gpu[:,1,1], b_gpu[:,1,1], c_gpu[:,1,1])
c_gpu[:,1,1] = a_gpu[:,1,1] * b_gpu[:,1,1]
stoptime_gpu = time.time()


# ==============================================================================
# CPU CODE
# ------------------------------------------------------------------------------

# copy data from device to host
a_cpu = a_gpu.get()
b_cpu = b_gpu.get()
c_gpuFetch = c_gpu.get()

# perform multiplication on cpu
starttime_cpu = time.time()
c_cpu = a_cpu*b_cpu
stoptime_cpu = time.time()

# check cpu vs gpu discrepancy
import numpy.linalg as la
assert la.norm((c_gpuFetch - c_cpu)) < 1e-5


# ==============================================================================
# RUNTIME COMPARISON
# ------------------------------------------------------------------------------

cputime = stoptime_cpu - starttime_cpu
gputime = stoptime_gpu - starttime_gpu
print("GPU time: %2.3e s" %(gputime))
print("CPU time: %2.3e s" %(cputime))
