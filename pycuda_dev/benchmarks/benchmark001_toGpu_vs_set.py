"""
Testing performance of set vs to_gpu, save data
"""

# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Utils
import time

# Standard Python modules
import numpy as np

# ==============================================================================
def benchmark(sz, iters):
# ------------------------------------------------------------------------------
    """
    Args:
        sz: sz**3 gives the number of elements of the 3x3x3 array
     iters: number of iterations
    """

    # generate random 3x3x3 arrays on host
    a = np.random.randn(sz, sz, sz)
    b = np.random.randn(sz, sz, sz)
    
    # preallocate space for the result on host
    c = np.empty(1, a.dtype)
    
    # reduce on host (burrent solution)
    starttime = time.time()
    for ii in range(0, iters):
        c_cpu = sum(sum(sum(np.multiply(a, b))))
    endtime = time.time()

    cputime = endtime - starttime
    print("cpu -- size: %d -- iterations: %d -- in %2.3e s" \
          %(sz, iters, cputime))
    
    # gpuarray.to_gpu -- push arrays to device, perform reduction
    starttime = time.time()
    
    for ii in range(0, iters):
        (gpuarray.dot( gpuarray.to_gpu(a), gpuarray.to_gpu(b))).get(c)
    
    endtime = time.time()

    gputime_toGpu = endtime - starttime
    print("gpuarray.to_gpu -- size: %d -- iterations: %d -- in %2.3e s" \
          %(sz, iters, gputime_toGpu))
    
    # gpuarray.set -- preallocate on device, push arrays to device, perform reduction
    starttime = time.time()
    
    a_gpu = gpuarray.empty( (sz, sz, sz), a.dtype)
    b_gpu = gpuarray.empty_like(a_gpu)
    
    for ii in range(0, iters):
        a_gpu.set(a)
        b_gpu.set(b)
        (gpuarray.dot( a_gpu, b_gpu )).get(c)
    
    endtime = time.time()

    gputime_set = endtime - starttime
    print("gpuarray.set -- size: %d -- iterations: %d -- in %2.3e s" \
          %(sz, iters, gputime_set))
    
    return [cputime, gputime_toGpu, gputime_set]

# ==============================================================================
# ------------------------------------------------------------------------------

# array size, from 64 to 256 in 16-increments
SZ = [16*x for x in range(4,17)]

# number of iterations, in powers of 2 from 32 to 128
ITERS = [2**x for x in range(5, 8)]

times = np.empty([3, len(SZ), len(ITERS)], np.float64)

for (isz, sz) in enumerate(SZ):
    for (iiters, iters) in enumerate(ITERS):
        [cpu, gpu_toGpu, gpu_set] = benchmark(sz, iters)
        times[:,isz,iiters] = [cpu, gpu_toGpu, gpu_set]

np.savez('outData.npz', size=SZ, iterations=ITERS, times=times)
