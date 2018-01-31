"""
Vector-dot product of two vectors stored as three-dimensional arrays.
"""

# Specific Python modules
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
import numpy as np

# Standard Python modules
# from pyns.standard import *

# =============================================================================
def benchmark(dimension, sourceModule=False):
# -----------------------------------------------------------------------------
    """
    Args:
        x: Three-dimensional array holding vector for multiplication.
        y: Three-dimensional array holding vector for multiplication.
    
        sourceModule: if True, run on cpu using a cuda kernel
    
    Returns:
      Result of the vector-dot product.
    
    Note:
      Try to find a better way to summ the elements of a matrix than
      sum(sum(sum()))
    """
    # test data
    x = np.random.random([dimension, dimension, dimension])
    y = np.random.random([dimension, dimension, dimension])

    
    # cuda-specific parameters
    block_size = 512
    blocks = x.size // block_size + 1
    
    # timers
    start = cuda.Event()
    end = cuda.Event()
    
    gpuArray = True;
    #sourceModule = True;
    reductionKernel = True;
    
    # ==== gpuarray ===========================================================
    if gpuArray:
        # populate the GPUArray with the values from x and y
        # return gpuarray.dot( gpuarray.to_gpu(x), gpuarray.to_gpu(y)).get()
        start.record()
        gpuArray_out =  gpuarray.dot( gpuarray.to_gpu(x), gpuarray.to_gpu(y)).get()
        end.record()
        end.synchronize()
        garray = start.time_till(end)*1e-3
        print("GPUArray time and result:")
        print("%fs, %s" % (garray, str(gpuArray_out)))
    # -------------------------------------------------------------------------
    
    # ==== sourceModule =======================================================
    if sourceModule:
        dest = np.zeros_like(x).astype(np.float32)
    
        mod = SourceModule(
            """
            __global__ void vec_vec_cpp(float *dest, float *a, float *b)
            {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            dest[i] = a[i] * b[i];
            }
            """
        )
            
        vec_vec_kernel = mod.get_function("vec_vec_cpp")
    
        start.record()
        vec_vec_kernel(cuda.Out(dest),
                       cuda.In(x.astype(np.float32)),
                       cuda.In(y.astype(np.float32)),
                       grid = (blocks, 1),
                       block = (block_size, 1, 1))
    
        sourceModule_out = np.sum(np.sum(np.sum(dest)))
        end.record()
        end.synchronize()
        cKernel = start.time_till(end)*1e-3
        #print("SourceModule time and result:")
        #print("%fs, %2.3e" % (secs, sourceModule_out))
    
        # return gpuarray.sum(dest)
    # -------------------------------------------------------------------------
    
    
    
    # ==== reduction kernel ===================================================
    if reductionKernel:
        reduction_kernel = ReductionKernel(np.float32, neutral="0",
                                           reduce_expr="a+b", map_expr="x[i]*y[i]",
                                           arguments="float *x, float *y")
    
        start.record()
        xgpu = gpuarray.to_gpu(x.astype(np.float32))
        ygpu = gpuarray.to_gpu(y.astype(np.float32))
        dest_reduction = reduction_kernel(xgpu, ygpu).get()
        end.record()
        end.synchronize()
        redKernel = start.time_till(end)*1e-3
        print("ReductionKernel time and result:")
        print("%fs, %s" % (redKernel, str(dest_reduction)))
    # -------------------------------------------------------------------------
    
        
    
    # ==== cpu ================================================================
    #return sum( sum( sum( multiply(x, y) ) ) )
    start.record()
    cpu_out = sum( sum( sum( np.multiply(x, y) ) ) )
    end.record()
    end.synchronize()
    cpu = start.time_till(end)*1e-3
    print("cpu time and result:")
    print("%fs, %s" % (cpu, str(cpu_out)))
    # -------------------------------------------------------------------------
    
    return cpu, garray, redKernel

# ============================================================================
# ============================================================================

import numpy as np

# array size, from 64 to 512 in 16-increments
SZ = [16*x for x in range(4,16)]
print(SZ)
ITERS = [ 1 ]

times = np.empty([3, len(SZ), len(ITERS)], np.float64)

for (isz, sz) in enumerate(SZ):
    for (iiters, iters) in enumerate(ITERS):
        [cpu, garray, redKernel] = benchmark(sz,False)
        times[:,isz,iiters] = [cpu, garray, redKernel]

np.savez('benchmark007_outData.npz', size=SZ, iterations=ITERS, times=times)
