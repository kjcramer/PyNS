"""
Matrix-vector product, including booundary values, for PyNS matrix format.
"""

# Standard Python modules
from pyns.standard import *

# Specific Python modules
import time

# PyNS modules
from pyns.constants import W, E, S, N, B, T
from pyns.operators import cat_x, cat_y, cat_z

# =============================================================================
def mat_vec_bnd(a, phi, gpu=False):
# -----------------------------------------------------------------------------
    """
    Args:
      a:   Object of the type "Matrix", holding the matrix for multiplication.
      phi: Three-dimensional array holding a vector for multiplication.

    Returns:
      r: Result of the matrix-vector product, which is a vector stored
         in a three-dimensional array.
    """

    phi.exchange()
    
    if gpu:

        import pycuda.driver as cuda
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
	from numpy import shape
	from numpy import pad
        
	#start_gpu=time.time()
    
        # initialize and push data to gpu
        r_gpu = gpuarray.zeros(phi.val.shape,phi.val.dtype)
        a_gpu = gpuarray.to_gpu(a.C)
        phi_gpu = gpuarray.to_gpu(phi.val)
        
        r_gpu = a_gpu * phi_gpu
    
        a_gpu.set(a.W)
        phi_gpu.set(cat_x( (phi.bnd[W].val[ :1,:,:], 
                            phi.val       [:-1,:,:]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
    
        a_gpu.set(a.E)
        phi_gpu.set(cat_x( (phi.val       [ 1:,:,:], 
                            phi.bnd[E].val[ :1,:,:]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
    
        a_gpu.set(a.S)
        phi_gpu.set(cat_y( (phi.bnd[S].val[:, :1,:], 
                            phi.val       [:,:-1,:]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
        
        a_gpu.set(a.N)    
        phi_gpu.set(cat_y( (phi.val       [:, 1:,:], 
                            phi.bnd[N].val[:, :1,:]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
        
        a_gpu.set(a.B)
        phi_gpu.set(cat_z( (phi.bnd[B].val[:,:, :1], 
                            phi.val       [:,:,:-1]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
    
        a_gpu.set(a.T)
        phi_gpu.set(cat_z( (phi.val       [:,:, 1:], 
                            phi.bnd[T].val[:,:, :1]) ))
        r_gpu = r_gpu - a_gpu * phi_gpu
        r_gpu = r_gpu.get()      
    
        #stop_gpu=time.time()
        #print("GPU time: %2.3e s" %(stop_gpu-start_gpu))
   
        return r_gpu
   
   
    #start_cpu = time.time()

    r = zeros(phi.val.shape)
    
    r[:]  = a.C[:] * phi.val[:]

    r[:] -= a.W[:] * cat_x( (phi.bnd[W].val[ :1,:,:], 
                             phi.val       [:-1,:,:]) )

    r[:] -= a.E[:] * cat_x( (phi.val       [ 1:,:,:], 
                             phi.bnd[E].val[ :1,:,:]) )
    
    r[:] -= a.S[:] * cat_y( (phi.bnd[S].val[:, :1,:], 
                             phi.val       [:,:-1,:]) )

    r[:] -= a.N[:] * cat_y( (phi.val       [:, 1:,:], 
                             phi.bnd[N].val[:, :1,:]) )
    
    r[:] -= a.B[:] * cat_z( (phi.bnd[B].val[:,:, :1], 
                             phi.val       [:,:,:-1]) )

    r[:] -= a.T[:] * cat_z( (phi.val       [:,:, 1:], 
                             phi.bnd[T].val[:,:, :1]) )
    
    #stop_cpu=time.time()
    #print("CPU time: %2.3e s" %(stop_cpu-start_cpu))
        
    return r  # end of function
