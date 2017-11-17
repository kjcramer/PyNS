"""
Preconditioned Bi-Conjugate Gradient Stabilized (BiCGStab) solver.

Source:
  http://www.netlib.org/templates/templates.pdf
"""

from __future__ import print_function

# Specific Python modules
import time

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants      import *
from pyns.display        import write
from pyns.discretization import Unknown

# Sisters from this module
from pyns.solvers.mat_vec_bnd import mat_vec_bnd
from pyns.solvers.vec_vec     import vec_vec
from pyns.solvers.norm        import norm

# =============================================================================
def bicgstab(a, phi, b, tol, 
             verbose = False,
             max_iter = -1):
# -----------------------------------------------------------------------------
    """
    Args:
      a: ...... Object of the type "Matrix", holding the system matrix.
      phi: .... Object of the type "Unknown" to be solved.
      b: ...... Three-dimensional array holding the source term.
      tol: .... Absolute solver tolerance
      verbose:  Logical variable setting if solver will be verbose (print
                info on Python console) or not.
      max_iter: Maxiumum number of iterations.

    Returns:
      x: Three-dimensional array with solution.
    """

    # if gpu == True, run CUDA-accelerated version of routines
    gpu = False

    if verbose is True:
        write.at(__name__)

    # Helping variable
    x = phi.val

    # Initialize arrays
    p       = zeros(x.shape)
    p_hat   = Unknown("vec_p_hat", phi.pos, x.shape, -1, per=phi.per, 
                      verbose=False)
    r       = zeros(x.shape)
    r_tilda = zeros(x.shape)
    s       = zeros(x.shape)
    s_hat   = Unknown("vec_s_hat", phi.pos, x.shape, -1, per=phi.per, 
                      verbose=False)
    v       = zeros(x.shape)

    # r = b - A * x
    r[:,:,:] = b[:,:,:] - mat_vec_bnd(a, phi, gpu)

    # Chose r~
    r_tilda[:,:,:] = r[:,:,:]

    # ---------------
    # Iteration loop
    # ---------------

    start = time.time()    
    
    if max_iter == -1:
        max_iter = prod(phi.val.shape)
        
    for i in range(1, max_iter):

        if verbose is True:
            print("  iteration: %3d:" % (i), end = "" )

        # rho = r~ * r
        rho = vec_vec(r_tilda, r, gpu)

        # If rho == 0 method fails
        if abs(rho) < TINY * TINY:
            write.at(__name__)
            print("  Fails becuase rho = %12.5e" % rho)
            end = time.time() 
            print("Elapsed time in bigstab %2.3e" %(end - start))
            return x

        if i == 1:
            # p = r
            p[:,:,:] = r[:,:,:]

        else:
            # beta = (rho / rho_old)(alfa/omega)
            beta = rho / rho_old * alfa / omega

            # p = r + beta (p - omega v)
            p[:,:,:] = r[:,:,:] + beta * (p[:,:,:] - omega * v[:,:,:])

        # Solve M p_hat = p
        p_hat.val[:,:,:] = p[:,:,:] / a.C[:,:,:]

        # v = A * p^
        v[:,:,:] = mat_vec_bnd(a, p_hat, gpu)

        # alfa = rho / (r~ * v)
        alfa = rho / vec_vec(r_tilda, v, gpu)

        # s = r - alfa v
        s[:,:,:] = r[:,:,:] - alfa * v[:,:,:]

        # Check norm of s, if small enough set x = x + alfa p_hat and stop
        res = norm(s)
        if res < tol:
            if verbose is True == True:  
                write.at(__name__)
                print("  Fails because rho = %12.5e" % rho)
            x[:,:,:] += alfa * p_hat.val[:,:,:]
            end = time.time() 
            print("Elapsed time in bigstab %2.3e" %(end - start))
            return x

        # Solve M s^ = s
        s_hat.val[:,:,:] = s[:,:,:] / a.C[:,:,:]

        # t = A s^
        t = mat_vec_bnd(a, s_hat, gpu)  

        # omega = (t * s) / (t * t)
        omega = vec_vec(t, s, gpu) / vec_vec(t, t, gpu)

        # x = x + alfa p^ + omega * s^
        x[:,:,:] += alfa * p_hat.val[:,:,:] + omega * s_hat.val[:,:,:]

        # r = s - omega q^
        r[:,:,:] = s[:,:,:] - omega * t[:,:,:]

        # Compute residual
        res = norm(r)

        if verbose is True:
            print("%12.5e" %res)

        # If tolerance has been reached, get out of here
        if res < tol:
            end = time.time() 
            print("Elapsed time in bigstab %2.3e" %(end - start))
            return x

        # Prepare for next iteration
        rho_old = rho

    return x  # end of function


# == full gpu version =========================================================

    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    import numpy as np
   
    # push input to gpu
    phi_gpu.val = gpuarray.to_gpu(phi.val.astype(np.float32)) 
    phi_gpu.bnd[W].val = gpuarray.to_gpu(phi.bnd[W].val.astype(np.float32)) 
    phi_gpu.bnd[E].val = gpuarray.to_gpu(phi.bnd[E].val.astype(np.float32))
    phi_gpu.bnd[S].val = gpuarray.to_gpu(phi.bnd[S].val.astype(np.float32)) 
    phi_gpu.bnd[N].val = gpuarray.to_gpu(phi.bnd[N].val.astype(np.float32)) 
    phi_gpu.bnd[B].val = gpuarray.to_gpu(phi.bnd[B].val.astype(np.float32)) 
    phi_gpu.bnd[T].val = gpuarray.to_gpu(phi.bnd[T].val.astype(np.float32)) 

    a_gpu.c = gpuarray.to_gpu(a.c.astype(np.float32)) 
    a_gpu.w = gpuarray.to_gpu(a.w.astype(np.float32))
    a_gpu.e = gpuarray.to_gpu(a.e.astype(np.float32)) 
    a_gpu.s = gpuarray.to_gpu(a.s.astype(np.float32)) 
    a_gpu.n = gpuarray.to_gpu(a.n.astype(np.float32)) 
    a_gpu.b = gpuarray.to_gpu(a.b.astype(np.float32)) 
    a_gpu.t = gpuarray.to_gpu(a.t.astype(np.float32)) 

    b_gpu = gpuarray.to_gpu(b.astype(np.float32))

    tol_gpu = gpuarray.to_gpu(tol.astype(np.float32))

    # --- Helping variable
    # x = phi.val
    x_gpu = gpuarray.to_gpu(phi.val.astype(np.float32))


    # --- Initialize arrays
    # p       = zeros(x.shape)
    p_gpu = gpuarray.zeros(x_gpu.shape, x_gpu.dtype)
    # p_hat   = Unknown("vec_p_hat", phi.pos, x.shape, -1, per=phi.per, 
    #                   verbose=False)
    p_hat_gpu.val = gpuarray.zeros_like(x_gpu)
    p_hat_gpu.bnd[W].val = gpuarray.zeros(phi.bnd[W].val.shape, x_gpu.dtype)
    p_hat_gpu.bnd[E].val = gpuarray.zeros(phi.bnd[E].val.shape, x_gpu.dtype)
    p_hat_gpu.bnd[S].val = gpuarray.zeros(phi.bnd[S].val.shape, x_gpu.dtype)
    p_hat_gpu.bnd[N].val = gpuarray.zeros(phi.bnd[N].val.shape, x_gpu.dtype)
    p_hat_gpu.bnd[B].val = gpuarray.zeros(phi.bnd[B].val.shape, x_gpu.dtype)
    p_hat_gpu.bnd[T].val = gpuarray.zeros(phi.bnd[T].val.shape, x_gpu.dtype)

    # r       = zeros(x.shape)
    r_gpu = gpuarray.zeros_like(x_gpu)
    # r_tilda = zeros(x.shape)
    r_tilda_gpu = gpuarray.zeros_like(x_gpu)
    # s       = zeros(x.shape)
    s_gpu = gpuarray.zeros_like(x_gpu)
    #s_hat   = Unknown("vec_s_hat", phi.pos, x.shape, -1, per=phi.per, 
    #                  verbose=False)
    s_hat_gpu.val = gpuarray.zeros_like(x_gpu)
    s_hat_gpu.bnd[W].val = gpuarray.zeros(phi.bnd[W].val.shape, x_gpu.dtype)
    s_hat_gpu.bnd[E].val = gpuarray.zeros(phi.bnd[E].val.shape, x_gpu.dtype)
    s_hat_gpu.bnd[S].val = gpuarray.zeros(phi.bnd[S].val.shape, x_gpu.dtype)
    s_hat_gpu.bnd[N].val = gpuarray.zeros(phi.bnd[N].val.shape, x_gpu.dtype)
    s_hat_gpu.bnd[B].val = gpuarray.zeros(phi.bnd[B].val.shape, x_gpu.dtype)
    s_hat_gpu.bnd[T].val = gpuarray.zeros(phi.bnd[T].val.shape, x_gpu.dtype)
    
    # v       = zeros(x.shape)
    v_gpu = gpuarray.zeros_like(x_gpu)

    # --- r = b - A * x
    r[:,:,:] = b[:,:,:] - mat_vec_bnd(a, phi, gpu) # FIXME

    # --- Chose r~
    # r_tilda[:,:,:] = r[:,:,:]
    r_tilda_gpu = r_gpu.copy()

    # ---------------
    # Iteration loop
    # ---------------

    start = time.time()    
    
    if max_iter == -1:
        max_iter = prod(phi.val.shape) # FIXME
        
    for i in range(1, max_iter):

        if verbose is True:
            print("  iteration: %3d:" % (i), end = "" )

        # --- rho = r~ * r
        # rho = vec_vec(r_tilda, r, gpu)
        rho_gpu = vec_vec(r_tilda_gpu, r_gpu, gpu)

        # If rho == 0 method fails
        # if abs(rho) < TINY * TINY:
        if cumath.fabs(rho_gpu) < TINY * TINY:
            write.at(__name__)
            print("  Fails because rho = %12.5e" % rho)
            end = time.time() 
            print("Elapsed time in bigstab %2.3e" %(end - start))
            return x

        if i == 1:
            # p = r
            # p[:,:,:] = r[:,:,:]
            p_gpu = r_gpu.copy()

        else:
            # --- beta = (rho / rho_old)(alfa/omega)
            # beta = rho / rho_old * alfa / omega
            beta_cpu = rho_gpu / rho_old_gpu * alfa_gpu / omega_gpu

            # --- p = r + beta (p - omega v)
            # p[:,:,:] = r[:,:,:] + beta * (p[:,:,:] - omega * v[:,:,:])
            p_gpu = r_gpu + beta_gpu * (p_gpu * v_gpu)

        # --- Solve M p_hat = p
        p_hat.val[:,:,:] = p_gpu / a.C[:,:,:] # FIXME

        # --- v = A * p^
        # v[:,:,:] = mat_vec_bnd(a, p_hat)
        v[:,:,:] = mat_vec_bnd(a, p_hat)

        # alfa = rho / (r~ * v)
        alfa = rho / vec_vec(r_tilda, v, gpu)

        # s = r - alfa v
        s[:,:,:] = r[:,:,:] - alfa * v[:,:,:]

        # Check norm of s, if small enough set x = x + alfa p_hat and stop
        res = norm(s)
        if res < tol:
            if verbose is True == True:  
                write.at(__name__)
                print("  Fails because rho = %12.5e" % rho)
            x[:,:,:] += alfa * p_hat.val[:,:,:]
            end = time.time() 
            print("Elapsed time in bigstab %2.3e" %(end - start))
            return x
