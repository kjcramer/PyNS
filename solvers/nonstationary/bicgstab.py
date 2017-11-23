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
    verbose = True

# == full gpu version =========================================================

    if gpu:
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import pycuda.cumath as cumath
        import numpy as np
        from pyns.pycuda_dev import gpu_object

        # push input to gpu
        phi_gpu = gpu_object(np.shape(phi.val))
        phi_gpu.val = gpuarray.to_gpu(phi.val.astype(np.float64))
        phi_gpu.bnd[W].val[:1,:,:] = gpuarray.to_gpu(phi.bnd[W].val.astype(np.float64)) 
        phi_gpu.bnd[E].val[:1,:,:] = gpuarray.to_gpu(phi.bnd[E].val.astype(np.float64))
        phi_gpu.bnd[S].val[:,:1,:] = gpuarray.to_gpu(phi.bnd[S].val.astype(np.float64)) 
        phi_gpu.bnd[N].val[:,:1,:] = gpuarray.to_gpu(phi.bnd[N].val.astype(np.float64)) 
        phi_gpu.bnd[B].val[:,:,:1] = gpuarray.to_gpu(phi.bnd[B].val.astype(np.float64)) 
        phi_gpu.bnd[T].val[:,:,:1] = gpuarray.to_gpu(phi.bnd[T].val.astype(np.float64)) 
        
        # quick'n'dirty definition      
        agpu_object = namedtuple("agpu_object", "C W E S N B T")

        a_gpu = agpu_object
        a_gpu.C = gpuarray.to_gpu(a.C.astype(np.float64))
        a_gpu.W = gpuarray.to_gpu(a.W.astype(np.float64))
        a_gpu.E = gpuarray.to_gpu(a.E.astype(np.float64))
        a_gpu.S = gpuarray.to_gpu(a.S.astype(np.float64))
        a_gpu.N = gpuarray.to_gpu(a.N.astype(np.float64))
        a_gpu.B = gpuarray.to_gpu(a.B.astype(np.float64))
        a_gpu.T = gpuarray.to_gpu(a.T.astype(np.float64))
        
        b_gpu = gpuarray.to_gpu(b.astype(np.float64))
        
        tol_gpu = gpuarray.to_gpu(np.asarray(tol).astype(np.float64))
        
        # --- Helping variable
        # x = phi.val
        x_gpu = gpuarray.to_gpu(phi.val.astype(np.float64))
        
        
        # --- Initialize arrays
        # p       = zeros(x.shape)
        p_gpu = gpuarray.zeros(x_gpu.shape, x_gpu.dtype)
        # p_hat   = Unknown("vec_p_hat", phi.pos, x.shape, -1, per=phi.per, 
        #                   verbose=False)
        p_hat_gpu = gpu_object(np.shape(phi.val))
 
        # r       = zeros(x.shape)
        r_gpu = gpuarray.zeros_like(x_gpu)
        # r_tilda = zeros(x.shape)
        r_tilda_gpu = gpuarray.zeros_like(x_gpu)
        # s       = zeros(x.shape)
        s_gpu = gpuarray.zeros_like(x_gpu)
        #s_hat   = Unknown("vec_s_hat", phi.pos, x.shape, -1, per=phi.per, 
        #                  verbose=False)
        s_hat_gpu = gpu_object(np.shape(phi.val))
        
        # v       = zeros(x.shape)
        v_gpu = gpuarray.zeros_like(x_gpu)
        
        # --- r = b - A * x
        # r[:,:,:] = b[:,:,:] - mat_vec_bnd(a, phi, gpu)
        r_gpu = b_gpu - mat_vec_bnd(a_gpu, phi_gpu, gpu)
        
        # --- Chose r~
        # r_tilda[:,:,:] = r[:,:,:]
        r_tilda_gpu = r_gpu.copy()
        
        # ---------------
        # Iteration loop
        # ---------------
        
        start = time.time()    
        
        if max_iter == -1:
            # max_iter = prod(phi.val.shape)
            max_iter = prod(phi_gpu.val.shape)
            
        for i in range(1, max_iter):
        
            if verbose is True:
                print("  iteration: %3d:" % (i), end = "" )
        
            # --- rho = r~ * r
            # rho = vec_vec(r_tilda, r, gpu)
            rho_gpu = vec_vec(r_tilda_gpu, r_gpu, gpu)
            # DEBUG
            np.savez('L138_gpu_iter_' + str(i)  + '.npz',
                     rho_gpu = rho_gpu,
                     r_tilda_gpu = r_tilda_gpu,
                     r_gpu = r_gpu)

        
            # If rho == 0 method fails
            # if abs(rho) < TINY * TINY:
            if cumath.fabs(rho_gpu).get() < TINY * TINY: 
                write.at(__name__)
                print("  Fails because rho = %12.5e" % rho_gpu.get())
                end = time.time() 
                print("Elapsed time in bigstab %2.3e" %(end - start))
                return x_gpu.get()
        
            if i == 1:
                # p = r
                # p[:,:,:] = r[:,:,:]
                p_gpu = r_gpu.copy()
        
            else:
                # --- beta = (rho / rho_old)(alfa/omega)
                # beta = rho / rho_old * alfa / omega
                beta_gpu = rho_gpu / rho_old_gpu * alfa_gpu / omega_gpu

                # --- p = r + beta (p - omega v)
                # p[:,:,:] = r[:,:,:] + beta * (p[:,:,:] - omega * v[:,:,:])
                p_gpu = r_gpu + beta_gpu.get() * (p_gpu - omega_gpu.get() * v_gpu)
                # print('p_gpu = ', p_gpu)

            # --- Solve M p_hat = p
            # p_hat.val[:,:,:] = p[:,:,:] / a.C[:,:,:]
            p_hat_gpu.val = p_gpu / a_gpu.C
        
            # --- v = A * p^
            # v[:,:,:] = mat_vec_bnd(a, p_hat)
            v_gpu = mat_vec_bnd(a_gpu, p_hat_gpu, gpu)
        
            # --- alfa = rho / (r~ * v)
            # alfa = rho / vec_vec(r_tilda, v, gpu)
            alfa_gpu = rho_gpu / vec_vec(r_tilda_gpu, v_gpu, gpu)
        
            # --- s = r - alfa v
            # s[:,:,:] = r[:,:,:] - alfa * v[:,:,:]
            # FIXME alfa_gpu.get() is nonsensical, can we broadcast from gpu to gpu?
            s_gpu = r_gpu - alfa_gpu.get() * v_gpu
        
            # --- Check norm of s, if small enough set x = x + alfa p_hat and stop
            # res = norm(s)
            res_gpu = cumath.sqrt( gpuarray.dot(s_gpu, s_gpu))
            # if res < tol:
            if res_gpu.get() < tol_gpu.get():
                if verbose is True == True:  
                    write.at(__name__)
                    print("  Fails because rho = %12.5e" % rho_gpu.get())
                x_gpu += alfa_gpu.get() * p_hat_gpu.val
                end = time.time() 
                print("Elapsed time in bigstab %2.3e" %(end - start))
                return x_gpu.get()
        
            # --- Solve M s^ = s
            # s_hat.val[:,:,:] = s[:,:,:] / a.C[:,:,:]
            s_hat_gpu.val = s_gpu / a_gpu.C
        
            # --- t = A s^
            # t = mat_vec_bnd(a, s_hat, gpu)  
            t_gpu = mat_vec_bnd(a_gpu, s_hat_gpu, gpu)
        
            # --- omega = (t * s) / (t * t)
            # omega = vec_vec(t, s, gpu) / vec_vec(t, t, gpu)
            omega_gpu = vec_vec(t_gpu, s_gpu, gpu) / vec_vec(t_gpu, t_gpu, gpu)
        
            # --- x = x + alfa p^ + omega * s^
            # x[:,:,:] += alfa * p_hat.val[:,:,:] + omega * s_hat.val[:,:,:]
            x_gpu += alfa_gpu.get() * p_hat_gpu.val + omega_gpu.get() * s_hat_gpu.val
        
            # --- r = s - omega q^
            # r[:,:,:] = s[:,:,:] - omega * t[:,:,:]
            r_gpu = s_gpu - omega_gpu.get() * t_gpu
        
            # --- Compute residual
            # res = norm(r)
            res_gpu = cumath.sqrt( gpuarray.dot(r_gpu, r_gpu))
        
            if verbose is True:
                # print("%12.5e" %res_gpu)
                print("%12.5e" %res_gpu.get())
        
            # If tolerance has been reached, get out of here
            # if res_gpu < tol:
            if res_gpu.get() < tol_gpu.get():
                end = time.time() 
                print("Elapsed time in bigstab %2.3e" %(end - start))
                return x_gpu.get()
        
            # --- Prepare for next iteration
            # rho_old = rho
            rho_old_gpu = rho_gpu
        
        return x_gpu.get()  # end of function
    

# == full cpu version =========================================================

    # DEBUG -- np only needed to save data
    import numpy as np

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
        # DEBUG
        np.savez('L138_cpu_iter_' + str(i)  + '.npz',
                 rho = rho,
                 r_tilda = r_tilda,
                 r = r)

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
            # print('p = ', p)

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


