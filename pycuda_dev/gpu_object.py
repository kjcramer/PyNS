"""
Defines class of the type "Unknown".  "Unknown" is the object which holds the
values inside the computational domain and on the boundaries.  Value inside
the domain is stored in a full three-dimensional array.  Boundary values are
also formally stored in three-dimensional arrays, but depending on their
position (W, E, S, N, B or T), one dimension is set to one. For boundaries,
also the type of boundary conditions are specified.

To access an unknown, say its name is "phi" later in the program, the
following syntax is used:

Value inside the domain, at the cell with coordinates (12, 34, 56):

  phi.val[12, 34, 56] = 0.0

Value on the east boundary, at coordinates (34, 56).

  phi.bnd[E].val[1, 34, 56] = 0.0

Type of boundary condition at the same boundary cell from above:

  phi.bnd[E].typ[1, 34, 56] == NEUMANN

Unknowns in PyNS are shown below for a two-dimensional grid (for simplicity):

  Here, collocated resolution is:                  Legend:

    nx = 6                                            o   ... scalars
    ny = 4                                           ---  ... u-velocities
                             [N]                      |   ... v-velocities

      +-------+-------+-------+-------+-------+-------+
      |       |       |       |       |       |       |
      |   o  ---  o  ---  o  ---  o  ---  o  ---  o   | j=ny-1
      |       |       |       |       |       |       |
      +---|---+---|---+---|---+---|---+---|---+---|---+     j=ny-2
      |       |       |       |       |       |       |
      |   o  ---  o  ---  o  ---  o  ---  o  ---  o   | ...
      |       |       |       |       |       |       |
 [W]  +---|---+---|---+---|---+---|---+---|---+---|---+     j=1     [E]
      |       |       |       |       |       |       |
      |   o  ---  o  ---  o  ---  o  ---  o  ---  o   | j=1
      |       |       |       |       |       |       |
      +---|---+---|---+---|---+---|---+---|---+---|---+     j=0  (v-velocity)
      |       |       |       |       |       |       |
      |   o  ---  o  ---  o  ---  o  ---  o  ---  o   | j=0      (scalar)
      |       |       |       |       |       |       |
      +-------+-------+-------+-------+-------+-------+
      .  i=0     i=1     ...     ...   i=nx-2  i=nx-1 .    (scalar)
      :      i=0      i=1    ...    i=nx-3  i=nx-2    :    (u-velocity)
      |                      [S]                      |
  y   |                                               |
 ^    |<--------------------------------------------->|
 |                      domain lenght
 +---> x

"""


from __future__ import print_function

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants import *
from pyns.operators import *
from pyns.display   import write

# Specific Python modules
import pycuda.gpuarray as gpuarray
import numpy as np

class gpu_object:

    # =========================================================================
    def __init__(self, res, 
                 verbose = False):
    # -------------------------------------------------------------------------
        """
        Args:
          res: ...... Vector specifying resolutions in "x", "y" and "z" 
                     

        Returns:
          Oh well, its own self, isn't it?
        """

        if verbose is True:
            write.at(__name__)  
    

        nx, ny, nz = res
        # Allocate memory space for new and old values
        self.val = gpuarray.zeros((nx, ny, nz),dtype=np.float32)

        # Create boundary tuple
        key = namedtuple("key", "typ val")
        self.bnd = (key(gpuarray.zeros((1,ny,nz), dtype=int), gpuarray.zeros((1,ny,nz), dtype=np.float32)),
                    key(gpuarray.zeros((1,ny,nz), dtype=int), gpuarray.zeros((1,ny,nz), dtype=np.float32)),
                    key(gpuarray.zeros((nx,1,nz), dtype=int), gpuarray.zeros((nx,1,nz), dtype=np.float32)),
                    key(gpuarray.zeros((nx,1,nz), dtype=int), gpuarray.zeros((nx,1,nz), dtype=np.float32)),
                    key(gpuarray.zeros((nx,ny,1), dtype=int), gpuarray.zeros((nx,ny,1), dtype=np.float32)),
                    key(gpuarray.zeros((nx,ny,1), dtype=int), gpuarray.zeros((nx,ny,1), dtype=np.float32)))

        # Prescribe default boundary conditions
        self.bnd[W].typ[0,:,:] = (np.ones((ny,nz))*NEUMANN).astype(int)
        self.bnd[E].typ[0,:,:] = (np.ones((ny,nz))*NEUMANN).astype(int)
        self.bnd[S].typ[:,0,:] = (np.ones((nx,nz))*NEUMANN).astype(int)
        self.bnd[N].typ[:,0,:] = (np.ones((nx,nz))*NEUMANN).astype(int)
        self.bnd[B].typ[:,:,0] = (np.ones((nx,ny))*NEUMANN).astype(int)
        self.bnd[T].typ[:,:,0] = (np.ones((nx,ny))*NEUMANN).astype(int)

        self.bnd[W].val[0,:,:] = (np.zeros((ny,nz))).astype(np.float32)
        self.bnd[E].val[0,:,:] = (np.zeros((ny,nz))).astype(np.float32)
        self.bnd[S].val[:,0,:] = (np.zeros((nx,nz))).astype(np.float32)
        self.bnd[N].val[:,0,:] = (np.zeros((nx,nz))).astype(np.float32)
        self.bnd[B].val[:,:,0] = (np.zeros((nx,ny))).astype(np.float32)
        self.bnd[T].val[:,:,0] = (np.zeros((nx,ny))).astype(np.float32)

        if verbose is True:
            print("  Created unknown:", self.name)

        return  # end of function

    # =========================================================================
    def exchange(self):
    # -------------------------------------------------------------------------
        """
        Function to refresh buffers.  For the time being it only takes
        care of periodic boundary conditions, but in the future it may
        also refresh buffers used in parallel execution.

        Periodicity for scalar cells:

          - value in cell "0" is identical to value in "nx-1"

          .---<-------<-------<-------<-------<---.
          |                    send to buffers    |
          |               +--->------->------->---)--->------->---.
          |               |                       |               |
          v --+-------+---|---+-------+-------+---|---+-------+-- v
              |       |   ^   |       |       |   ^   |       |
          o   |   o   |   o   |   o   |   o   |   o   |   o   |   o
              |       |       |       |       |       |       |
          - --+-------+-------+-------+-------+-------+-------+-- -
         [W]     i=0     i=1     ...     ...   i=nx-2  i=nx-1    [E]
          =       =                                       =       =
         nx-2    nx-1                                     0       1
                  |        effective domain lenght        |
                  |<------------------------------------->|

        Periodicity for vector cells:

          - value in vector "0" is east from value in "nx-2"
          - value in vector "

                      .--->------->------->------->------->---.
                      |        send to buffers                |
              .---<---)---<-------<-------<-------<---.       |
              |       |                               |       |
              |-------|-------+-------+-------+-------|-------|
              v       ^       |       |       |       ^       v
             ---     ---     ---     ---     ---     ---     ---
              |       |       |       |       |       |       |
              +-------+-------+-------+-------+-------+-------+
             [W]     i=0     i=1     ...     ...   i=nx-2    [E]
              =                                               =
            nx-2                                              0
                  |        effective domain lenght        |
                  |<------------------------------------->|

        """

        if self.per[X] == True:
            if self.pos == X:
                self.bnd[W].val[:] = self.val[-1:,:,:]
                self.bnd[E].val[:] = self.val[ :1,:,:]
            else:
                self.bnd[W].val[:] = self.val[-2:-1,:,:]
                self.bnd[E].val[:] = self.val[ 1: 2,:,:]
        if self.per[Y] == True:
            if self.pos == Y:
                self.bnd[S].val[:] = self.val[:,-1:,:]
                self.bnd[N].val[:] = self.val[:, :1,:]
            else:
                self.bnd[S].val[:] = self.val[:,-2:-1,:]
                self.bnd[N].val[:] = self.val[:, 1: 2,:]
        if self.per[Z] == True:
            if self.pos == Z:
                self.bnd[B].val[:] = self.val[:,:,-1:]
                self.bnd[T].val[:] = self.val[:,:, :1]
            else:
                self.bnd[B].val[:] = self.val[:,:,-2:-1]
                self.bnd[T].val[:] = self.val[:,:, 1: 2]

        return  # end of function
