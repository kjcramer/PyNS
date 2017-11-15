# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler

# Utils
import time

# Standard Python modules
import numpy as np

# ==============================================================

# tune these numbers!!!
blocks = 64
block_size = 128

# generate test array and push to gpu
a_cpu = np.random.random([2, 2, 3]).astype(np.float32)
a_gpu = gpuarray.to_gpu(a_cpu)

mod_old = compiler.SourceModule("""
__global__ void doublify(float *a)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int x_width = blockDim.x * gridDim.x;
  int y_width = blockDim.y * gridDim.y;
  for(int idz = 0; idz < 3; idz++)
  {
      int flat_id = idx + x_width * idy + (x_width * y_width) * idz;
      a[flat_id] *= 2;
  }
}
""")

mod = compiler.SourceModule("""
__global__ void doublify(float *a)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;
  int x_width = blockDim.x * gridDim.x;
  int y_width = blockDim.y * gridDim.y;
  int flat_id = idx + x_width * idy + (x_width * y_width) * idz;
  a[flat_id] *= 2;
}
""")


# get the function from the compiled source code
doublify = mod.get_function("doublify")

# call the kernel on the card
# may complain!
doublify(a_gpu, grid=(blocks, 1), block=(block_size, 1, 1))

a_gpu_fetch = a_gpu.get()

print(a_cpu)
print("---")
print(a_gpu)

