# CUDA-related
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
import pycuda.tools as tools

# Utils
import time

# Standard Python modules
import numpy as np

# ==============================================================
# https://stackoverflow.com/a/5731911

(free,total)=cuda.mem_get_info()
print("Global memory occupancy:%f%% free"%(free*100/total))

for devicenum in range(cuda.Device.count()):
    device=cuda.Device(devicenum)
    attrs=device.get_attributes()

    #Beyond this point is just pretty printing
    print("\n===Attributes for device %d"%devicenum)
    for (key,value) in attrs.items():
        print("%s:%s"%(str(key),str(value)))

# ==============================================================

# tune these numbers!!!
blocks = 1
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
starttime = time.time()
doublify(a_gpu, grid=(blocks, 1), block=(block_size, 1, 1))
a_gpu_fetch = a_gpu.get()
stoptime = time.time()



# print(a_cpu)
# print("---")
# print(a_gpu)

print("GPU time: %2.3e" %(stoptime-starttime))
