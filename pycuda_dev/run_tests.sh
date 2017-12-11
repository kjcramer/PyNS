#!/bin/bash
# dirty but effective

python pycuda_thinner_collocated.py > out.txt
<<<<<<< HEAD
mkdir gpu_tegner_256_64_64_run1
mv fig* gpu_tegner_256_64_64_run1
mv out.txt gpu_tegner_256_64_64_run1

python pycuda_thinner_collocated.py > out.txt
mkdir gpu_tegner_256_64_64_run2
mv fig* gpu_tegner_256_64_64_run2
mv out.txt gpu_tegner_256_64_64_run2

python pycuda_thinner_collocated.py > out.txt
mkdir gpu_tegner_256_64_64_run3
mv fig* gpu_tegner_256_64_64_run3
mv out.txt gpu_tegner_256_64_64_run3
=======
mkdir gpu_gaia_128_32_32_run1
mv fig* gpu_gaia_128_32_32_run1
mv out.txt gpu_gaia_128_32_32_run1

python pycuda_thinner_collocated.py > out.txt
mkdir gpu_gaia_128_32_32_run2
mv fig* gpu_gaia_128_32_32_run2
mv out.txt gpu_gaia_128_32_32_run2

python pycuda_thinner_collocated.py > out.txt
mkdir gpu_gaia_128_32_32_run3
mv fig* gpu_gaia_128_32_32_run3
mv out.txt gpu_gaia_128_32_32_run3
>>>>>>> b01292157f9cb731ab19468dafeaa00746d379f8
