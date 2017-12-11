#!/bin/bash
# dirty but effective

python pycuda_thinner_collocated.py > out.txt
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
