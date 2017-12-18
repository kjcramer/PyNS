#!/bin/bash
# run all tests in a serial way.
#
# For each run:
# - make a subdirectory
# - save hostname and cpuinfo of the machine
# - save standard output and a final figure
# - move all these files in the subdirectory

mesh_sizes=( "128_16_16"
             "256_64_64" )

for ii in "${mesh_sizes[@]}"
do
    echo "$ii"
done

# python pycuda_thinner_collocated.py > out.txt
# mkdir gpu_tegner_256_64_64_run1
# mv fig* gpu_tegner_256_64_64_run1
# mv out.txt gpu_tegner_256_64_64_run1

# python pycuda_thinner_collocated.py > out.txt
# mkdir gpu_tegner_256_64_64_run2
# mv fig* gpu_tegner_256_64_64_run2
# mv out.txt gpu_tegner_256_64_64_run2

# python pycuda_thinner_collocated.py > out.txt
# mkdir gpu_tegner_256_64_64_run3
# mv fig* gpu_tegner_256_64_64_run3
# mv out.txt gpu_tegner_256_64_64_run3
