#!/bin/bash
# run all tests in a serial way.
#
# For each run:
# - make a subdirectory
# - save hostname and cpuinfo of the machine
# - save standard output and a final figure
# - move all these files in the subdirectory

prefix="local_"

mesh_sizes=( "64_16_16"
             "128_16_16" )
             # "128_32_32"
             # "256_32_32"
             # "256_64_64"
             # "512_64_64" )

suffix="_run1"

for mesh_size in "${mesh_sizes[@]}"; do
    folder_name="$prefix$mesh_size$suffix"
    mkdir "$folder_name"
    hostname > "$folder_name"/hostname.txt
    cat /proc/cpuinfo > "$folder_name"/cpuinfo.txt
    echo "Now running ${mesh_size}..."
    python pycuda_thinner_collocated.py "$mesh_size" > out.txt
    mv fig* "$folder_name"
    mv out.txt "$folder_name"
done
