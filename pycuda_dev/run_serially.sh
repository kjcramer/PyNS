#!/bin/bash
# run all tests in a serial way.
#
# For each run:
# - make a subdirectory
# - save hostname and cpuinfo of the machine in the subdir
# - save standard output and a final figure in the subdir

dataset_id="tegner_cpu"

# mesh_sizes=( "64_16_16"
#              "128_16_16"
#              "128_32_32"
#              "256_32_32"
#              "256_64_64"
#              "512_64_64"
#              "512_128_128" )

mesh_sizes=( "512_128_128" )

suffix="run"${1}

for mesh_size in "${mesh_sizes[@]}"; do
    folder_name="${dataset_id}_${mesh_size}_${suffix}"
    mkdir "$folder_name"
    hostname > "$folder_name"/hostname.txt
    cat /proc/cpuinfo > "$folder_name"/cpuinfo.txt
    echo "Now running ${mesh_size} on core ${1}..."
    python pycuda_thinner_collocated.py "$mesh_size" "$folder_name" > "$folder_name"/out.txt
done
