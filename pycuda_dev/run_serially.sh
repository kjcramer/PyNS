#!/bin/bash -l
# run all tests in a serial way.
#
# For each run:
# - make a subdirectory
# - save hostname and cpuinfo of the machine in the subdir
# - save standard output and a final figure in the subdir

## Modules to preload  (from launcher-scripts/bash/serial/launcher_serial.sh
## MODULE_TO_LOAD=(lang/Python/3.5.3-intel-2017)

## Load the required modules
##for m in ${MODULE_TO_LOAD[*]}; do
##      module load $m
##done

dataset_id="gaia_cpu"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

mesh_sizes=(   "64_16_16"
               "128_16_16"
               "128_32_32"
               "256_32_32"
	       "256_64_64" ) 
             # "512_64_64"   
	     # "512_128_128" )

suffix="run"${1}

for mesh_size in "${mesh_sizes[@]}"; do
    folder_name="${dataset_id}_${mesh_size}_${suffix}"
    mkdir "$folder_name"
    hostname > "$folder_name"/hostname.txt
    cat /proc/cpuinfo > "$folder_name"/cpuinfo.txt
    echo "Now running ${mesh_size} on core ${1}..."
    python2.7 pycuda_thinner_collocated.py "$mesh_size" "$folder_name" > "$folder_name"/out.txt
done
