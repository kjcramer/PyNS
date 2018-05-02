#!/bin/bash -l

#OAR -l nodes=1/core=11,walltime=120:00:00
##OAR -t gpu
##OAR -p "gputype='K80'"
#OAR -n pyns_membrane_vertical

# run pyns membrane vertical module on gaia
module load lang/Python/3.6.0-foss-2017a
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python validation_40.py

