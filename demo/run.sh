#!/bin/bash -l

#OAR -l nodes=1/core=16,walltime=120:00:00
#OAR -t gpu
##OAR -p "gputype='K80'"
#OAR -n pyns_membrane_vertical

# run pyns membrane vertical module on gaia

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python2.7 demo_membrane_vertical_module.py

