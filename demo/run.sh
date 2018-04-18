#!/bin/bash -l

#OAR -l nodes=1/core=1,walltime=120:00:00
#OAR -n pyns_membrane

# run pyns membrane configs on gaia
# input filename required

python2.7 ${1}
