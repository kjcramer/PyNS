#!/bin/bash -l

#OAR -l nodes=1/core=1,walltime=120:00:00
#OAR -n pyns_membrane

# run pyns membrane configs on gaia
# input required:
# ${1} is the inlet temperature
# ${2} is the inlet velocity
# ${3} is the airgap in {05, 2, 8}
# ${4} is the restart from file option {True, False}

python2.7 normal_gaia.py ${1} ${2} ${3} ${4}
