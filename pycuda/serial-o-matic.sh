#!/bin/bash -l
# .......... The SERIAL-O-MATIC-6000 (TM) ..........
# A bash creation by Kerstin and Luca
#
# This starts sequential instances of our validation code.
#

# number of times the instance should be run
n_runs=3

# run n_runs times, sequentially
for my_pid in $(seq 0 $(expr $n_runs - 1)); do
    ./run_serially.sh $my_pid
done
