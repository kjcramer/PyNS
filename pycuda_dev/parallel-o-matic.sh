#!/bin/bash -l
# .......... The PARALLEL-O-MATIC-6000 (TM) ..........
# A bash creation by Kerstin and Luca
#
# This starts as many instances of our validation code as physical
# cores. It starts all processes in background.
#

# get number of cores installed (assume machine has hyperthreading)
n_cores=$(expr `nproc` / 2)

# start one process per physical core
for my_pid in $(seq 0 $(expr $n_cores - 1)); do
    numactl -C $(expr $my_pid \* 2) ./run_serially.sh $my_pid &
done
