'''
A script to compare results.

Generates a plot where:
  -- on the x axis is the mesh size
  -- on the y axis is the time normalized wrt the number of iterations

'''

from results_parser import results_parser

import numpy as np
from collections import namedtuple

import matplotlib
from matplotlib import pyplot as plt

dataset_id = "gaia_cpu"

mesh_sizes = ["64_16_16",
              "128_16_16",
              "128_32_32",
              "256_32_32",
              "256_64_64",
              "512_64_64",
              "512_128_128"]

# number of time steps run
ts = 200

# every simulation is run run_n times, then the results will be averaged
run_n = 3
runs = range(0, run_n)

# 9 different information, ts time steps coming from each out.txt file
results_avgd = np.zeros((9, ts, len(mesh_sizes)))

for ii, mesh_size in enumerate(mesh_sizes):
    # 9 different information, 160 time steps
    avg = np.zeros((9, ts, run_n))
    # load and compute average from all the runs
    for run in runs:
        avg[:,:,run-1] = results_parser("../" + dataset_id + "_" +
                                        str(mesh_size) +
                                        "_run" + str(run) +
                                        "/out.txt")
    avg = np.sum(avg, axis=2) / run_n

    results_avgd[:,:,ii] = avg

np.savez(('%s.npz'%(dataset_id)),mesh_sizes=mesh_sizes,results_avgd=results_avgd)

# summing over all iteration times and number of iterations
results_avgd_sum = np.sum(results_avgd, axis=1)

# plotting
plt.ion()
plt.plot(results_avgd_sum[6,:]/results_avgd_sum[7,:])
plt.xticks(range(len(mesh_sizes)), [x.replace("_","x") for x in mesh_sizes])
plt.xlabel("Mesh size")
plt.ylabel("Average time per iteration [s]")
plt.savefig(('%s.png'%(dataset_id)),bbox_inches='tight')
