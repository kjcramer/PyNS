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

dataset_id = "cpu_tegner"

mesh_sizes = ["64_16_16",
              "128_16_16",
              "128_32_32",
              "256_32_32",
              "256_64_64"]

# every simulation is run 3 times
runs = range(1,4)

# 9 different information, 160 time steps coming from each out.txt file
results_avgd = np.zeros((9, 160, len(mesh_sizes)))

for ii, mesh_size in enumerate(mesh_sizes):
    # 9 different information, 160 time steps and 3 runs
    avg = np.zeros((9, 160, 3))
    # load and compute average from 3 runs
    for run in runs:
        avg[:,:,run-1] = results_parser("../" + dataset_id + "_" +
                                        str(mesh_size) +
                                        "_run" + str(run) +
                                        "/out.txt")
    avg = np.sum(avg, axis=2) / 3

    results_avgd[:,:,ii] = avg

# summing over all iteration times and number of iterations
results_avgd_sum = np.sum(results_avgd, axis=1)

# plotting
plt.ion()
plt.plot(results_avgd_sum[6,:]/results_avgd_sum[7,:])
plt.xticks(range(len(mesh_sizes)), [x.replace("_","x") for x in mesh_sizes])
plt.xlabel("Mesh size")
plt.ylabel("Average time per iteration [s]")
