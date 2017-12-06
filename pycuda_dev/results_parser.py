# usage: python results_parser.py out.txt

import numpy as np
import sys

# the file to parse

assert len(sys.argv) == 2, "Please specify an output file!"

path = str(sys.argv[1])

# read all lines
f = open(path,'r')
lines = f.readlines()
f.close()

# bicgstab occurs 4 times per time step; computing u, v, w, p
bicgstab_occurrence = 1

# empty list that will contain the time steps
time_step = []

# empty lists that will contain the elapsed times and iterations per variable
time_u = []
iterations_u = []
time_v = []
iterations_v = []
time_w = []
iterations_w = []
time_p = []
iterations_p = []

for line in lines:
    # extract time step
    text_ts = "Time step = "
    if text_ts in line:
        time_step.append(int(''.join(filter(str.isdigit, line))))

    # extract elapsed time and number of iterations in bicgstab
    text_bicgstab = "Elapsed time in bigstab"
    if text_bicgstab in line:
        # a typical line looks like
        #   Elapsed time in bigstab 2.057e-02 --- iterations: 75
        # so it is enough to check that the first character is a number
        bicgstab_info = [float(s) for s in line.split() if s[0].isdigit()]
        # reading u
        if bicgstab_occurrence % 4 == 1:
            time_u.append(bicgstab_info[0])
            iterations_u.append(int(bicgstab_info[1]))
        # reading v
        if bicgstab_occurrence % 4 == 2:
            time_v.append(bicgstab_info[0])
            iterations_v.append(int(bicgstab_info[1]))
        # reading w
        if bicgstab_occurrence % 4 == 3:
            time_w.append(bicgstab_info[0])
            iterations_w.append(int(bicgstab_info[1]))
        # reading p
        if bicgstab_occurrence % 4 == 0:
            time_p.append(bicgstab_info[0])
            iterations_p.append(int(bicgstab_info[1]))
        bicgstab_occurrence += 1

# somehow we should save/plot/put in an array/ decide what to do with all this data
