def results_parser(infile):
    """
    Parse the standard output of pycuda_thinner_collocated.py in order
    to gather information on its performance.

    Returns a 2D numpy array of shape (8, TS):

    * 8  is meant to contain
         0 time       elapsed by bicgstab in solving for u
         1 iterations .................................. u
         2 time       elapsed by bicgstab in solving for v
         3 iterations .................................. v
         4 time       elapsed by bicgstab in solving for w
         5 iterations .................................. w
         6 time       elapsed by bicgstab in solving for p
         7 iterations .................................. p
    * TS is the total number of time steps;


    ---
    Usage example
    ---
        bash> python pycuda_thinner_collocated.py > k80_tegner.out
      python> results_parser("k80_tegner.out")

    """

    import numpy as np
    import sys

    # read all lines
    f = open(infile,'r')
    lines = f.readlines()
    f.close()

    # bicgstab occurs 4 times per time step;
    # on the first occurrence u is computed, then v, w, p
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

    # parse all lines of the input file
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

    # put the gathered data into a numpy array
    return np.array([time_u, iterations_u,
                     time_v, iterations_v,
                     time_w, iterations_w,
                     time_p, iterations_p])
