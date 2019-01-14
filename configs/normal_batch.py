#!/usr/bin/python

import sys
sys.path.append("../..")

from pyns.configs.normal_config_8 import *
from pyns.configs.normal_config_2 import *
from pyns.configs.normal_config_05 import *

#%%
#u_h_in = [0.025, 0.05, 0.1]

#for uu in u_h_in:
reversed_config_05(70, 0.1, 100000, True)
