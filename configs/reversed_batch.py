#!/usr/bin/python

import sys
sys.path.append("../..")

#from pyns.configs.reversed_config_8 import *
#from pyns.configs.reversed_config_2 import *
#from pyns.configs.reversed_config_05 import *

from pyns.configs.reversed_config_8 import *
from pyns.configs.reversed_config_2 import *
from pyns.configs.reversed_config_05 import *

#%%
#u_h_in = [0.025, 0.05, 0.1]

#for uu in u_h_in:
reversed_config_8(80, 0.1, 130000, True)
