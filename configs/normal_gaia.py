#!/usr/bin/python

import sys
sys.path.append("../..")

from pyns.configs.normal_config_8 import *
from pyns.configs.normal_config_2 import *
from pyns.configs.normal_config_05 import *

sys.argv[1] = float(t_h_in)
sys.argv[2] = float(u_h_in)
sys.argv[3] = airgap

if airgap == '05':
  normal_config_05(t_h_in, u_h_in, 70000, False)
elif airgap == '2':
  normal_config_2(t_h_in, u_h_in, 70000, False)
elif airgap == '8':
  normal_config_8(t_h_in, u_h_in, 70000, False)

exit()
