#!/usr/bin/python

import sys
sys.path.append("../..")

from pyns.configs.normal_config_8 import *
from pyns.configs.normal_config_2 import *
from pyns.configs.normal_config_05 import *

t_h_in = int(float(sys.argv[1]))
u_h_in = float(sys.argv[2])
airgap = sys.argv[3]

if airgap == '05':
  print('=============== running normal_config_05 now ==============' )
  normal_config_05(t_h_in, u_h_in, 70000, True)
elif airgap == '2':
  print('=============== running normal_config_2 now ==============' )
  normal_config_2(t_h_in, u_h_in, 70000, True)
elif airgap == '8':
  print('=============== running normal_config_8 now ==============' )
  normal_config_8(t_h_in, u_h_in, 70000, True)
else:
  print( 'airgap info not valid')

exit()
