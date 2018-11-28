#!/usr/bin/python

import sys
sys.path.append("../..")

from pyns.configs.normal_config_8 import *
from pyns.configs.normal_config_2 import *
from pyns.configs.normal_config_05 import *

t_h_in = int(float(sys.argv[1]))
u_h_in = float(sys.argv[2])
airgap = sys.argv[3]

if sys.argv[4] == 'True':
  restart = True
elif sys.argv[4] == 'False':
  restart = False
else:
  print('restart argument not valid')

if airgap == '05':
  print('=============== running normal_config_05 now ==============' )
  normal_config_05(t_h_in, u_h_in, 70000, restart)
elif airgap == '2':
  print('=============== running normal_config_2 now ==============' )
  normal_config_2(t_h_in, u_h_in, 70000, restart)
elif airgap == '8':
  print('=============== running normal_config_8 now ==============' )
  normal_config_8(t_h_in, u_h_in, 70000, restart)
else:
  print( 'airgap info not valid')

exit()
