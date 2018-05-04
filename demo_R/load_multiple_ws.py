# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:30:31 2018

@author: kerstin.cramer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import sys
sys.path.append("../..")

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *


text_id = 'Output_R.txt'
text_file = open(text_id, "w")
text_file.write("Config        m_j\n")

for i in range(1,4):
  for j in range(1,4):
    for k in range(0,5):
      data=np.load('ws_temp_Membrane_R_{}_{}_{}.npz'.format(i,j,k))
      m_j = data['arr_13']
      massflow = np.sum(np.sum(m_j))/(0.0015)*3600 
      text_file.write("R_{0:}_{1:}_{2:}   {3:2.3e}\n".format(i,j,k,massflow))
text_file.close()
      