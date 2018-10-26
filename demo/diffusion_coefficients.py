# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:48:45 2018

@author: kerstin.cramer
"""

#!/usr/bin/python

import sys
sys.path.append("../..")

from pyns.physical           import properties

import numpy as np
from scipy.constants import R
from pyns.demo.p_v_sat import *

# d is thickness in m, kap is thermal conductivity in W/mK
# eps is porosity, tau is tortuosity
# r is pore radius

pi = np.pi

# thermodynamic conditions:
t = 343.15
p = 1e5
pv = p_v_sat(t-273.15)

# thermodynamic properties
diff = 4.0E-4
M_H2O = 18e-3
M_air = 28e-3
[_, mu, _, _] = properties.air(round(t-273.15,-1),np.ones(1))


# membrane properties
tau = 2.0
r = np.array([0.11e-6, 0.1e-6, 0.225e-6])
d = np.array([81e-6, 65e-6, 65e-6])
eps = np.array([80.0, 85.0, 85.0])

# vapor content (not needed right now)
x = pv*M_H2O/(p*M_air)

# Knudsen diffusion [s/m] # remove d to have [s]
C_K = 2.0*eps*r/(3.0*tau*d)  \
              *np.power(8.0*M_H2O/(t*R*pi),0.5)

# Molecular diffusion [s/m] # remove d to have [s]        
C_M = eps*p*diff/(d*tau*R*t*(p-pv))

# Viscous flow coefficient [s]
C_V = eps*r*r*M_H2O*pv/(8.0*R*t*tau*mu)

# Permeability [m^2]
k = eps * r * r / (8.0 * tau)