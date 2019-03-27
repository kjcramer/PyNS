# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:12:35 2017

@author: kerstin.cramer
"""

# Standard Python modules
from pyns.standard import *

import numpy as np

#--------------------------------------------------------------------------
def rho_salt(a,t,rho):
#--------------------------------------------------------------------------  
  # values taken from Millero and Huang, 2009

  # inputs: salt concentration in kg/kg, temperature in C and density in kg/m^3
  # output density in kg/m^3 corrected for presence of salt
  
  a = a*1000 # convert salt concentration to g/kg  
  
  # coefficients:
  a0 =  8.197247E-01
  a1 = -3.779454E-03
  a2 =  6.821795E-05
  a3 = -8.009571E-07
  a4 =  6.158885E-09
  a5 = -2.001919E-11
  
  b0 = -5.808305E-03
  b1 =  5.354872E-05
  b2 = -4.714602E-07
  
  c0 =  5.249266E-04
  
  if (isinstance(t,int) or isinstance(t,float)):
    if   (t >=0 and t<=90):
      A = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
      B = b0 + b1 * t + b2 * t**2
      C = c0
      rho_s= rho + A * a + B * a**1.5 + C * a**2
    else:
      rho_s= float("nan")
 
  else:
    rho_s=zeros(rho.shape)
    
    A = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    B = b0 + b1 * t + b2 * t**2
    C = c0
    rho_s= rho + A * a + B * a**1.5 + C * a**2
            
    rho_s[t<0]= float("nan")
    rho_s[t>90]= float("nan")
  
  return rho_s # end of function
