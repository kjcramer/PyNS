# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:42:49 2018

@author: kerstin.cramer
"""

import sys
sys.path.append("../..")
import numpy
from pyns.solvers import Matrix
from preconditioner import preconditioner_form
    
x = numpy.ones((5,5,5))
a = Matrix(x.shape)
numpy.fill_diagonal(a.C,5.0)
numpy.fill_diagonal(a.B,4.0)
numpy.fill_diagonal(a.S,6.0)
numpy.fill_diagonal(a.W,7.0)

m = preconditioner_form(a,x,True)