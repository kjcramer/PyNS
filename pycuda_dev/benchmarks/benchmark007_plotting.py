#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:29:44 2015

@author: kerstin.cramer
"""

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib


data = np.load('benchmark007_outData.npz')
fig_name = 'benchmark007_outData_gaia_K80.eps'


times = data['times']
iters = data['iterations']
size = data['size']

iteration = 0;
siz = 7;

ax1 = plt.figure 
#fig, (ax1,ax2) = plt.subplots(2,1)
plt.plot(size,times[0,:,iteration],label="CPU",linewidth=1.8)
plt.plot(size,times[1,:,iteration],label="gpuarray",linewidth=1.8)
plt.plot(size,times[2,:,iteration],label="redKernel",linewidth=1.8)
plt.legend(loc=2,fontsize=20)
##plt.yscale('log');
plt.xlabel('Matrix Size',fontsize=20)
plt.xlim([50,525]);
plt.ylabel('Elapsed Time [s]' , fontsize=20)
#ax1.set_title('Iteration = ' + str(iters[iteration]), fontsize= 14)
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('ytick', labelsize=20)
##plt.figtext(0,1,title,fontsize='11', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.figtext(0,1.1, Mass_Flux_str,fontsize='9', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.tight_layout();

pylab.savefig(fig_name, dpi=100,bbox_inches='tight');
pylab.show()
