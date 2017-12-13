#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:29:44 2015

@author: kerstin.cramer
"""

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

data = np.load('benchmark002_outData_tegner_K80_20171006.npz')
fig_name = 'benchmark002_outData_tegner_K80.eps'


times = data['times']
iters = data['iterations']
size = data['size']

iteration = 1;
siz = 7;

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(size,times[0,:,iteration],label="CPU",linewidth=1.5)
ax1.plot(size,times[1,:,iteration],label="GPU to_gpu",linewidth=1.5)
ax1.plot(size,times[2,:,iteration],label="GPU set",linewidth=1.5)
ax1.legend(loc=2,fontsize=14)
##plt.yscale('log');
ax1.set_xlabel('Matrix Size',fontsize=14)
##plt.ylim([0,0.03]);
ax1.set_ylabel('Elapsed Time [s]' , fontsize=14)
ax1.set_title('Iteration = ' + str(iters[iteration]), fontsize= 14)
ax1.tick_params(axis='both', which='major', labelsize=14)
##plt.figtext(0,1,title,fontsize='11', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.figtext(0,1.1, Mass_Flux_str,fontsize='9', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.tight_layout();

ax2.plot(iters,times[0,siz,:],label="CPU",linewidth=1.5)
ax2.plot(iters,times[1,siz,:],label="GPU to_gpu",linewidth=1.5)
ax2.plot(iters,times[2,siz,:],label="GPU set",linewidth=1.5)
ax2.legend(loc=2,fontsize=14)
ax2.set_xlabel('Iterations',fontsize=14)
ax2.set_ylabel('Elapsed Time [s]',fontsize=14 )
ax2.set_yticks([0,1,2,3])
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title('Matrix Size = ' + str(size[siz]),fontsize=14)
pylab.tight_layout()
pylab.savefig(fig_name, dpi=100,bbox_inches='tight');
#pylab.show()
