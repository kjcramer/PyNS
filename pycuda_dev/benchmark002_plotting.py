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

times = data['times']
iters = data['iterations']
size = data['size']

iteration = 1;
siz = 7;

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(size,times[0,:,iteration],label="CPU")
ax1.plot(size,times[1,:,iteration],label="GPU_toGpu")
ax1.plot(size,times[2,:,iteration],label="GPU_set")
ax1.legend(loc=4)
##plt.yscale('log');
ax1.set_xlabel('Matrix Size')
##plt.ylim([0,0.03]);
ax1.set_ylabel('Elapsed Time [s]' )
ax1.set_title('Iteration = ' + str(iters[iteration]))
##plt.figtext(0,1,title,fontsize='11', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.figtext(0,1.1, Mass_Flux_str,fontsize='9', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.tight_layout();
##pylab.savefig(fig_name, dpi=100,bbox_inches='tight');

ax2.plot(iters,times[0,siz,:],label="CPU")
ax2.plot(iters,times[1,siz,:],label="GPU_toGpu")
ax2.plot(iters,times[2,siz,:],label="GPU_set")
ax2.legend(loc=4)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Elapsed Time [s]' )
ax2.set_title('Matrix Size = ' + str(size[siz]))
pylab.show()