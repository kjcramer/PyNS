import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

data = np.load('benchmark003_outData_gaia_k80.npz')
fig_name = 'benchmark003_outData_gaia_K80.eps'

times = data['times']
iters = data['iterations']
size = data['size']

iteration = 0;
siz = 7;

# cosmetics
main_text_size = 12
title_size = 14
legend_size = 10

# figure
fig, (ax1,ax2) = plt.subplots(2,1)

# top figure
ax1.plot(size,times[0,:,iteration],label="CPU",linewidth=1.5)
ax1.plot(size,times[1,:,iteration],label="gpuarray",linewidth=1.5)
ax1.plot(size,times[2,:,iteration],label="redKernel",linewidth=1.5)

ax1.legend(loc=2,fontsize=legend_size)
ax1.set_xlabel('Matrix Size', fontsize=main_text_size)
ax1.set_ylabel('Elapsed Time [s]', fontsize=main_text_size)
ax1.set_title('All Matrix Sizes', fontsize=title_size)
ax1.tick_params(axis='both', which='major', labelsize=main_text_size)
ax1.set_xlim([50,525]);

# bottom figure
ax2.plot(size,times[0,:,iteration],label="CPU",linewidth=1.5)
ax2.plot(size,times[1,:,iteration],label="gpuarray",linewidth=1.5)
ax2.plot(size,times[2,:,iteration],label="redKernel",linewidth=1.5)

ax2.legend(loc=2,fontsize=legend_size)
ax2.set_xlabel('Matrix Size', fontsize=main_text_size)
ax2.set_ylabel('Elapsed Time [s]',fontsize=main_text_size)
ax2.set_title('Magnified View', fontsize=title_size)
ax2.tick_params(axis='both', which='major', labelsize=main_text_size)
ax2.set_xlim([50,210])
ax2.set_ylim([0,0.05])

# global tweaks and exports
plt.tight_layout();
pylab.savefig(fig_name, dpi=100,bbox_inches='tight');
