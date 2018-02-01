import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

data = np.load('benchmark002_outData.npz')
fig_name = 'benchmark002_outData_gaia_K80.eps'

times = data['times']
iters = data['iterations']
size = data['size']

iteration = 2;
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
ax1.plot(size,times[2,:,iteration],label="cKernel",linewidth=1.5)
ax1.plot(size,times[3,:,iteration],label="EW w. loop",linewidth=1.5)
ax1.plot(size,times[4,:,iteration],label="EW",linewidth=1.5)

ax1.legend(ncol=2, loc=2,fontsize=legend_size)
ax1.set_xlabel('Vector Length',fontsize=main_text_size)
ax1.set_ylabel('Elapsed Time [s]' , fontsize=main_text_size)
ax1.set_title('Iterations = ' + str(iters[iteration]), fontsize=title_size)
ax1.tick_params(axis='both', which='major', labelsize=main_text_size)
ax1.set_xlim([50,275]);

# bottom figure
ax2.plot(iters,times[0,siz,:],label="CPU",linewidth=1.5)
ax2.plot(iters,times[1,siz,:],label="gpuarray",linewidth=1.5)
ax2.plot(iters,times[2,siz,:],label="cKernel",linewidth=1.5)
ax2.plot(iters,times[3,siz,:],label="EW w. loop",linewidth=1.5)
ax2.plot(iters,times[4,siz,:],label="EW",linewidth=1.5)

ax2.legend(ncol=2, loc=2,fontsize=legend_size)
ax2.set_xlabel('Iterations',fontsize=main_text_size)
ax2.set_ylabel('Elapsed Time [s]',fontsize=main_text_size)
ax2.set_title('Vector Length = ' + str(size[siz]),fontsize=title_size)
ax2.tick_params(axis='both', which='major', labelsize=main_text_size)

# global tweaks and exports
pylab.tight_layout()
pylab.savefig(fig_name, dpi=100,bbox_inches='tight');
