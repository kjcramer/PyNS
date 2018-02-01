import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

data = np.load('benchmark001_outData_tegner_K80_20171006.npz')
fig_name = 'benchmark001_outData_tegner_K80.eps'

times = data['times']
iters = data['iterations']
size = data['size']

iteration = 1;
siz = 7;

# cosmetics
main_text_size = 12
title_size = 14
legend_size = 10

# figure
fig, (ax1,ax2) = plt.subplots(2,1)

# top figure
ax1.plot(size,times[0,:,iteration],label="CPU",linewidth=1.5)
ax1.plot(size,times[1,:,iteration],label="GPU to_gpu",linewidth=1.5)
ax1.plot(size,times[2,:,iteration],label="GPU set",linewidth=1.5)

ax1.legend(loc=2,fontsize=legend_size)
ax1.set_xlabel('Matrix Size',fontsize=main_text_size)
ax1.set_ylabel('Elapsed Time [s]' , fontsize=main_text_size)
ax1.set_title('Iterations = ' + str(iters[iteration]), fontsize=title_size)
ax1.tick_params(axis='both', which='major', labelsize=main_text_size)

# bottom figure
ax2.plot(iters,times[0,siz,:],label="CPU",linewidth=1.5)
ax2.plot(iters,times[1,siz,:],label="GPU to_gpu",linewidth=1.5)
ax2.plot(iters,times[2,siz,:],label="GPU set",linewidth=1.5)

ax2.legend(loc=2,fontsize=legend_size)
ax2.set_xlabel('Iterations',fontsize=main_text_size)
ax2.set_ylabel('Elapsed Time [s]',fontsize=main_text_size)
ax2.set_title('Matrix Size = ' + str(size[siz]),fontsize=title_size)
ax2.tick_params(axis='both', which='major', labelsize=main_text_size)
ax2.set_yticks([0,1,2,3])

# global tweaks and exportxs
pylab.tight_layout()
pylab.savefig(fig_name, dpi=100,bbox_inches='tight');
