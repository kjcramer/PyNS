# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:58:34 2018

@author: kerstin.cramer
"""

from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

#%% load gaia gpu data

dataset_id_gaia_gpu = "gaia_gpu"

data = np.load(('%s.npz'%(dataset_id_gaia_gpu)))
mesh_sizes = data['mesh_sizes']
results_gaia_gpu = data['results_avgd']

# summing over all iteration times and number of iterations
results_sum_gaia_gpu = np.sum(results_gaia_gpu, axis=1)

#%% load gaia cpu data

dataset_id_gaia_cpu = "gaia_cpu"

data = np.load(('%s.npz'%(dataset_id_gaia_cpu)))
mesh_sizes = data['mesh_sizes']
results_gaia_cpu = data['results_avgd']

# summing over all iteration times and number of iterations
results_sum_gaia_cpu = np.sum(results_gaia_cpu, axis=1)

#%% load tegner cpu data

dataset_id_tegner_cpu = "tegner_cpu"

data = np.load(('%s.npz'%(dataset_id_tegner_cpu)))
mesh_sizes = data['mesh_sizes']
results_tegner_cpu = data['results_avgd']

# summing over all iteration times and number of iterations
results_sum_tegner_cpu = np.sum(results_tegner_cpu, axis=1)

#%% plot final comparison plots average time per iteration

size=np.arange(len(mesh_sizes))
fig_name_1='Comparison_time_per_iteration.png'

# plotting average time per iteration 
plt.ion()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(size+1,results_sum_gaia_cpu[6,:]/results_sum_gaia_cpu[7,:],'--',color='g',label="CPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_sum_gaia_gpu[6,:]/results_sum_gaia_gpu[7,:],'-',color='g',label="GPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_sum_tegner_cpu[6,:]/results_sum_tegner_cpu[7,:],'--',color='k',label="CPU Tegner",linewidth=1.5)
ax1.legend(loc=2,fontsize=14)
ax1.set_xlabel('Mesh',fontsize=14)
##plt.ylim([0,0.03]);
ax1.set_ylabel('Average Time per Iteration [s]' , fontsize=14)
ax1.set_title('All Meshes:',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
##plt.figtext(0,1,title,fontsize='11', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.figtext(0,1.1, Mass_Flux_str,fontsize='9', bbox=dict(facecolor='none', edgecolor='black', pad=10.0));
##plt.tight_layout();

# magnifying view

ax2.plot(size[0:4]+1,results_sum_gaia_cpu[6,0:4]/results_sum_gaia_cpu[7,0:4],'--',color='g',label="CPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_sum_gaia_gpu[6,0:4]/results_sum_gaia_gpu[7,0:4],'-',color='g',label="GPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_sum_tegner_cpu[6,0:4]/results_sum_tegner_cpu[7,0:4],'--',color='k',label="CPU Tegner",linewidth=1.5)
ax2.legend(loc=2,fontsize=14)
ax2.set_xlabel('Mesh',fontsize=14)
##plt.ylim([0,0.03]);
ax2.set_xticks([1,2,3,4])
ax2.set_ylabel('Average Time per Iteration [s]' , fontsize=14)
ax2.set_title('Magnifying View:',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
pylab.tight_layout()
pylab.savefig(fig_name_1, dpi=100,bbox_inches='tight');

#%% plot final comparison plots total time

fig_name_2='Comparison_total_time.png' 

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(size+1,results_gaia_cpu[8,199,:]/3600,'--',color='g',label="CPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_gaia_gpu[8,199,:]/3600,'-',color='g',label="GPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_tegner_cpu[8,199,:]/3600,'--',color='k',label="CPU Tegner",linewidth=1.5)
ax1.legend(loc=2,fontsize=14)
ax1.set_xlabel('Mesh',fontsize=14)
ax1.set_ylabel('Elapsed Time [h]',fontsize=14 )
#ax1.set_xticks([1,2,3,4,5,6,7])
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title('All Meshes:',fontsize=14)

# magnifying view

ax2.plot(size[0:4]+1,results_gaia_cpu[8,199,0:4]/60,'--',color='g',label="CPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_gaia_gpu[8,199,0:4]/60,'-',color='g',label="GPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_tegner_cpu[8,199,0:4]/60,'--',color='k',label="CPU Tegner",linewidth=1.5)
ax2.legend(loc=2,fontsize=14)
ax2.set_xticks([1,2,3,4])
ax2.set_xlabel('Mesh',fontsize=14)
ax2.set_ylabel('Elapsed Time [min]',fontsize=14 )
#ax1.set_xticks([1,2,3,4,5,6,7])
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title('Magnifying View:',fontsize=14)
pylab.tight_layout()
pylab.savefig(fig_name_2, dpi=100,bbox_inches='tight');

#%% calculate speed up

speed_gaia = results_gaia_cpu[8,199,:]/results_gaia_gpu[8,199,:]
# change to tegner_gpu 
speed_tegner = results_tegner_cpu[8,199,:]/results_gaia_gpu[8,199,:]

