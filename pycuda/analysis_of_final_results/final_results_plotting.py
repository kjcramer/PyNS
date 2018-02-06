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

#%% load tegner gpu data

dataset_id_tegner_gpu = "tegner_gpu"

data = np.load(('%s.npz'%(dataset_id_tegner_gpu)))
mesh_sizes = data['mesh_sizes']
results_tegner_gpu = data['results_avgd']

# summing over all iteration times and number of iterations
results_sum_tegner_gpu = np.sum(results_tegner_gpu, axis=1)

#%% plot final comparison plots average time per iteration

# cosmetics
main_text_size = 12
title_size = 14
legend_size = 10

size=np.arange(len(mesh_sizes))
fig_name_1='Comparison_time_per_iteration.png'

# plotting average time per iteration 
plt.ion()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(size+1,results_sum_gaia_cpu[6,:]/results_sum_gaia_cpu[7,:],'--',color='g',label="CPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_sum_gaia_gpu[6,:]/results_sum_gaia_gpu[7,:],'-',color='g',label="GPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_sum_tegner_cpu[6,:]/results_sum_tegner_cpu[7,:],'--',color='k',label="CPU Tegner",linewidth=1.5)
ax1.plot(size+1,results_sum_tegner_gpu[6,:]/results_sum_tegner_gpu[7,:],'-',color='k',label="GPU Tegner",linewidth=1.5)
ax1.legend(loc=2,fontsize=legend_size)
ax1.set_xlabel('Mesh',fontsize=main_text_size)
ax1.set_ylabel('Average Time per Iteration [s]' , fontsize=main_text_size)
ax1.set_title('All Meshes',fontsize=title_size)
ax1.tick_params(axis='both', which='major', labelsize=main_text_size)


# magnifying view

ax2.plot(size[0:4]+1,results_sum_gaia_cpu[6,0:4]/results_sum_gaia_cpu[7,0:4],'--',color='g',label="CPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_sum_gaia_gpu[6,0:4]/results_sum_gaia_gpu[7,0:4],'-',color='g',label="GPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_sum_tegner_cpu[6,0:4]/results_sum_tegner_cpu[7,0:4],'--',color='k',label="CPU Tegner",linewidth=1.5)
ax2.plot(size[0:4]+1,results_sum_tegner_gpu[6,0:4]/results_sum_tegner_gpu[7,0:4],'-',color='k',label="GPU Tegner",linewidth=1.5)
ax2.legend(loc=2,fontsize=legend_size)
ax2.set_xlabel('Mesh',fontsize=main_text_size)
ax2.set_xticks([1,2,3,4])
ax2.set_ylabel('Average Time per Iteration [s]' , fontsize=main_text_size)
ax2.set_title('Magnified View',fontsize=title_size)
ax2.tick_params(axis='both', which='major', labelsize=main_text_size)
pylab.tight_layout()
pylab.savefig(fig_name_1, dpi=100,bbox_inches='tight');

#%% plot final comparison plots total time

fig_name_2='Comparison_total_time.png' 

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(size+1,results_gaia_cpu[8,199,:]/3600,'--',color='g',label="CPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_gaia_gpu[8,199,:]/3600,'-',color='g',label="GPU Gaia",linewidth=1.5)
ax1.plot(size+1,results_tegner_cpu[8,199,:]/3600,'--',color='k',label="CPU Tegner",linewidth=1.5)
ax1.plot(size+1,results_tegner_gpu[8,199,:]/3600,'-',color='k',label="GPU Tegner",linewidth=1.5)
ax1.legend(loc=2,fontsize=legend_size)
ax1.set_xlabel('Mesh',fontsize=main_text_size)
ax1.set_ylabel('Elapsed Time [h]',fontsize=main_text_size )
ax1.tick_params(axis='both', which='major', labelsize=main_text_size)
ax1.set_title('All Meshes',fontsize=title_size)

# magnifying view

ax2.plot(size[0:4]+1,results_gaia_cpu[8,199,0:4]/60,'--',color='g',label="CPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_gaia_gpu[8,199,0:4]/60,'-',color='g',label="GPU Gaia",linewidth=1.5)
ax2.plot(size[0:4]+1,results_tegner_cpu[8,199,0:4]/60,'--',color='k',label="CPU Tegner",linewidth=1.5)
ax2.plot(size[0:4]+1,results_tegner_gpu[8,199,0:4]/60,'-',color='k',label="GPU Tegner",linewidth=1.5)
ax2.legend(loc=2,fontsize=legend_size)
ax2.set_xticks([1,2,3,4])
ax2.set_xlabel('Mesh',fontsize=main_text_size)
ax2.set_ylabel('Elapsed Time [min]',fontsize=main_text_size )
ax2.tick_params(axis='both', which='major', labelsize=main_text_size)
ax2.set_title('Magnified View',fontsize=title_size)
pylab.tight_layout()
pylab.savefig(fig_name_2, dpi=100,bbox_inches='tight');

#%% calculate speed up

speed_gaia = results_gaia_cpu[8,199,:]/results_gaia_gpu[8,199,:]
speed_tegner = results_tegner_cpu[8,199,:]/results_tegner_gpu[8,199,:]

