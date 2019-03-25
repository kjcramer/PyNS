#!/bin/bash -l
## Request one core and 2GB of the memory available on an `iris` cluster node for five days (e.g. for sequential code requesting a lot of memory)
## Valentin Plugaru <Valentin.Plugaru@uni.lu>
#SBATCH -J pyns_salt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=kerstin.cramer@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB
#SBATCH --time=5-00:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#SBATCH -C skylake

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
# Your more useful application can be started below!

module load lang/Python

# run pyns membrane configs on iris
# input required:
# ${1} is the inlet temperature
# ${2} is the inlet velocity
# ${3} is the airgap in {05, 2, 8}
# ${4} is the restart from file option {True, False}
# ${5} is the number of timesteps/1000

python normal_gaia_salt.py ${1} ${2} ${3} ${4} ${5}
