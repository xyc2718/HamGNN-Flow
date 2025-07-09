#!/bin/bash
#SBATCH --partition 8-4090       # 8-4090 or  4v100
#SBATCH --nodes=1                     
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=32G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --gpus=1                 # 
#SBATCH --output=./log/OpenmxServerOutput.log  
#SBATCH --error=./log/OpenmxServerError.log  
#SBATCH --job-name=Server_Openmx

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
module load compiler/oneAPI/2023.2.0
export LD_LIBRARY_PATH=/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/gsl/lib:$LD_LIBRARY_PATH
# Remove stack size limit to avoid overflow in parallel runs
ulimit -s unlimited
# Match OMP threads to CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job and environment info for logging/debugging
cat << EOF
====================== Job Information ======================
Job ID:           $SLURM_JOB_ID
Job Name:         $SLURM_JOB_NAME
Partition:        $SLURM_JOB_PARTITION
Total Nodes:      $SLURM_JOB_NUM_NODES
Total MPI Tasks:  $SLURM_NTASKS
CPUs per Task:    $SLURM_CPUS_PER_TASK
Node List:        $SLURM_JOB_NODELIST
OMP Threads:      $OMP_NUM_THREADS
Job Start Time:   $(date +"%Y-%m-%d %H:%M:%S")
============================================================

EOF
python --version

python -m core.openmx-flow.openmxServer