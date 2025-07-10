#!/bin/bash
#SBATCH --partition chu      
#SBATCH --nodes=1                     
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=16G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --output=./log/TaskServerOutput.log  
#SBATCH --error=./log/TaskServerINFO.log  
#SBATCH --job-name=Server_tasks

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
# Remove stack size limit to avoid overflow in parallel runs
conda activate hamgnn
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

celery -A core.tasks.celery_app worker -B --loglevel=INFO