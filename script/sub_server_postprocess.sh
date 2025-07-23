#!/bin/bash
#SBATCH --partition chu       
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task=12      # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=12G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu   
#SBATCH --output=./log/PostProcessServerOutput.log  
#SBATCH --error=./log/PostProcessServerINFO.log  
#SBATCH --job-name=Server_PostProcess

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
module load compiler/oneAPI/2023.2.0
export LD_LIBRARY_PATH=/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/gsl/lib:$LD_LIBRARY_PATH
# Remove stack size limit to avoid overflow in parallel runs
ulimit -s unlimited
# Match OMP threads to CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
conda activate hamgnn
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

# gunicorn --workers 16 --bind 0.0.0.0:41151 core.openmx-flow.postprocessServer:app --timeout 3000
python -m core.openmx-flow.postprocessServer