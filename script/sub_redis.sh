#!/bin/bash
#SBATCH --partition chu       
#SBATCH --nodes=1                     
#SBATCH --ntasks=1    
#SBATCH --nodelist=c5           # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=12G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --output=./log/redisOutput.log  
#SBATCH --error=./log/redisINFO.log  
#SBATCH --job-name=Redis

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
# Remove stack size limit to avoid overflow in parallel runs
ulimit -s unlimited
conda activate hamgnn
# Match OMP threads to CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Starting Redis Server..."
redis-server
echo "clear all redis data"
redis-cli FLUSHDB