#!/bin/bash
#SBATCH --partition 8-4090       # 8-4090 or  4v100
#SBATCH --nodes=1                     
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=64G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --gpus=1                 # 
#SBATCH --output=./log/HamGNNServer/Output%j.log  
#SBATCH --error=./log/HamGNNServer/Info%j.log  
#SBATCH --job-name=Server_HamGNN

ulimit -s unlimited
module purge
module load conda/2024.10.1
source /ssd/app/anaconda3/etc/profile.d/conda.sh
conda activate hamgnn

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python --version

python -m core.HamGNN.hamgnnServer --config /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/universal/config_predict.yaml
