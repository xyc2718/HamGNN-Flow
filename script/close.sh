#!/bin/bash
#SBATCH --partition chu      
#SBATCH --nodes=1                     
#SBATCH --ntasks=1
#SBATCH --nodelist=c2               # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=12G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --output=./log/close.log  
#SBATCH --error=./log/close.log  
#SBATCH --job-name=Server_tasks
redis-server
redis-cli FLUSHALL
pkill redis-server
pkill celery
redis-cli SHUTDOWN