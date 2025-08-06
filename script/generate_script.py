import os
from pathlib import Path
server_node = "c6"
server_partition = "xu"
mem_hamgnn=12
cpu_hamgnn=64
N_hamgnn=2
mem_openmx=4
cpu_openmx=12
partition_openmx="xu"
mem_postprocess=4
cpu_postprocess=12
partition_postprocess="xu"
cpu_orchestrator=4
mem_orchestrator=12
cpu_tasks=48
mem_tasks=48
mem_redis=12
cpu_redis=12
script_path = Path(__file__).parent
with open(script_path / "sub_server_hamgnn.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition 8-4090       # 8-4090 or  4v100
#SBATCH --nodes=1                     
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task={cpu_hamgnn}       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_hamgnn}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
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
""")
    
with open(script_path / "sub_server_openmx.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {partition_openmx}       # 8-4090 or  4v100
#SBATCH --nodes=1                     
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task={cpu_openmx}       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_openmx}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --output=./log/OpenmxServerOutput.log  
#SBATCH --error=./log/OpenmxServerINFO.log  
#SBATCH --job-name=Server_Openmx

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
module load compiler/oneAPI/2023.2.0
export LD_LIBRARY_PATH=/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/gsl/lib:$LD_LIBRARY_PATH
# Remove stack size limit to avoid overflow in parallel runs
ulimit -s unlimited
conda activate hamgnn
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
            """)
    
with open(script_path / "sub_server_postprocess.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {partition_postprocess}              
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 
#SBATCH --cpus-per-task={cpu_postprocess}      # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_postprocess}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu
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
""")

with open(script_path / "sub_server_orchestrator.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {server_partition}              
#SBATCH --nodes=1                     
#SBATCH --ntasks=1    
#SBATCH --nodelist={server_node}           # 
#SBATCH --cpus-per-task={cpu_orchestrator}       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_orchestrator}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu
#SBATCH --output=./log/OrchestratorOutput.log  
#SBATCH --error=./log/OrchestratorINFO.log  
#SBATCH --job-name=Server_Orchestrator

ulimit -s unlimited
module purge
source /ssd/app/anaconda3/etc/profile.d/conda.sh
# Remove stack size limit to avoid overflow in parallel runs
ulimit -s unlimited
conda activate hamgnn
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
echo "Starting Orchestrator Server..."
python -m core.orchestrator_server
            """)
    
with open(script_path / "sub_server_tasks.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {server_partition}             
#SBATCH --nodes=1                     
#SBATCH --ntasks=1
#SBATCH --nodelist={server_node}               # 
#SBATCH --cpus-per-task={cpu_tasks}       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_tasks}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
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
celery -A core.tasks.celery_app worker purge
celery -A core.tasks.celery_app worker -B --concurrency=48 --loglevel=INFO
""")

with open(script_path / "sub_redis.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {server_partition}              
#SBATCH --nodes=1                     
#SBATCH --ntasks=1    
#SBATCH --nodelist={server_node}           # 
#SBATCH --cpus-per-task={cpu_redis}       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem={mem_redis}G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
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
echo "Redis Server started successfully."
""")
    
with open(script_path / "close.sh", "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --partition {server_partition}     
#SBATCH --nodes=1                     
#SBATCH --ntasks=1
#SBATCH --nodelist={server_node}               # 
#SBATCH --cpus-per-task=2       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=2G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
#SBATCH --output=./log/close.log  
#SBATCH --error=./log/close.log  
#SBATCH --job-name=Close_tasks

module purge
module load conda/2024.10.1
source /ssd/app/anaconda3/etc/profile.d/conda.sh
conda activate hamgnn
python -m core.close
# 启动 Redis 服务
redis-server &

# 等待 Redis 启动
sleep 2

# 清空数据
redis-cli FLUSHALL

sleep 2

echo "Redis 数据已清空"

# 打印所有数据（安全遍历）
echo "=== Redis 数据内容 ==="
redis-cli --scan --pattern '*' | while read key; do
  type=$(redis-cli type "$key")
  echo -n "Key: $key | Type: $type | Value: "
  case $type in
    "string") redis-cli get "$key" ;;
    "hash") redis-cli hgetall "$key" ;;
    "list") redis-cli lrange "$key" 0 -1 ;;
    "set") redis-cli smembers "$key" ;;
    "zset") redis-cli zrange "$key" 0 -1 withscores ;;
    *) echo "Unsupported type" ;;
  esac
done
echo "======================"

# 关闭 Redis
redis-cli SHUTDOWN

pkill celery


"""
)
    
with open(script_path / "start.sh", "w") as f:
    f.write(f"""sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_redis.sh
sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_server_orchestrator.sh
sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_server_tasks.sh
sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_server_openmx.sh
sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_server_postprocess.sh
""")
    for i in range(N_hamgnn):
        f.write(f"sbatch /ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/script/sub_server_hamgnn.sh\n")

print("Slurm job scripts generated successfully.")