#!/bin/bash
#SBATCH --partition chu      
#SBATCH --nodes=1                     
#SBATCH --ntasks=1
#SBATCH --nodelist=c2               # 
#SBATCH --cpus-per-task=12       # 8-4090 <= 12 per gpu   4v100  <=4  per gpu
#SBATCH --mem=12G               # 8-4090 <= 100G per gpu 4v100  <=50G per gpu 
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


