# core/tasks.py
import os
import yaml
import json
import logging
import subprocess
import time
from pathlib import Path
from celery import Celery
import requests
import redis
import traceback
from .utils import get_package_path, get_server_url
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import heapq
import random
import asyncio
import httpx # 引入 httpx
from redis import asyncio as aioredis # 引入异步redis

# --- 初始化与配置 ---
# 初始化日志,建议在Celery worker启动时也配置好日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TASK_CONFIG_PATH = get_package_path('task_basic_config.yaml')
config = yaml.safe_load(open(TASK_CONFIG_PATH, 'r', encoding='utf-8'))
logger.info(f"加载任务配置: {TASK_CONFIG_PATH}")

# 初始化一个全局的Redis客户端,用于我们自定义的状态缓存操作
# decode_responses=True 确保我们从Redis获取的值是字符串而不是字节
try:
    # 假设Redis在本地运行,如果不是,请修改host
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Celery任务模块成功连接到Redis。")
except redis.exceptions.ConnectionError as e:
    logger.error(f"无法连接到Redis,请确保Redis服务正在运行: {e}")
    # 在无法连接到Redis时,抛出异常,阻止worker启动
    raise



# 初始化Celery应用
# 'tasks'是当前模块的名称
# broker是"任务订单栏"的地址,backend是"任务状态显示屏"的地址
celery_app = Celery('tasks',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0')
# celery_app.conf.worker_prefetch_multiplier = 1  # 降低预取数量
# celery_app.conf.task_acks_late = True  # 任务完成后才确认
# celery_app.conf.worker_max_tasks_per_child = 10  # 处理10个任务后重启worker进程

@celery_app.task
def worker_healthcheck():
    """检查worker健康状态并记录"""
    return {"status": "healthy", "memory_usage": psutil.virtual_memory().percent}

# --- 队列定义 ---
# 使用语义化命名的五个队列
QUEUE_PENDING = 'pending_tasks'               # 队列A: 待处理队列
QUEUE_OPENMX_WAITING = 'openmx_waiting_tasks' # 队列B: openmx等待队列
QUEUE_HAMGNN_WAITING = 'hamgnn_waiting_tasks' # 队列C: hamgnn等待队列
QUEUE_POST_WAITING = 'postprocess_waiting_tasks' # 队列D: 后处理等待队列
QUEUE_COMPLETED = 'completed_tasks'           # 队列E: 完成队列

TIMEOUT=1200

# --- 辅助函数 ---
def submit_request(process_url, job_ticket):
    """将请求提交到线程池的辅助函数"""
    try:
        response = requests.post(
            process_url, 
            json=job_ticket, 
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()  # 返回响应的JSON内容
    except Exception as e:
        logger.error(f"请求失败: {e}")
        raise

def _move_task(task_id, from_queue, to_queue, task_data=None):
    """
    使用Redis WATCH机制实现真正原子性的队列任务移动。
    在高并发环境下确保任务只被移动一次。
    
    Args:
        task_id: 任务ID
        from_queue: 源队列名称
        to_queue: 目标队列名称
        task_data: 可选的更新任务数据
        
    Returns:
        bool: 是否成功移动任务
    """
    # 获取当前任务数据(如果未提供)
    if task_data is None:
        current_data = redis_client.hget(from_queue, task_id)
        if not current_data:
            logger.debug(f"任务 {task_id} 不在队列 {from_queue} 中，无法移动")
            return False
        try:
            task_data = json.loads(current_data)
        except json.JSONDecodeError:
            logger.error(f"任务 {task_id} 数据格式无效，无法解析JSON")
            return False
    
    # 使用WATCH和事务确保原子操作
    max_retries = 3  # 最多重试3次
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with redis_client.pipeline() as pipe:
                # 监视源队列中的任务，如果发生变化则事务会失败
                pipe.watch(f"{from_queue}")
                
                # 再次检查任务是否存在于源队列(可能已被其他进程移走)
                if not redis_client.hexists(from_queue, task_id):
                    pipe.unwatch()
                    logger.debug(f"任务 {task_id} 已不在队列 {from_queue} 中，可能已被移走")
                    return False
                
                # 更新任务状态日志
                task_data['status_log'] = task_data.get('status_log', [])
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'from_queue': from_queue,
                    'to_queue': to_queue,
                    'worker_id': os.getpid(),  # 记录处理任务的worker ID
                    'move_attempt': retry_count + 1
                })
                
                # 开始事务
                pipe.multi()
                
                # 从源队列移除并添加到目标队列
                pipe.hdel(from_queue, task_id)
                pipe.hset(to_queue, task_id, json.dumps(task_data))
                
                # 执行事务
                results = pipe.execute()
                success = all(results)
                
                if success:
                    logger.debug(f"成功将任务 {task_id} 从 {from_queue} 移动到 {to_queue}")
                    return True
                else:
                    logger.warning(f"移动任务 {task_id} 从 {from_queue} 到 {to_queue} 失败，结果: {results}")
                    return False
                
        except redis.WatchError:
            # 如果监视的键在事务执行前被修改，Redis会抛出WatchError
            logger.debug(f"任务 {task_id} 在移动过程中被其他进程修改，第 {retry_count+1} 次重试")
            retry_count += 1
            time.sleep(0.1 * (2 ** retry_count))  # 指数退避策略
            continue
            
        except Exception as e:
            # 处理其他可能的异常
            logger.error(f"移动任务 {task_id} 时发生错误: {e}")
            return False
    
    # 达到最大重试次数仍失败
    logger.warning(f"移动任务 {task_id} 从 {from_queue} 到 {to_queue} 失败，达到最大重试次数 ({max_retries})")
    return False

async def _async_move_task(redis_cli: aioredis.Redis, task_id: str, from_queue: str, to_queue: str, task_data: dict) -> bool:
    """
    _move_task 的异步版本，接收一个异步redis客户端作为参数。
    """
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with redis_cli.pipeline() as pipe:
                await pipe.watch(from_queue)
                if not await pipe.hexists(from_queue, task_id):
                    await pipe.unwatch()
                    logger.debug(f"任务 {task_id} 已不在队列 {from_queue} 中，异步移动取消")
                    return False
                
                # 更新状态日志
                task_data['status_log'] = task_data.get('status_log', [])
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'from_queue': from_queue,
                    'to_queue': to_queue,
                    'worker_id': f"async-{os.getpid()}",
                    'move_attempt': retry_count + 1
                })
                
                pipe.multi()
                pipe.hdel(from_queue, task_id)
                pipe.hset(to_queue, task_id, json.dumps(task_data))
                
                results = await pipe.execute()
                return all(results)
        except aioredis.WatchError:
            logger.debug(f"异步移动任务 {task_id} 时发生竞争，第 {retry_count + 1} 次重试")
            retry_count += 1
            await asyncio.sleep(0.1 * (2 ** retry_count))
            continue
        except Exception as e:
            logger.error(f"异步移动任务 {task_id} 时发生严重错误: {e}")
            return False
            
    logger.warning(f"异步移动任务 {task_id} 达到最大重试次数，移动失败")
    return False

def _write_failure_file(workdir, stage_name, details):
    """
    在指定工作目录下写入统一格式的FAILURE.json文件。

    Args:
        workdir (str): 任务的工作目录。
        stage_name (str): 任务失败时所处的阶段名称 (例如, '1/4: OpenMX预处理')。
        details (str): 详细的错误信息。
    """
    if not workdir:
        logger.warning("工作目录未指定，无法写入FAILURE.json文件")
        return
    try:
        # 确保工作目录存在，如果不存在则创建
        Path(workdir).mkdir(parents=True, exist_ok=True)
        
        failure_info = {
            'stage_code': 'FAILED',
            'stage_name': stage_name,
            'details': str(details),
            'workdir': str(workdir)
        }
        failure_file_path = os.path.join(workdir, 'FAILURE.json')
        
        with open(failure_file_path, 'w', encoding='utf-8') as f:
            json.dump(failure_info, f, ensure_ascii=False, indent=4)
            
        logger.info(f"已在 {workdir} 写入失败信息: {failure_file_path}")
    except Exception as file_error:
        # 即便写入文件失败，也只记录日志，不影响主流程
        logger.error(f"写入失败信息到 {workdir} 时发生严重错误: {file_error}")



def _get_best_partition(ncpus:int=4):
    """
    【最终修正】从Redis缓存中获取最空闲的Slurm分区，并原子性地扣除所需CPU数量。
    
    Args:
        ncpus (int): 本次任务需要消耗的CPU核心数。
        
    Returns:
        str: 成功预留资源的分区名。如果没有找到合适的分区，则返回默认分区。
    """
    partitions_key = 'slurm_partition_status'

    
    temp_val = config.get("slurm_monitor", {}).get("default_partition", "chu")
    # 检查 temp_val 是否为列表，然后赋给 default_partition
    # 如果是列表，则随机选择一个元素
    # 如果不是列表（即字符串），则直接使用该值
    default_partition = random.choice(temp_val) if isinstance(temp_val, list) else temp_val


    AVAILABLE_STATES = {'idle', 'mixed', 'up', 'alloc', 'aggregated'}

    try:
        all_partitions = redis_client.hgetall(partitions_key)
        if not all_partitions:
            logger.warning(f"Redis中没有Slurm分区信息，将使用默认分区: {default_partition}")
            return default_partition

        # 1. 筛选出所有满足CPU数要求的候选分区
        candidate_partitions = []
        for name, data_json in all_partitions.items():
            try:
                data = json.loads(data_json)
                if data.get('state') in AVAILABLE_STATES and data.get('idle_cpus', 0) >= ncpus:
                    # 排除带'*'的特殊分区
                    if '*' in name or '*' in str(data.get('total_cpus', '')):
                        continue
                    candidate_partitions.append({'name': name, 'idle_cpus': data.get('idle_cpus', 0)})
            except (json.JSONDecodeError, ValueError):
                continue
        
        if not candidate_partitions:
            logger.warning(f"没有找到任何至少有 {ncpus} 个空闲CPU的可用分区，将使用默认分区。")
            return default_partition
            
        # 2. 按空闲CPU数降序排序，优先选择最空闲的
        candidate_partitions.sort(key=lambda p: p['idle_cpus'], reverse=True)

        # 3. 循环尝试，直到成功为一个分区预留资源
        for candidate in candidate_partitions:
            partition_name = candidate['name']
            
            # 使用 WATCH 监视整个哈希键
            with redis_client.pipeline() as pipe:
                try:
                    pipe.watch(partitions_key)
                    
                    # 在事务开始前，再次获取最新的数据
                    latest_data_json = pipe.hget(partitions_key, partition_name)
                    if not latest_data_json:
                        # 分区信息在此期间消失了，跳过
                        pipe.unwatch()
                        continue
                    
                    latest_data = json.loads(latest_data_json)
                    
                    # 再次确认CPU数量是否足够
                    if latest_data.get('idle_cpus', 0) < ncpus:
                        pipe.unwatch()
                        continue
                        
                    # 开始事务
                    pipe.multi()
                    
                    # 计算并设置新的空闲CPU数
                    latest_data['idle_cpus'] -= ncpus
                    updated_data_json = json.dumps(latest_data)
                    pipe.hset(partitions_key, partition_name, updated_data_json)
                    
                    # 执行事务
                    results = pipe.execute()
                    
                    # 如果事务成功（没有被WATCH中断），则说明资源预留成功
                    if results:
                        logger.info(f"成功为任务在分区 {partition_name} 预留 {ncpus} 个CPU，"
                                    f"该分区剩余空闲CPU: {latest_data['idle_cpus']}")
                        return partition_name

                except redis.WatchError:
                    # 如果发生WatchError，说明在WATCH到EXEC之间数据被改变了
                    # 记录日志并继续循环，尝试下一个候选分区
                    logger.debug(f"尝试为分区 {partition_name} 预留资源时发生竞争，将尝试下一个分区。")
                    continue
        
        # 4. 如果所有候选分区都尝试失败（高并发），则返回默认分区
        logger.warning(f"所有候选分区在高并发下均预留失败，将使用默认分区: {default_partition}")
        return default_partition

    except Exception as e:
        logger.error(f"获取最佳分区时发生严重错误: {e}，将使用默认分区: {default_partition}")
        return default_partition




def wait_for_slurm_job(job_id: str, first_time=config.get("slurm_monitor", {}).get("first_time", 10),
                      poll_interval: int = config.get("slurm_monitor", {}).get("poll_interval", 1), 
                      timeout: int = 7200) -> bool:
    """通过查询Redis缓存来等待一个Slurm作业完成。"""
    logger.info(f"开始通过Redis缓存监控作业 {job_id}...")
    start_time = time.time()
    
    redis_client.sadd('monitored_slurm_jobs', job_id)
    
    time.sleep(first_time)  # 初始等待,给Slurm一些时间来更新状态
    while ((time.time() - start_time) < timeout):
        logger.info(f"正在查询作业 {job_id} 的状态...")
        state = redis_client.hget('slurm_job_status_cache', job_id)
        if state:
            state = state.strip()
            if "COMPLETED" in state:
                logger.info(f"从缓存中检测到作业 {job_id} 成功完成。")
                redis_client.srem('monitored_slurm_jobs', job_id)
                redis_client.hdel('slurm_job_status_cache', job_id)
                return True
            elif any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
                logger.error(f"从缓存中检测到作业 {job_id} 失败,状态为: {state}。")
                redis_client.srem('monitored_slurm_jobs', job_id)
                redis_client.hdel('slurm_job_status_cache', job_id)
                return False
        
        time.sleep(poll_interval)
        
    logger.error(f"等待作业 {job_id} 超时 ({timeout}秒)。")
    redis_client.srem('monitored_slurm_jobs', job_id)
    redis_client.hdel('slurm_job_status_cache', job_id)
    return False




@celery_app.task
def update_slurm_partition_status():
    """
    定期查询并聚合Slurm各分区的状态，更新到Redis缓存。
    此版本解决了因sinfo对同一分区输出多行而导致的数据覆盖问题，
    并使用正确的列来计算总CPU和空闲CPU。
    """
    logger.info("开始查询并聚合Slurm分区状态...")
    excluded_partitions = set(config.get("monitoring", {}).get("excluded_partitions", []))
    try:
        # 简化命令，我们只需要 分区名(P), 节点状态(T), 和CPU详情(C)
        command = ["sinfo", "-h", "-o", "%P|%T|%C"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # 1. 在内存中聚合数据
        partitions_aggregated = {}

        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.strip().split('|')
            if len(parts) != 3:
                logger.warning(f"预期的sinfo格式不符(P|T|C)，跳过该行: '{line}'")
                continue

            try:
                partition_name, state, cpus_state_str = parts

                if partition_name in excluded_partitions:
                    continue  # 如果在列表中，则跳过此行
                
                # 忽略带'*'的默认分区汇总行
                if '*' in partition_name:
                    continue

                cpu_stats = cpus_state_str.split('/')
                if len(cpu_stats) != 4:
                    logger.warning(f"CPU状态格式不为A/I/O/T，跳过: '{cpus_state_str}'")
                    continue

                # 从 A/I/O/T 中提取空闲(I)和总数(T)
                idle_cpus_line = int(cpu_stats[1])
                total_cpus_line = int(cpu_stats[3])

                # 如果分区是第一次出现，则初始化
                if partition_name not in partitions_aggregated:
                    partitions_aggregated[partition_name] = {'idle_cpus': 0, 'total_cpus': 0}
                
                # 累加空闲和总CPU数
                partitions_aggregated[partition_name]['idle_cpus'] += idle_cpus_line
                partitions_aggregated[partition_name]['total_cpus'] += total_cpus_line

            except (ValueError, IndexError) as e:
                logger.warning(f"解析sinfo聚合行失败: '{line}', 错误: {e}")
                continue
        
        # 2. 将聚合后的最终结果写入Redis
        pipe = redis_client.pipeline()
        partitions_key = 'slurm_partition_status'
        pipe.delete(partitions_key)
        
        updated_partitions = []
        for name, data in partitions_aggregated.items():
            # 为聚合后的数据准备一个统一的state
            # 注意：这里的 'state' 字段仅用于存储，实际选择逻辑已不依赖它
            partition_data = {
                'state': 'aggregated',
                'total_cpus': data['total_cpus'],
                'idle_cpus': data['idle_cpus'],
                'updated_at': time.time()
            }
            pipe.hset(partitions_key, name, json.dumps(partition_data))
            updated_partitions.append(name)
        
        pipe.execute()

        if updated_partitions:
            # logger.info(f"成功聚合并更新了 {len(updated_partitions)} 个Slurm分区的状态: {updated_partitions}")
            logger.info(f"当前Slurm分区状态: {partitions_aggregated}")
        else:
            logger.warning("没有成功更新任何Slurm分区状态，请检查sinfo输出和日志。")

        return f"已聚合更新 {len(updated_partitions)} 个分区的状态。"

    except subprocess.CalledProcessError as e:
        logger.error(f"执行sinfo命令失败: {e.stderr}")
        return f"sinfo命令执行失败: {e}"
    except Exception as e:
        logger.error(f"更新Slurm分区状态时发生未知错误: {e}")
        return f"更新分区状态失败: {e}"




@celery_app.task
def poll_slurm_jobs():
    """
    批量查询所有被监控的Slurm作业的状态,并更新Redis缓存。
    这个函数由Celery Beat定期调用。
    """
    monitored_jobs = redis_client.smembers('monitored_slurm_jobs')
    if not monitored_jobs:
        return "当前没有需要监控的作业。"
    
    job_ids_str = ",".join(monitored_jobs)
    logger.info(f"批量查询 {monitored_jobs} 个作业的状态...")
    
    try:
        command = ["sacct", "-j", job_ids_str, "-o", "JobID,State", "-n", "-P"]
        
        # 【关键修正】确保这里是完整的函数调用
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        pipe = redis_client.pipeline()
        updated_jobs = set()
        
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            try:
                job_id, state = line.strip().split('|')[:2]
                state = state.strip()
                pipe.hset('slurm_job_status_cache', job_id, state)
                updated_jobs.add(job_id.split('.')[0])
            except ValueError:
                continue
        
        pipe.execute()
        # 检查哪些作业已经终结并从相关监控列表中移除
        TERMINAL_STATES = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]
        finished_jobs = set()
        for job_id in updated_jobs:
            state = redis_client.hget('slurm_job_status_cache', job_id)
            if state and any(term in state for term in TERMINAL_STATES):
                finished_jobs.add(job_id)
        if finished_jobs:
            redis_client.srem('monitored_slurm_jobs', *finished_jobs)
            # 注意：这里我们不再从running_preprocess_jobs中移除，
            # 因为我们需要在poll_slurm_and_dispatch中区分不同类型的作业
            logger.info(f"作业 {finished_jobs} 已完成,从监控列表移除。")
        return f"已更新 {updated_jobs} 个作业的状态。"
    except subprocess.CalledProcessError as e:
        # 如果sacct命令因为某些原因(比如所有作业都不在近期历史记录中)而返回非零退出码
        logger.warning(f"sacct命令可能执行失败(这在作业完成后是正常的): {e.stderr}")
        # 在这种情况下,我们可以认为所有被查询的作业都已经终结
        redis_client.srem('monitored_slurm_jobs', *monitored_jobs)
        return "sacct查询无有效返回,已清理所有监控中的作业。"
    except Exception as e:
        logger.error(f"批量查询Slurm作业时发生未知错误: {e}")
        return f"批量查询失败: {e}"

# --- 核心工作流任务 ---

# 在celery_app初始化后添加
def initialize_redis_keys():
    """确保所有计数器和集合都是正确的类型"""
    # 清理并重新初始化后处理计数器
    redis_client.delete('running_postprocess_jobs')
    # 清理并重新初始化OpenMX计数器
    if redis_client.type('running_preprocess_jobs') != 'set':
        redis_client.delete('running_preprocess_jobs')
    # 清理并重新初始化HamGNN计数器
    if redis_client.type('running_hamgnn_jobs') != 'string':
        redis_client.delete('running_hamgnn_jobs')
        redis_client.set('running_hamgnn_jobs', '0')


@celery_app.task(
    bind=True,
    autoretry_for=(requests.exceptions.RequestException, ConnectionError),
    retry_kwargs={'max_retries': config.get("celery_task", {}).get("max_retries", 1)},
    default_retry_delay=config.get("celery_task", {}).get("default_retry_delay", 60)
)
def start_workflow(self, structure_file_path: str, workflow_params: dict = {}):
    """
    【入口函数】- 仅负责将任务放入待处理队列。
    
    Args:
        structure_file_path: 结构文件路径
        workflow_params: 工作流参数
        
    Returns:
        dict: 包含任务ID和状态URL的信息
    """
    try:
        #分区调度逻辑修改为在每个任务内
        # ncpus = workflow_params.get('ncpus', 4)
        # if workflow_params.get('partition') == 'auto':
        #     logger.info("检测到 'partition' 参数为 'auto'，开始自动选择分区...")
        #     best_partition = _get_best_partition(ncpus=ncpus)
        #     workflow_params['partition'] = best_partition
        #     logger.info(f"已自动选择分区: {best_partition}")

        
        # 创建任务唯一标识符
        task_id = self.request.id
        
        # 准备工作目录
        workdir = workflow_params.get('output_path',None)
        
        # 准备任务数据
        task_data = {
            'task_id': task_id,
            'structure_file_path': structure_file_path,
            'workflow_params': workflow_params,
            'workdir': workdir,
            'status': 'pending',
            'created_at': time.time(),
            'status_log': [
                {
                    'timestamp': time.time(),
                    'status': 'created',
                    'message': '任务已创建并加入待处理队列'
                }
            ]
        }
        
        # 将任务添加到待处理队列
        redis_client.hset(QUEUE_PENDING, task_id, json.dumps(task_data))
        
        logger.info(f"任务 {task_id} 已加入待处理队列,结构文件: {structure_file_path}")
        
        # 更新任务状态
        self.update_state(
            state='PROGRESS', 
            meta={
                'stage_code': 'QUEUED',
                'stage_name': '0/4: 待处理队列', 
                'details': '任务已加入处理队列,等待调度',
                'workdir': workdir
            }
        )
        
        return {
            'task_id': task_id,
            'status': 'queued',
            'message': '任务已加入处理队列'
        }
    except Exception as e:
        logger.error(f"创建任务时发生错误: {e}")
        if 'workdir' in locals() and workdir:
             _write_failure_file(
                 workdir=workdir,
                 stage_name='0/4: 初始化',
                 details=f'创建任务失败: {str(e)}'
             )
        self.update_state(
            state='FAILURE',
            meta={
                'stage_code': 'FAILED',
                'stage_name': '0/4: 初始化', 
                'details': f'创建任务失败: {str(e)}'
            }
        )
        raise

# --- 四个独立的定时任务函数 --
Max_Openmx_Batch_Size = config.get('concurrency', {}).get('max_openmx_batch_size', 16)
@celery_app.task
def dispatch_openmx_tasks():
    """
    F1定时任务: 检查openmx并发限制,将待处理队列的任务转移到openmx等待队列并提交给openmxServer
    【已修正】: 增加了分布式锁来防止多个worker处理同一个任务。
    """
    # 从配置中获取最大并发数
    max_concurrent = config.get('concurrency', {}).get('max_openmx_jobs', 10)
    
    # 检查当前运行的预处理作业数
    current_running = redis_client.scard('running_preprocess_jobs')
    
    # 计算可提交的任务数量
    slots_available = max(0, max_concurrent - current_running)
    submission_limit = min(slots_available, Max_Hamgnn_Batch_Size)
    
    if slots_available <= 0:
        logger.info(f"OpenMX并发数已达上限({max_concurrent}),当前运行: {current_running}")
        return f"OpenMX并发数已达上限({max_concurrent}),跳过调度"
    
    # 获取待处理队列中的所有任务
    pending_tasks = redis_client.hgetall(QUEUE_PENDING)
    
    if not pending_tasks:
        return "没有待处理的任务"
    
    # 按创建时间排序(先进先出)
    sorted_tasks = []
    for task_id, task_data_json in pending_tasks.items():
        try:
            task_data = json.loads(task_data_json)
            sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
        except json.JSONDecodeError:
            logger.error(f"任务 {task_id} 的数据格式无效，跳过")
            continue
    
    sorted_tasks = heapq.nsmallest(submission_limit, sorted_tasks, key=lambda x: x[2])
    
    # 处理可提交的任务
    tasks_processed = 0
    for task_id, task_data, _ in sorted_tasks:
        if tasks_processed >= submission_limit:
            break

        # 为每个任务创建一个锁，确保同一任务不会被并发处理
        task_lock_key = f"openmx_processing_lock:{task_id}"
        
        # 尝试获取锁，如果已被锁定则跳过 (nx=True: set if not exist)
        if not redis_client.set(task_lock_key, "1", nx=True, ex=600):  # 10分钟锁
            logger.info(f"OpenMX任务 {task_id} 正在被其他进程处理，跳过")
            continue

        try:
            # 再次检查任务是否在队列中(可能在获取锁的过程中被其他进程移除)
            if not redis_client.hexists(QUEUE_PENDING, task_id):
                logger.info(f"任务 {task_id} 不在待处理队列中，可能已被处理")
                continue

            # 提取任务参数
            structure_file_path = task_data.get('structure_file_path')
            workflow_params = task_data.get('workflow_params', {})
            # logger.info(f"work_para:{workflow_params}")
            ncpus = workflow_params.get('ncpus', 4)
            workflow_params_openmx = workflow_params.copy()
            if workflow_params.get('partition') == 'auto':
                logger.info("检测到 'partition' 参数为 'auto'，开始自动选择分区...")
                best_partition = _get_best_partition(ncpus=ncpus)
                workflow_params_openmx['partition'] = best_partition
                logger.info(f"已自动选择分区: {best_partition}")
                # logger.info(f"work_para:{workflow_params}")
            workdir = task_data.get('workdir')
            
            # 更新任务状态
            task_data['status'] = 'submitting_to_openmx'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'submitting_to_openmx',
                'message': '正在提交给OpenMX预处理服务'
            })
            
            # 确定是否是SCF计算
            ifscf = workflow_params.get('ifscf', False)
            output_path = workflow_params.get('output_path', None)
            
            # 确定预处理URL
            if ifscf:
                preprocess_url = get_server_url("openmx") + "/scf"
            else:
                preprocess_url = get_server_url("openmx") + "/pre_process"
                
            # 提交给OpenMX服务器
            logger.info(f"提交任务 {task_id} 到OpenMX服务器: {preprocess_url}")
            response = requests.post(
                preprocess_url, 
                json={
                    "structure": str(structure_file_path), 
                    "graph_para": workflow_params_openmx,
                    "output_path": output_path,
                    "timeout": 120
                }
            )
            
            # 解析响应
            response_data = response.json()
            workdir = response_data['workdir']
            task_data['workdir'] = workdir
            response.raise_for_status() # 如果请求失败，将在这里抛出异常
            preprocess_job_id = response_data['job_id']
            # 更新任务数据
            task_data['openmx_job_id'] = preprocess_job_id
            task_data['status'] = 'submitted_to_openmx'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'submitted_to_openmx',
                'message': f'已提交给OpenMX,作业ID: {preprocess_job_id}'
            })
            
            # 将Slurm作业ID添加到监控列表
            redis_client.sadd('running_preprocess_jobs', preprocess_job_id)
            redis_client.sadd('monitored_slurm_jobs', preprocess_job_id)
            
            # 将任务从待处理队列移动到openmx等待队列
            if _move_task(task_id, QUEUE_PENDING, QUEUE_OPENMX_WAITING, task_data):
                logger.info(f"任务 {task_id} 已提交给OpenMX服务器并移至等待队列,Slurm作业ID: {preprocess_job_id}")
                tasks_processed += 1
            else:
                logger.error(f"移动任务 {task_id} 到OpenMX等待队列失败")
                # 如果移动失败，可能需要考虑回滚操作，例如从running_preprocess_jobs中移除ID
                redis_client.srem('running_preprocess_jobs', preprocess_job_id)
                redis_client.srem('monitored_slurm_jobs', preprocess_job_id)

        except Exception as e:
            logger.error(f"提交任务 {task_id} 到OpenMX服务器时出错: {e}")
            # 更新任务状态为失败
            task_data['status'] = 'failed'
            task_data['error'] = str(e)
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'failed',
                'message': f'提交到OpenMX服务器失败: {str(e)}'
            })
            _write_failure_file(
                workdir=task_data.get('workdir'),
                stage_name='1/4: OpenMX预处理',
                details=f'提交到OpenMX服务器失败: {str(e)}'
            )
            # 将失败的任务直接移到完成队列
            _move_task(task_id, QUEUE_PENDING, QUEUE_COMPLETED, task_data)
        
        finally:
            #无论成功或失败，最后都释放锁
            redis_client.delete(task_lock_key)
            
    return f"处理了 {tasks_processed} 个任务"

@celery_app.task
def poll_slurm_and_dispatch():
    """
    F2定时任务: 轮询Slurm状态,检查所有等待队列中的Slurm作业状态,并移动已完成的任务到下一队列
    现在处理两种类型的作业:
    1. OpenMX作业 (openmx_job_id): 完成后从openmx等待队列移动到hamgnn等待队列
    2. Postprocess作业 (postprocess_job_id): 完成后从后处理等待队列移动到完成队列
    """
    # 先执行轮询Slurm作业的任务,更新所有作业状态
    poll_result = poll_slurm_jobs()
    # logger.info(f"轮询Slurm作业状态结果: {poll_result}")
    
    # 处理openmx等待队列中的任务
    openmx_tasks_processed = process_openmx_waiting_tasks()
    
    # 处理postprocess等待队列中的任务
    postprocess_tasks_processed = process_postprocess_waiting_tasks()
    
    return f"已处理 {openmx_tasks_processed} 个OpenMX任务和 {postprocess_tasks_processed} 个后处理任务"

def process_openmx_waiting_tasks():
    """处理openmx等待队列中的任务"""
    # 获取openmx等待队列中的所有任务
    openmx_tasks = redis_client.hgetall(QUEUE_OPENMX_WAITING)
    
    if not openmx_tasks:
        return 0
    
    tasks_processed = 0
    
    for task_id, task_data_json in openmx_tasks.items():
        task_data = json.loads(task_data_json)
        openmx_job_id = task_data.get('openmx_job_id')
        
        if not openmx_job_id:
            logger.error(f"任务 {task_id} 中没有找到OpenMX作业ID")
            continue
            
        # 检查Slurm作业状态
        state = redis_client.hget('slurm_job_status_cache', openmx_job_id)
        
        if state and "COMPLETED" in state:
            # 作业已完成,更新任务状态
            task_data['status'] = 'openmx_completed'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'openmx_completed',
                'message': 'OpenMX预处理已完成'
            })
            
            # 从监控列表中移除作业ID
            redis_client.srem('monitored_slurm_jobs', openmx_job_id)
            redis_client.srem('running_preprocess_jobs', openmx_job_id)
            
            # 将任务从openmx等待队列移动到hamgnn等待队列
            if _move_task(task_id, QUEUE_OPENMX_WAITING, QUEUE_HAMGNN_WAITING, task_data):
                logger.info(f"任务 {task_id} 的OpenMX预处理已完成,已移至HamGNN等待队列")
                tasks_processed += 1
            else:
                logger.error(f"移动任务 {task_id} 到HamGNN等待队列失败")
                
        elif state and any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
            # 作业失败,更新任务状态
            task_data['status'] = 'openmx_failed'
            task_data['error'] = f"OpenMX预处理作业失败,状态: {state}"
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'openmx_failed',
                'message': f'OpenMX预处理作业失败,状态: {state}'
            })
            _write_failure_file(
                workdir=task_data.get('workdir'),
                stage_name='1/4: OpenMX预处理',
                details=f'OpenMX预处理作业失败,状态: {state}'
            )
            # 从监控列表中移除作业ID
            redis_client.srem('monitored_slurm_jobs', openmx_job_id)
            redis_client.srem('running_preprocess_jobs', openmx_job_id)
            
            # 将失败的任务直接移到完成队列
            _move_task(task_id, QUEUE_OPENMX_WAITING, QUEUE_COMPLETED, task_data)
            
    return tasks_processed

def process_postprocess_waiting_tasks():
    """处理后处理等待队列中的任务"""
    # 获取后处理等待队列中的所有任务
    postprocess_tasks = redis_client.hgetall(QUEUE_POST_WAITING)
    
    if not postprocess_tasks:
        return 0
    
    tasks_processed = 0
    
    for task_id, task_data_json in postprocess_tasks.items():
        task_data = json.loads(task_data_json)
        postprocess_job_id = task_data.get('postprocess_job_id')
        
        # 跳过没有postprocess_job_id的任务(可能是尚未提交的任务)
        if not postprocess_job_id:
            continue
            
        # 检查Slurm作业状态
        state = redis_client.hget('slurm_job_status_cache', postprocess_job_id)
        
        if state and "COMPLETED" in state:
            # 作业已完成,更新任务状态
            task_data['status'] = 'postprocess_completed'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'postprocess_completed',
                'message': '后处理已完成'
            })
            
            # 从监控列表中移除作业ID
            redis_client.srem('monitored_slurm_jobs', postprocess_job_id)
            redis_client.srem('running_postprocess_jobs', postprocess_job_id)
            
            # 写入成功标记文件
            try:
                final_result = {
                    'stage_code': 'COMPLETED', 
                    'stage_name': '全部完成', 
                    'details': '工作流所有阶段已成功执行完毕。', 
                    'result_dir': str(task_data.get('workdir'))
                }
                with open(os.path.join(task_data.get('workdir'), 'SUCCESS.json'), 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, ensure_ascii=False, indent=4)
            except Exception as file_error:
                logger.error(f"写入成功信息到 {task_data.get('workdir')} 时出错: {file_error}")
            
            # 标记任务为已完成(用于幂等性检查)
            redis_client.set(f"completed:{task_id}", "1", ex=7*24*60*60)  # 7天过期
            
            # 将任务从后处理等待队列移动到完成队列
            if _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data):
                logger.info(f"任务 {task_id} 的后处理已完成,已移至完成队列")
                tasks_processed += 1
            else:
                logger.error(f"移动任务 {task_id} 到完成队列失败")
                
        elif state and any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
            # 作业失败,更新任务状态
            task_data['status'] = 'postprocess_failed'
            task_data['error'] = f"后处理作业失败,状态: {state}"
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'postprocess_failed',
                'message': f'后处理作业失败,状态: {state}'
            })
            
            # 从监控列表中移除作业ID
            redis_client.srem('monitored_slurm_jobs', postprocess_job_id)
            redis_client.srem('running_postprocess_jobs', postprocess_job_id)
            
            # 写入失败标记文件
            try:
                _write_failure_file(
                    workdir=task_data.get('workdir'),
                    stage_name='后处理',
                    details=f'后处理作业失败,状态: {state}'
                )
            except Exception as file_error:
                logger.error(f"写入失败信息到 {task_data.get('workdir')} 时出错: {file_error}")
            
            # 将失败的任务直接移到完成队列
            _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
            
    return tasks_processed
Max_Hamgnn_Batch_Size = config.get('concurrency', {}).get('max_hamgnn_batch_size', 12)
@celery_app.task
def dispatch_hamgnn_tasks_async():
    """
    F3定时任务的【异步重构版本】: 这是一个同步的Celery任务，作为异步流程的启动器。
    """
    try:
        # 【关键】使用 asyncio.run() 启动并运行整个异步调度流程直到完成
        result_message = asyncio.run(run_hamgnn_dispatcher())
        logger.info(result_message)
        return result_message
    except Exception as e:
        logger.error(f"HamGNN异步调度任务执行时发生顶层错误: {traceback.format_exc()}")
        return f"执行错误: {str(e)}"


async def run_hamgnn_dispatcher():
    """
    实际的异步调度器逻辑。它负责创建客户端并编排任务。
    """
    # 【关键】在此函数内创建异步客户端，确保它们属于同一个事件循环
    redis_cli = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)

    try:
        max_concurrent = config.get('concurrency', {}).get('max_hamgnn_jobs', 5)
        current_running_key = 'running_hamgnn_jobs'
        
        await redis_cli.setnx(current_running_key, 0)
        current_running = int(await redis_cli.get(current_running_key) or 0)
        
        slots_available = max(0, max_concurrent - current_running)
        submission_limit = min(slots_available, Max_Hamgnn_Batch_Size)
        
        if submission_limit <= 0:
            return f"HamGNN并发数已达上限({max_concurrent}),当前运行: {current_running}"
            
        hamgnn_tasks = await redis_cli.hgetall(QUEUE_HAMGNN_WAITING)
        if not hamgnn_tasks:
            return "没有等待中的HamGNN任务"

        sorted_tasks_meta = []
        for task_id, task_data_json in hamgnn_tasks.items():
            try:
                task_data = json.loads(task_data_json)
                sorted_tasks_meta.append((task_id, task_data, task_data.get('created_at', 0)))
            except json.JSONDecodeError:
                continue
        tasks_to_process = heapq.nsmallest(submission_limit, sorted_tasks_meta, key=lambda x: x[2])
        
        if not tasks_to_process:
            return "没有可处理的HamGNN任务"

        # 【关键】使用 async with 创建 http 客户端
        async with httpx.AsyncClient(timeout=TIMEOUT) as http_client:
            coroutines = []
            for task_id, task_data, _ in tasks_to_process:
                # 【关键】将客户端作为参数传递
                coroutines.append(process_single_hamgnn_task(http_client, redis_cli, task_id, task_data))
            
            # 【关键】并发执行所有任务
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        return f"异步处理了 {len(results)} 个任务, 成功 {success_count} 个"

    finally:
        # 【关键】确保在使用后关闭Redis连接池
        await redis_cli.close()



async def process_single_hamgnn_task(http_client: httpx.AsyncClient, redis_cli: aioredis.Redis, task_id: str, task_data: dict):
    """
    处理单个HamGNN任务的协程。
    所有客户端都作为参数传入，以确保它们属于同一个事件循环。
    """
    task_lock_key = f"hamgnn_processing_lock:{task_id}"
    current_running_key = 'running_hamgnn_jobs'
    workdir = task_data.get('workdir')

    if not await redis_cli.set(task_lock_key, "1", nx=True, ex=600):
        logger.info(f"HamGNN任务 {task_id} 正在被其他进程处理，跳过")
        return False

    try:
        if not await redis_cli.hexists(QUEUE_HAMGNN_WAITING, task_id):
            logger.info(f"任务 {task_id} 不在HamGNN队列中，可能已被处理")
            return False

        predict_url = get_server_url("hamgnn") + "/predict"
        graph_data_path = os.path.join(workdir, "graph_data.npz")
        
        job_ticket = {
            "request_id": f"{task_id}_{int(time.time())}",
            "graph_data_path": str(graph_data_path), 
            "output_path": workdir, 
            "evaluate_loss": task_data.get('workflow_params', {}).get('evaluate_loss', False)
        }
        
        await redis_cli.incr(current_running_key)
        
        logger.info(f"异步提交任务 {task_id} 到HamGNN服务器: {predict_url}")
        response = await http_client.post(predict_url, json=job_ticket)
        response.raise_for_status()
        
        response_data = response.json()
        hamiltonian_path = response_data.get('output_file')
        if not hamiltonian_path:
            raise ValueError("预测结果中未包含哈密顿量文件路径")

        task_data['hamiltonian_path'] = hamiltonian_path
        task_data['status'] = 'hamgnn_completed'
        task_data['status_log'].append({ 'timestamp': time.time(), 'status': 'hamgnn_completed', 'message': 'HamGNN预测已完成' })
        
        if await _async_move_task(redis_cli, task_id, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, task_data):
            logger.info(f"任务 {task_id} 的HamGNN预测已完成(异步), 已移至后处理等待队列")
            await redis_cli.set(f"hamgnn_processed:{task_id}", "1", ex=7*24*60*60)
            return True
        else:
            logger.error(f"异步移动任务 {task_id} 到后处理等待队列失败")
            return False

    except Exception as e:
        logger.error(f"处理异步HamGNN任务 {task_id} 时出错: {e}")
        await _async_handle_hamgnn_task_failure(redis_cli, task_id, task_data, workdir, str(e), f"异步请求执行失败: {traceback.format_exc()}")
        return e
    finally:
        await redis_cli.decr(current_running_key)
        await redis_cli.delete(task_lock_key)


async def _async_handle_hamgnn_task_failure(redis_cli: aioredis.Redis, task_id: str, task_data: dict, workdir: str, error_message: str, status_message: str):
    """
    handle_hamgnn_task_failure 的异步版本，接收一个异步redis客户端作为参数。
    """
    task_data['status'] = 'hamgnn_failed'
    task_data['error'] = error_message
    task_data['status_log'] = task_data.get('status_log', [])
    task_data['status_log'].append({
        'timestamp': time.time(),
        'status': 'hamgnn_failed',
        'message': f'HamGNN预测失败: {status_message}'
    })
    
    # 文件写入是同步操作，可以直接调用
    _write_failure_file(
        workdir=workdir,
        stage_name='2/4: HamGNN预测',
        details=f'HamGNN预测失败: {error_message}'
    )
    
    # 调用异步移动函数
    await _async_move_task(redis_cli, task_id, QUEUE_HAMGNN_WAITING, QUEUE_COMPLETED, task_data)

Max_PostProcess_Batch_Size= config.get('concurrency', {}).get('max_postprocess_batch_size', 16)

@celery_app.task
def dispatch_postprocess_tasks():
    """
    F4定时任务: 将后处理等待队列的任务提交给postprocessServer,得到job_id后留在队列中等待作业完成
    【已修正】: 重构并优化了分布式锁逻辑，使其更健壮和清晰。
    """
    try:
        # 从配置中获取最大并发数
        max_concurrent = config.get('concurrency', {}).get('max_postprocess_jobs', 10)
        
        # 检查当前运行的后处理作业数
        current_running = redis_client.scard('running_postprocess_jobs')
        
        # 计算可提交的任务数量
        slots_available = max(0, max_concurrent - current_running)
        submission_limit = min(slots_available, Max_PostProcess_Batch_Size)
        
        if submission_limit <= 0:
            logger.info(f"后处理并发数已达上限({max_concurrent}),当前运行: {current_running}")
            return f"后处理并发数已达上限({max_concurrent}),跳过调度"
        
        # 获取后处理等待队列中的所有任务
        postprocess_tasks = redis_client.hgetall(QUEUE_POST_WAITING)
        
        if not postprocess_tasks:
            return "没有等待中的后处理任务"
        
        # 按创建时间排序(先进先出)
        sorted_tasks = []
        for task_id, task_data_json in postprocess_tasks.items():
            try:
                task_data = json.loads(task_data_json)
                # 跳过已经提交给后处理服务器的任务
                if 'postprocess_job_id' in task_data:
                    continue
                sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
            except json.JSONDecodeError:
                logger.error(f"任务 {task_id} 的数据格式无效，跳过")
                # 考虑直接将格式错误的任务移到完成队列
                _write_failure_file(
                    workdir=json.loads(task_data_json).get('workdir'),
                    stage_name='4/4: 后处理',
                    details=f'任务数据格式无效: {task_data_json}'
                )
                _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, json.loads(task_data_json))
                continue
        
        sorted_tasks = heapq.nsmallest(submission_limit, sorted_tasks, key=lambda x: x[2])
        
        # 处理可提交的任务
        tasks_processed = 0
        for task_id, task_data, _ in sorted_tasks:
            if tasks_processed >= submission_limit:
                break
            
            # 【修正】为每个任务创建一个唯一的锁
            task_lock_key = f"postprocess_processing_lock:{task_id}"
            
            # 【修正】尝试获取锁
            if not redis_client.set(task_lock_key, "1", nx=True, ex=600):  # 10分钟锁
                logger.info(f"后处理任务 {task_id} 正在被其他进程处理，跳过")
                continue
            
            try:
                # 【修正】再次检查任务是否在队列中
                if not redis_client.hexists(QUEUE_POST_WAITING, task_id):
                    logger.info(f"任务 {task_id} 不在后处理队列中，可能已被处理")
                    continue
                
                # 提取任务参数
                hamiltonian_path = task_data.get('hamiltonian_path')
                workdir = task_data.get('workdir')
                workflow_params = task_data.get('workflow_params', {})
                output_path = workdir

                ncpus = workflow_params.get('ncpus', 4)
                workflow_params_postprocess = workflow_params.copy()
                if workflow_params.get('partition') == 'auto':
                    logger.info("检测到 'partition' 参数为 'auto'，开始自动选择分区...")
                    best_partition = _get_best_partition(ncpus=ncpus)
                    workflow_params_postprocess['partition'] = best_partition
                    logger.info(f"已自动选择分区: {best_partition}")
                
                # 检查必要参数
                if not hamiltonian_path:
                    raise ValueError("任务数据中缺少哈密顿量文件路径")
                    
                # 构建后处理URL和请求参数
                postprocess_url = get_server_url("postprocess") + "/band_cal"
                graph_data_path = os.path.join(workdir, "graph_data.npz")
                
                # 更新任务状态
                task_data['status'] = 'submitting_to_postprocess'
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'submitting_to_postprocess',
                    'message': '正在提交给后处理服务',
                    'worker_id': os.getpid()
                })
                
                # 创建请求参数
                request_id = f"{task_id}_{int(time.time())}"
                job_ticket = {
                    "request_id": request_id,
                    "hamiltonian_path": str(hamiltonian_path),
                    "graph_data_path": str(graph_data_path),
                    "band_para": workflow_params_postprocess,
                    "output_path": output_path
                }
                
                # 提交给后处理服务器
                logger.info(f"提交任务 {task_id} 到后处理服务器: {postprocess_url}")
                response = requests.post(postprocess_url, json=job_ticket, timeout=TIMEOUT)
                response.raise_for_status()
                
                # 解析响应
                response_data = response.json()
                postprocess_job_id = response_data['job_id']
                
                # 更新任务数据
                task_data['postprocess_job_id'] = postprocess_job_id
                task_data['status'] = 'submitted_to_postprocess'
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'submitted_to_postprocess',
                    'message': f'已提交给后处理服务器,作业ID: {postprocess_job_id}'
                })
                
                # 将Slurm作业ID添加到监控列表
                redis_client.sadd('running_postprocess_jobs', postprocess_job_id)
                redis_client.sadd('monitored_slurm_jobs', postprocess_job_id)
                
                # 更新队列中的任务数据(注意不移动队列，等待Slurm作业完成)
                redis_client.hset(QUEUE_POST_WAITING, task_id, json.dumps(task_data))
                
                logger.info(f"任务 {task_id} 已提交给后处理服务器,Slurm作业ID: {postprocess_job_id}")
                tasks_processed += 1
                
            except Exception as e:
                logger.error(f"提交任务 {task_id} 到后处理服务器时出错: {e}")
                task_data['status'] = 'postprocess_failed'
                task_data['error'] = str(e)
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'postprocess_failed',
                    'message': f'提交到后处理服务器失败: {str(e)}'
                })
                _write_failure_file(
                    workdir=task_data.get('workdir'),
                    stage_name='4/4: 后处理',
                    details=f'提交到后处理服务器失败: {str(e)}'
                )
                # 将失败的任务直接移到完成队列
                _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
            
            finally:
                # 【修正】无论成功或失败，最后都释放锁
                redis_client.delete(task_lock_key)

    except Exception as e:
        logger.error(f"后处理调度任务执行时发生未知错误: {e}")
        return f"执行错误: {str(e)}"
    
    return f"处理了 {tasks_processed} 个后处理任务"
# --- 清理任务 ---
@celery_app.task
def cleanup_stale_locks_and_counters():
    """定期清理过期的锁和不准确的计数器"""
    # 获取所有处理锁
    processing_locks = []
    for key in redis_client.scan_iter(match="*_lock:*"):
        processing_locks.append(key)
    
    # 获取当前队列中的任务
    all_tasks = set()
    all_tasks.update(redis_client.hkeys(QUEUE_PENDING))
    all_tasks.update(redis_client.hkeys(QUEUE_OPENMX_WAITING))
    all_tasks.update(redis_client.hkeys(QUEUE_HAMGNN_WAITING))
    all_tasks.update(redis_client.hkeys(QUEUE_POST_WAITING))
    
    # 清理不再需要的锁
    for lock in processing_locks:
        task_id = lock.split(":")[-1]
        if task_id not in all_tasks:
            redis_client.delete(lock)
            logger.info(f"清理了过期的任务锁: {lock}")
    
    # 修正运行计数器
    for counter_key, max_val in [
        ('running_preprocess_jobs', config.get('concurrency', {}).get('max_openmx_jobs', 10)),
        ('running_hamgnn_jobs', config.get('concurrency', {}).get('max_hamgnn_jobs', 5)),
        ('running_postprocess_jobs', config.get('concurrency', {}).get('max_postprocess_jobs', 10))
    ]:
        current_running = int(redis_client.get(counter_key) or 0)
        
        # 如果计数器值不合理(大于最大并发数的2倍)，则重置它
        if current_running > max_val * 2:
            redis_client.set(counter_key, 0)
            logger.warning(f"重置了异常的计数器 {counter_key}: {current_running} -> 0")
    
    return "清理完成"

@celery_app.task
def recover_stuck_tasks():
    """恢复处理中卡住的任务，检查所有带锁但长时间未更新的任务"""
    # 查找所有锁定的任务
    for task_type, lock_prefix, queue_name in [
        ("hamgnn", "hamgnn_processing_lock:", QUEUE_HAMGNN_WAITING),
        ("postprocess", "processing_lock:", QUEUE_POST_WAITING),
        ("openmx", "openmx_processing_lock:", QUEUE_OPENMX_WAITING)
    ]:
        # 获取队列中的所有任务
        tasks = redis_client.hgetall(queue_name)
        
        for task_id, task_data_json in tasks.items():
            # 检查任务是否有锁
            lock_key = f"{lock_prefix}{task_id}"
            if redis_client.exists(lock_key):
                # 检查任务数据中的时间戳
                try:
                    task_data = json.loads(task_data_json)
                    last_update = 0
                    
                    # 找到最近的状态更新时间
                    for log in task_data.get('status_log', []):
                        if log.get('timestamp', 0) > last_update:
                            last_update = log.get('timestamp', 0)
                    # 如果超过5分钟未更新，认为任务卡住了
                    if (time.time() - last_update) > 300:  # 5分钟
                        logger.warning(f"发现卡住的{task_type}任务: {task_id}，已超过5分钟未更新")
                        # 释放锁
                        redis_client.delete(lock_key)
                        # 重置计数器
                        counter_key = f"running_{task_type}_jobs"
                        current = int(redis_client.get(counter_key) or 0)
                        if current > 0:
                            redis_client.decr(counter_key)
                except Exception as e:
                    logger.error(f"处理卡住任务时出错: {e}")
                    
    return "卡住任务恢复完成"

# --- 清理任务 ---

@celery_app.task
def cleanup_completed_task_files():
    """
    定期回收任务：清理'completed_tasks'队列中任务的工作目录下的特定文件。
    删除 .scfout, .npz, .npy 后缀和无后缀的文件，并从队列中移除任务。
    """
    logger.info("开始执行已完成任务的文件清理工作...")
    
    # hgetall() 获取的是字典的副本，操作安全
    completed_tasks = redis_client.hgetall(QUEUE_COMPLETED)
    if not completed_tasks:
        return "完成队列中没有需要清理的任务。"

    tasks_cleaned = 0
    files_deleted_count = 0
    
    # 定义要删除的文件后缀
    suffixes_to_delete = ('.scfout', '.npz', '.npy',".xyz",".UCell",".std",".input")

    for task_id, task_data_json in completed_tasks.items():
        try:
            task_data = json.loads(task_data_json)
            workdir = task_data.get('workdir')

            # 确认工作目录有效
            if not workdir or not os.path.isdir(workdir):
                logger.warning(f"任务 {task_id} 的工作目录 '{workdir}' 无效或不存在，将从完成队列中移除。")
                redis_client.hdel(QUEUE_COMPLETED, task_id)
                continue

            logger.debug(f"正在清理任务 {task_id} 的工作目录: {workdir}")
            
            # 使用 os.scandir() 高效遍历目录
            for entry in os.scandir(workdir):
                # 确认是文件而不是目录
                if entry.is_file():
                    # 检查文件名是否符合删除条件
                    is_suffix_match = entry.name.endswith(suffixes_to_delete)
                    # 一个简单的无后缀判断：文件名中不包含'.'
                    has_no_suffix = '.' not in entry.name

                    if is_suffix_match or has_no_suffix:
                        try:
                            os.remove(entry.path)
                            logger.info(f"已删除文件: {entry.path}")
                            files_deleted_count += 1
                        except OSError as e:
                            logger.error(f"删除文件 {entry.path} 时失败: {e}")

            # 清理完文件后，从完成队列中移除该任务记录
            redis_client.hdel(QUEUE_COMPLETED, task_id)
            tasks_cleaned += 1
            logger.info(f"任务 {task_id} 的文件已清理，并已从完成队列移除。")

        except json.JSONDecodeError:
            logger.error(f"无法解析任务 {task_id} 的数据，将从队列中移除以防循环错误。")
            redis_client.hdel(QUEUE_COMPLETED, task_id)
        except Exception as e:
            logger.error(f"清理任务 {task_id} 时发生未知错误: {e}")

    return f"清理完成。共处理了 {tasks_cleaned} 个任务，删除了 {files_deleted_count} 个文件。"

# 调用初始化函数
initialize_redis_keys()
# --- 定时任务注册 ---
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    设置Celery Beat周期性任务。
    这个函数会在Celery worker启动时(如果也启动了beat服务)自动运行。
    """
    # 从配置文件加载执行间隔
    periodic_tasks_config = config.get('periodic_tasks', {})
    
    # 健康检查
    sender.add_periodic_task(30.0, worker_healthcheck.s(), name='worker healthcheck every 30s')
    
    # F1: 调度OpenMX任务
    dispatch_openmx_interval = periodic_tasks_config.get('dispatch_openmx_interval', 5.0)
    logger.info(f"设置OpenMX调度定时任务,执行间隔: {dispatch_openmx_interval} 秒。")
    sender.add_periodic_task(
        dispatch_openmx_interval, 
        dispatch_openmx_tasks.s(), 
        name=f'dispatch OpenMX tasks every {dispatch_openmx_interval}s'
    )
    
    # F2: 轮询Slurm状态并调度
    poll_slurm_interval = periodic_tasks_config.get('poll_slurm_interval', 5.0)
    logger.info(f"设置Slurm轮询定时任务,执行间隔: {poll_slurm_interval} 秒。")
    sender.add_periodic_task(
        poll_slurm_interval, 
        poll_slurm_and_dispatch.s(), 
        name=f'poll Slurm and dispatch every {poll_slurm_interval}s'
    )
    
    # F3: 调度HamGNN任务
    # dispatch_hamgnn_interval = periodic_tasks_config.get('dispatch_hamgnn_interval', 5.0)
    # logger.info(f"设置HamGNN调度定时任务,执行间隔: {dispatch_hamgnn_interval} 秒。")
    # sender.add_periodic_task(
    #     dispatch_hamgnn_interval, 
    #     dispatch_hamgnn_tasks.s(), 
    #     name=f'dispatch HamGNN tasks every {dispatch_hamgnn_interval}s'
    # )
    dispatch_hamgnn_interval = periodic_tasks_config.get('dispatch_hamgnn_interval', 5.0)
    logger.info(f"设置HamGNN调度定时任务,执行间隔: {dispatch_hamgnn_interval} 秒。")
    sender.add_periodic_task(
        dispatch_hamgnn_interval, 
        dispatch_hamgnn_tasks_async.s(), # 注意这里调用的是新函数
        name=f'dispatch HamGNN tasks ASYNC every {dispatch_hamgnn_interval}s'
    )
    
    # F4: 调度后处理任务
    dispatch_postprocess_interval = periodic_tasks_config.get('dispatch_postprocess_interval', 5.0)
    logger.info(f"设置后处理调度定时任务,执行间隔: {dispatch_postprocess_interval} 秒。")
    sender.add_periodic_task(
        dispatch_postprocess_interval, 
        dispatch_postprocess_tasks.s(), 
        name=f'dispatch postprocess tasks every {dispatch_postprocess_interval}s'
    )

    update_partition_interval = periodic_tasks_config.get('update_partition_interval', 1.0)
    logger.info(f"设置Slurm分区状态查询定时任务,执行间隔: {update_partition_interval} 秒。")
    sender.add_periodic_task(
        update_partition_interval,
        update_slurm_partition_status.s(),
        name=f'update slurm partition status every {update_partition_interval}s'
    )
    
    # 定期清理任务
    sender.add_periodic_task(
        3000.0,  
        cleanup_stale_locks_and_counters.s(),
        name='cleanup stale resources every 5min'
    )
    
    # 恢复卡住任务
    sender.add_periodic_task(
        3000.0,  # 5分钟
        recover_stuck_tasks.s(),
        name='recover stuck tasks every 5min'
    )

    if config.get('workflow', {}).get('cleanup', False):
        cleanup_interval = config.get('periodic_tasks', {}).get('cleanup_completed_tasks_interval', 3600)
        logger.info(f"设置已完成任务文件清理定时任务, 执行间隔: {cleanup_interval} 秒。")
        sender.add_periodic_task(
            cleanup_interval,
            cleanup_completed_task_files.s(),
            name=f'cleanup completed task files every {cleanup_interval}s'
        )