"""
本模块定义了整个计算工作流的核心Celery任务。
它利用Celery实现分布式任务调度，通过Redis管理任务队列和状态缓存，
并与Slurm作业管理系统进行交互。整个工作流被分解为多个阶段，
包括OpenMX预处理、HamGNN预测、后处理等，由不同的定时任务驱动和监控。
"""

# --- 导入标准库和第三方库 ---
import os
import yaml
import json
import logging
import subprocess
import time
import random
import asyncio
import traceback
from pathlib import Path


# --- 异步和性能相关库 ---
import httpx  # 用于异步HTTP请求，替代requests
import psutil # 用于系统资源监控 (如内存)
import heapq  # 用于高效地获取队列中优先级最高的任务

# --- 核心框架和中间件库 ---
from celery import Celery
import requests
import redis
from redis import asyncio as aioredis # 引入异步Redis客户端

# --- 自定义工具函数 ---
from .utils import get_package_path, get_server_url

# --- 1. 初始化与配置 ---

# 初始化日志记录器
# 建议在Celery worker启动时也配置好日志级别，以确保日志能被正确捕获
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载任务配置文件
TASK_CONFIG_PATH = get_package_path('task_basic_config.yaml')
config = yaml.safe_load(open(TASK_CONFIG_PATH, 'r', encoding='utf-8'))
logger.info(f"成功加载任务配置文件: {TASK_CONFIG_PATH}")
logger.info(f"当前配置: {config}")

# 初始化同步Redis客户端
# 该客户端用于Celery任务中的自定义状态缓存、队列管理等操作。
# `decode_responses=True` 确保从Redis获取的值是UTF-8字符串而不是字节串。
try:
    # 假设Redis在本机默认端口运行，如果部署在其他地方，请修改host和port
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping() # 测试连接是否成功
    logger.info("Celery任务模块成功连接到Redis。")
except redis.exceptions.ConnectionError as e:
    logger.error(f"无法连接到Redis，请确保Redis服务正在运行且配置正确: {e}")
    # 如果无法连接到Redis，则关键功能无法使用，直接抛出异常阻止服务启动
    raise

# 初始化Celery应用
# 'tasks' 是当前模块的名称，Celery会据此自动命名任务。
# broker 是消息代理（任务订单中心）的地址。
# backend 是任务结果存储（任务状态看板）的地址。
celery_app = Celery('tasks',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0')

# 以下为可选的高级Celery配置，用于优化性能和稳定性，默认注释掉
# celery_app.conf.worker_prefetch_multiplier = 1  # 每个worker进程一次只预取1个任务，防止任务堆积
# celery_app.conf.task_acks_late = True           # 任务执行成功后才向broker发送确认回执
# celery_app.conf.worker_max_tasks_per_child = 10 # 每个worker子进程处理10个任务后自动重启，释放内存

# --- 2. 队列定义与常量 ---

# 使用语义化命名的五个Redis哈希键，作为自定义的任务队列
QUEUE_PENDING = 'pending_tasks'             # 队列A: 新任务的入口，等待被调度
QUEUE_OPENMX_WAITING = 'openmx_waiting_tasks' # 队列B: 已提交到OpenMX(Slurm)，等待计算完成
QUEUE_HAMGNN_WAITING = 'hamgnn_waiting_tasks' # 队列C: OpenMX完成后，等待HamGNN处理
QUEUE_POST_WAITING = 'postprocess_waiting_tasks'# 队列D: HamGNN完成后，等待后处理(Slurm)
QUEUE_COMPLETED = 'completed_tasks'         # 队列E: 任务完成或失败的最终归宿

# 全局网络请求超时时间 (秒)
TIMEOUT = 1200

# --- 3. 辅助函数 ---

def submit_request(process_url, job_ticket):
    """
    一个简单的同步HTTP POST请求辅助函数。

    Args:
        process_url (str): 目标服务的URL。
        job_ticket (dict): 要发送的JSON数据。

    Returns:
        dict: 服务端返回的JSON响应。

    Raises:
        requests.exceptions.RequestException: 当请求失败时抛出。
    """
    try:
        response = requests.post(
            process_url,
            json=job_ticket,
            timeout=TIMEOUT
        )
        response.raise_for_status()  # 如果HTTP状态码不是2xx，则抛出异常
        return response.json()
    except Exception as e:
        logger.error(f"向 {process_url} 提交请求失败: {e}")
        raise

def _move_task(task_id, from_queue, to_queue, task_data=None):
    """
    使用Redis的WATCH和事务(TRANSACTION)机制，实现原子性的任务队列转移。
    这在高并发环境下至关重要，能确保一个任务不会被多个worker同时处理和移动。

    Args:
        task_id (str): 任务的唯一ID。
        from_queue (str): 源队列的Redis键名。
        to_queue (str): 目标队列的Redis键名。
        task_data (dict, optional): 要更新的任务数据。如果为None，则会从源队列重新获取。

    Returns:
        bool: 如果任务成功移动，返回True；否则返回False。
    """
    # 如果未提供任务数据，先从源队列获取
    if task_data is None:
        current_data = redis_client.hget(from_queue, task_id)
        if not current_data:
            logger.debug(f"任务 {task_id} 不在队列 {from_queue} 中，无法移动。")
            return False
        try:
            task_data = json.loads(current_data)
        except json.JSONDecodeError:
            logger.error(f"任务 {task_id} 的数据格式无效 (非JSON)，无法解析。")
            return False

    # 使用WATCH和事务确保原子性，设置重试机制以应对高并发冲突
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            with redis_client.pipeline() as pipe:
                # 监视源队列。如果在接下来的 `pipe.execute()` 执行前，
                # 有其他客户端修改了 `from_queue`，则整个事务将失败。
                pipe.watch(from_queue)

                # 在事务开始前，再次检查任务是否存在。
                # 这可以防止在 `watch` 之后，任务已被其他进程移走的情况。
                if not redis_client.hexists(from_queue, task_id):
                    pipe.unwatch()
                    logger.debug(f"任务 {task_id} 在准备移动时已消失，可能已被其他worker处理。")
                    return False

                # 更新任务状态变更日志
                task_data['status_log'] = task_data.get('status_log', [])
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'from_queue': from_queue,
                    'to_queue': to_queue,
                    'worker_id': os.getpid(),  # 记录当前操作的worker进程ID
                    'move_attempt': retry_count + 1
                })

                # 开始事务（将多个命令打包）
                pipe.multi()

                # 核心操作：从源队列删除，并添加到目标队列
                pipe.hdel(from_queue, task_id)
                pipe.hset(to_queue, task_id, json.dumps(task_data))

                # 执行事务
                results = pipe.execute()
                
                # `pipe.execute()` 在 `watch` 成功时返回命令结果列表，
                # 在 `WatchError` 时抛出异常。如果返回空列表，也表示事务未执行。
                if results:
                    logger.debug(f"成功将任务 {task_id} 从 {from_queue} 移动到 {to_queue}")
                    return True
                else:
                    # 理论上，如果 watch 失败，会抛出 WatchError，但作为安全措施添加此日志
                    logger.warning(f"移动任务 {task_id} 的事务未执行，可能发生冲突。")
                    # 继续循环重试

        except redis.WatchError:
            # 如果 `watch` 的键被修改，这里会捕获到异常
            logger.debug(f"移动任务 {task_id} 时发生并发冲突，第 {retry_count + 1} 次重试...")
            retry_count += 1
            # 指数退避策略：每次重试前等待更长时间，避免活锁
            time.sleep(0.1 * (2 ** retry_count))
            continue

        except Exception as e:
            # 处理其他意外错误
            logger.error(f"移动任务 {task_id} 时发生未知错误: {e}")
            return False

    # 如果达到最大重试次数仍然失败
    logger.error(f"移动任务 {task_id} 从 {from_queue} 到 {to_queue} 失败，已达到最大重试次数 ({max_retries})。")
    return False

async def _async_move_task(redis_cli: aioredis.Redis, task_id: str, from_queue: str, to_queue: str, task_data: dict) -> bool:
    """
    `_move_task` 的异步版本，使用 `aioredis` 客户端。
    所有操作都是非阻塞的。

    Args:
        redis_cli (aioredis.Redis): 异步Redis客户端实例。
        task_id (str): 任务的唯一ID。
        from_queue (str): 源队列的Redis键名。
        to_queue (str): 目标队列的Redis键名。
        task_data (dict): 要更新的任务数据。

    Returns:
        bool: 如果任务成功移动，返回True；否则返回False。
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
    在任务的工作目录下写入一个标准格式的 `FAILURE.json` 文件。
    这有助于用户或后续流程快速定位问题。

    Args:
        workdir (str): 任务的工作目录路径。
        stage_name (str): 任务失败时所处的阶段名称 (例如, '1/4: OpenMX预处理')。
        details (str): 详细的错误信息。
    """
    if not workdir:
        logger.warning("工作目录未指定，无法写入FAILURE.json文件。")
        return
    try:
        # 确保工作目录存在
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
            
        logger.info(f"已在 {workdir} 写入失败信息文件: FAILURE.json")
    except Exception as file_error:
        # 即使写入文件失败，也只记录日志，不应影响主流程
        logger.error(f"写入失败信息到 {workdir} 时发生严重错误: {file_error}")




def _get_best_partition(ncpus: int = 4):
    """
    从Redis缓存的Slurm分区状态中，选择一个最合适的分区来运行任务。
    选择策略是：优先选择空闲CPU最多的分区。
    同时，此函数会以原子方式从所选分区的空闲CPU数中扣除所需CPU，实现资源预留。

    Args:
        ncpus (int): 本次任务需要消耗的CPU核心数。
        
    Returns:
        str: 成功预留资源的分区名称。如果没有找到合适的分区或操作失败，则返回默认分区。
    """
    partitions_key = 'slurm_partition_status'
    
    # 从配置中获取默认分区，支持配置为列表（随机选择）或字符串
    temp_val = config.get("slurm_monitor", {}).get("default_partition", "chu")
    default_partition = random.choice(temp_val) if isinstance(temp_val, list) else temp_val
    
    # 定义可用的Slurm分区状态
    AVAILABLE_STATES = {'idle', 'mixed', 'up', 'alloc', 'aggregated'}

    try:
        all_partitions = redis_client.hgetall(partitions_key)
        if not all_partitions:
            logger.warning(f"Redis中没有Slurm分区信息，将使用默认分区: {default_partition}")
            return default_partition

        # 1. 筛选出所有满足CPU数要求且状态可用的候选分区
        candidate_partitions = []
        for name, data_json in all_partitions.items():
            try:
                data = json.loads(data_json)
                # 检查分区状态和空闲CPU数
                if data.get('state') in AVAILABLE_STATES and data.get('idle_cpus', 0) >= ncpus:
                    # 排除带'*'的特殊分区名（通常是默认分区的别名）
                    if '*' in name or '*' in str(data.get('total_cpus', '')):
                        continue
                    candidate_partitions.append({'name': name, 'idle_cpus': data.get('idle_cpus', 0)})
            except (json.JSONDecodeError, ValueError):
                # 如果JSON解析失败或数据格式不正确，则跳过该分区
                continue
        
        if not candidate_partitions:
            logger.warning(f"没有找到任何至少有 {ncpus} 个空闲CPU的可用分区，将使用默认分区: {default_partition}")
            return default_partition
            
        # 2. 按空闲CPU数降序排序，优先选择最空闲的分区
        candidate_partitions.sort(key=lambda p: p['idle_cpus'], reverse=True)

        # 3. 循环尝试，直到成功为一个分区预留资源
        for candidate in candidate_partitions:
            partition_name = candidate['name']
            
            # 使用 WATCH 和事务来原子性地更新CPU数量
            with redis_client.pipeline() as pipe:
                try:
                    # 监视整个哈希表
                    pipe.watch(partitions_key)
                    
                    # 在事务开始前，再次获取最新的数据以进行最终确认
                    latest_data_json = pipe.hget(partitions_key, partition_name)
                    if not latest_data_json:
                        # 如果在此期间分区信息消失了，则跳过
                        pipe.unwatch()
                        continue
                    
                    latest_data = json.loads(latest_data_json)
                    
                    # 再次确认CPU数量是否仍然足够
                    if latest_data.get('idle_cpus', 0) < ncpus:
                        pipe.unwatch()
                        continue # CPU已被占用，尝试下一个分区
                        
                    # 开始事务
                    pipe.multi()
                    
                    # 计算并更新该分区的空闲CPU数
                    latest_data['idle_cpus'] -= ncpus
                    updated_data_json = json.dumps(latest_data)
                    pipe.hset(partitions_key, partition_name, updated_data_json)
                    
                    # 执行事务
                    results = pipe.execute()
                    
                    # 如果事务成功执行（没有被WATCH中断），说明资源预留成功
                    if results:
                        logger.info(f"成功为任务在分区 {partition_name} 预留 {ncpus} 个CPU。 "
                                    f"该分区剩余空闲CPU: {latest_data['idle_cpus']}")
                        return partition_name

                except redis.WatchError:
                    # 如果发生WatchError，说明在WATCH到EXEC之间数据被其他进程改变了
                    logger.debug(f"尝试为分区 {partition_name} 预留资源时发生竞争，将尝试下一个候选分区。")
                    continue # 继续循环，尝试下一个最空闲的分区
        
        # 4. 如果所有候选分区都尝试失败（通常是高并发导致），则返回默认分区作为后备
        logger.warning(f"所有候选分区在高并发下均预留失败，将使用默认分区: {default_partition}")
        return default_partition

    except Exception as e:
        logger.error(f"获取最佳分区时发生严重错误: {e}，将使用默认分区: {default_partition}")
        return default_partition



# --- 4. Celery 后台任务 ---
@celery_app.task
def worker_healthcheck():
    """一个简单的健康检查任务，返回worker的健康状态和内存使用率。"""
    return {"status": "healthy", "memory_usage": psutil.virtual_memory().percent}

@celery_app.task
def update_slurm_partition_status():
    """
    【定时任务】定期执行`sinfo`命令查询Slurm各分区的状态，
    并将聚合后的信息更新到Redis缓存中。
    此版本解决了`sinfo`对同一分区可能输出多行（例如不同状态的节点）导致的数据覆盖问题。
    """
    logger.info("开始查询并聚合Slurm分区状态...")
    excluded_partitions = set(config.get("default_parameters", {}).get("excluded_partitions", []))
    try:
        # 使用 `-o "%P|%T|%C"` 精确获取所需信息：分区名 | 节点状态 | CPU详情(已分配/空闲/其他/总共)
        command = ["sinfo", "-h", "-o", "%P|%T|%C"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # 1. 在内存中聚合数据，处理一个分区名对应多行输出的情况
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

                if partition_name in excluded_partitions or '*' in partition_name:
                    continue  # 跳过被排除的分区和带'*'的默认汇总行
                
                # CPU状态格式为 A/I/O/T (Allocated/Idle/Other/Total)
                cpu_stats = cpus_state_str.split('/')
                if len(cpu_stats) != 4:
                    logger.warning(f"CPU状态格式不为A/I/O/T，跳过: '{cpus_state_str}'")
                    continue

                idle_cpus_line = int(cpu_stats[1])
                total_cpus_line = int(cpu_stats[3])

                # 如果分区首次出现，则初始化
                if partition_name not in partitions_aggregated:
                    partitions_aggregated[partition_name] = {'idle_cpus': 0, 'total_cpus': 0}
                
                # 累加空闲和总CPU数
                partitions_aggregated[partition_name]['idle_cpus'] += idle_cpus_line
                partitions_aggregated[partition_name]['total_cpus'] += total_cpus_line

            except (ValueError, IndexError) as e:
                logger.warning(f"解析sinfo聚合行失败: '{line}', 错误: {e}")
                continue
        
        # 2. 将聚合后的最终结果通过事务一次性写入Redis
        pipe = redis_client.pipeline()
        partitions_key = 'slurm_partition_status'
        pipe.delete(partitions_key) # 先清空旧数据
        
        updated_partitions = []
        for name, data in partitions_aggregated.items():
            partition_data = {
                'state': 'aggregated', # 标记为聚合后的数据
                'total_cpus': data['total_cpus'],
                'idle_cpus': data['idle_cpus'],
                'updated_at': time.time()
            }
            pipe.hset(partitions_key, name, json.dumps(partition_data))
            updated_partitions.append(name)
        
        pipe.execute()

        if updated_partitions:
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
    【定时任务】批量查询所有被监控的Slurm作业的状态，并更新到Redis缓存。
    这个任务由Celery Beat定期调用，是实现被动监控的核心。
    """
    monitored_jobs = redis_client.smembers('monitored_slurm_jobs')
    if not monitored_jobs:
        return "当前没有需要监控的Slurm作业。"
    
    job_ids_str = ",".join(monitored_jobs)
    logger.info(f"批量查询 {len(monitored_jobs)} 个作业的状态: {job_ids_str}")
    
    try:
        # 使用`sacct`批量查询作业状态，`-P`确保输出格式易于解析
        command = ["sacct", "-j", job_ids_str, "-o", "JobID,State", "-n", "-P"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        pipe = redis_client.pipeline()
        updated_jobs = set()
        
        # 解析`sacct`的输出并更新Redis缓存
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            try:
                job_id, state = line.strip().split('|')[:2]
                state = state.strip()
                pipe.hset('slurm_job_status_cache', job_id, state)
                updated_jobs.add(job_id.split('.')[0]) # 添加主作业ID
            except ValueError:
                continue
        
        pipe.execute()
        
        # 检查哪些作业已经进入终结状态，并将它们从监控集合中移除
        TERMINAL_STATES = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]
        finished_jobs = set()
        for job_id in updated_jobs:
            state = redis_client.hget('slurm_job_status_cache', job_id)
            if state and any(term in state for term in TERMINAL_STATES):
                finished_jobs.add(job_id)

        if finished_jobs:
            redis_client.srem('monitored_slurm_jobs', *finished_jobs)
            logger.info(f"作业 {finished_jobs} 已进入终结状态, 从被动监控列表移除。")

        return f"已更新 {len(updated_jobs)} 个作业的状态。"
    
    except subprocess.CalledProcessError as e:
        # 如果`sacct`命令失败（例如，所有作业都已完成并从活动队列中清除），
        # 这通常是正常现象。我们可以安全地假设所有被查询的作业都已终结。
        logger.warning(f"sacct命令可能执行失败: {e.stderr}")
        redis_client.srem('monitored_slurm_jobs', *monitored_jobs)
        return "sacct查询无有效返回，已清理所有本次查询的监控作业。"
    except Exception as e:
        logger.error(f"批量查询Slurm作业时发生未知错误: {e}")
        return f"批量查询失败: {e}"

# --- 核心工作流任务 ---

def initialize_redis_keys():
    """
    在worker启动时，确保所有用于计数的Redis键都存在且类型正确。
    """
    # 清理并初始化后处理作业计数器 (集合类型)
    redis_client.delete('running_postprocess_jobs')
    # 清理并初始化OpenMX作业计数器 (集合类型)
    if redis_client.type('running_preprocess_jobs') != 'set':
        redis_client.delete('running_preprocess_jobs')
    # 清理并初始化HamGNN作业计数器 (字符串类型)
    if redis_client.type('running_hamgnn_jobs') != 'string':
        redis_client.delete('running_hamgnn_jobs')
        redis_client.set('running_hamgnn_jobs', '0')



@celery_app.task(
    bind=True,
    autoretry_for=(requests.exceptions.RequestException, ConnectionError), # 对网络异常自动重试
    retry_kwargs={'max_retries': config.get("celery_task", {}).get("max_retries", 1)},
    default_retry_delay=config.get("celery_task", {}).get("default_retry_delay", 60)
)
def start_workflow(self, structure_file_path: str, workflow_params: dict = {}):
    """
    【工作流入口函数】
    此任务是整个计算流程的起点。它只负责接收用户请求，
    生成一个唯一的任务ID，并将任务信息存入初始的“待处理队列”(`QUEUE_PENDING`)。

    Args:
        self (celery.Task): Celery任务实例，通过 `bind=True` 注入。
        structure_file_path (str): 输入的结构文件路径。
        workflow_params (dict): 包含工作流所有参数的字典。

    Returns:
        dict: 包含任务ID和状态信息的字典，用于向客户端返回即时响应。
    """
    try:

        #分区调度逻辑修改为在每个任务内
        # ncpus = workflow_params.get('ncpus', 4)
        # if workflow_params.get('partition') == 'auto':
        #     logger.info("检测到 'partition' 参数为 'auto'，开始自动选择分区...")
        #     best_partition = _get_best_partition(ncpus=ncpus)
        #     workflow_params['partition'] = best_partition
        #     logger.info(f"已自动选择分区: {best_partition}")

        # 创建唯一的任务ID
        task_id = self.request.id
        # 准备工作目录
        workdir = workflow_params.get('output_path', None)
        
        # 构造任务的初始数据结构
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
        
        # 将任务数据以JSON字符串形式存入Redis的`pending_tasks`哈希表中
        redis_client.hset(QUEUE_PENDING, task_id, json.dumps(task_data))
        
        logger.info(f"任务 {task_id} 已加入待处理队列, 结构文件: {structure_file_path}")
        
        # 更新Celery的后端状态，以便可以通过任务ID查询到初始状态
        self.update_state(
            state='PROGRESS',
            meta={
                'stage_code': 'QUEUED',
                'stage_name': '0/4: 待处理队列',
                'details': '任务已加入处理队列，等待调度器处理。',
                'workdir': workdir
            }
        )
        
        return {
            'task_id': task_id,
            'status': 'queued',
            'message': '任务已成功加入处理队列'
        }
    except Exception as e:
        logger.error(f"创建任务 {self.request.id} 时发生严重错误: {e}")
        # 如果在初始化阶段就失败，也尝试写入失败文件
        if 'workdir' in locals() and workdir:
             _write_failure_file(
                workdir=workdir,
                stage_name='0/4: 初始化失败',
                details=f'创建任务失败: {str(e)}'
             )
        # 更新Celery后端状态为失败
        self.update_state(
            state='FAILURE',
            meta={
                'stage_code': 'FAILED',
                'stage_name': '0/4: 初始化失败',
                'details': f'创建任务失败: {str(e)}'
            }
        )
        raise


# --- 四个独立的定时任务函数 --
Max_Openmx_Batch_Size = config.get('concurrency', {}).get('max_openmx_batch_size', 16)
@celery_app.task
def dispatch_openmx_tasks():
    """
    【F1 定时任务】调度OpenMX任务。
    - 检查当前OpenMX并发数是否已达上限。
    - 如果有空闲资源，从 `QUEUE_PENDING` 中获取任务。
    - 调用 `_get_best_partition` 自动选择Slurm分区。
    - 向 `openmxServer` 提交计算请求。
    - 成功后，将任务从 `QUEUE_PENDING` 移动到 `QUEUE_OPENMX_WAITING`。
    """
    max_concurrent = config.get('concurrency', {}).get('max_openmx_jobs', 10)
    current_running = redis_client.scard('running_preprocess_jobs')
    
    slots_available = max(0, max_concurrent - current_running)
    submission_limit = min(slots_available, Max_Openmx_Batch_Size)
    
    if submission_limit <= 0:
        logger.info(f"OpenMX并发数已达上限({max_concurrent})，当前运行: {current_running}，本次跳过调度。")
        return f"OpenMX并发数已达上限({max_concurrent})，跳过调度。"
    
    pending_tasks = redis_client.hgetall(QUEUE_PENDING)
    if not pending_tasks:
        return "待处理队列中没有任务。"
    
    # 使用heapq高效获取创建时间最早的一批任务
    sorted_tasks_meta = []
    for task_id, task_data_json in pending_tasks.items():
        try:
            task_data = json.loads(task_data_json)
            sorted_tasks_meta.append((task_id, task_data, task_data.get('created_at', 0)))
        except json.JSONDecodeError:
            logger.error(f"任务 {task_id} 数据格式无效，跳过。")
            continue
    tasks_to_process = heapq.nsmallest(submission_limit, sorted_tasks_meta, key=lambda x: x[2])
    
    tasks_processed = 0
    for task_id, task_data, _ in tasks_to_process:

        if tasks_processed >= submission_limit:
            break

        # 为每个任务创建一个分布式锁，防止多个worker同时处理同一个任务
        task_lock_key = f"openmx_processing_lock:{task_id}"
        # `nx=True`表示仅当键不存在时才设置，这是一个原子操作
        if not redis_client.set(task_lock_key, "1", nx=True, ex=600):  # 10分钟后自动过期
            logger.info(f"OpenMX任务 {task_id} 正在被其他进程锁定处理，跳过。")
            continue

        try:
            if not redis_client.hexists(QUEUE_PENDING, task_id):
                logger.info(f"任务 {task_id} 在加锁后已不在待处理队列，可能已被处理。")
                continue

            # 提取任务参数并准备提交
            structure_file_path = task_data.get('structure_file_path')
            workflow_params = task_data.get('workflow_params', {})
            ncpus = workflow_params.get('ncpus', 4)
            workflow_params_openmx = workflow_params.copy()
            
            # 如果分区设置为'auto'，则动态选择
            if workflow_params.get('partition') == 'auto':
                logger.info(f"任务 {task_id}: 检测到 'partition' 为 'auto'，开始自动选择分区...")
                best_partition = _get_best_partition(ncpus=ncpus)
                workflow_params_openmx['partition'] = best_partition
                logger.info(f"任务 {task_id}: 已自动选择分区 -> {best_partition}")
            
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
            logger.error(f"提交任务 {task_id} 到OpenMX时发生错误: {e}")
            # 更新任务状态为失败
            task_data['status'] = 'failed'
            task_data['error'] = str(e)
            _write_failure_file(
                workdir=task_data.get('workdir'),
                stage_name='1/4: OpenMX预处理提交失败',
                details=f'提交到OpenMX服务器失败: {str(e)}'
            )
            # 将失败的任务直接移到最终的完成队列
            _move_task(task_id, QUEUE_PENDING, QUEUE_COMPLETED, task_data)
        
        finally:
            # 无论成功或失败，最后都必须释放锁
            redis_client.delete(task_lock_key)
            
    return f"成功调度了 {tasks_processed} 个OpenMX任务。"

@celery_app.task
def poll_slurm_and_dispatch():
    """
    【F2 定时任务】轮询并处理依赖Slurm的任务。
    这个任务是连接两个Slurm步骤（OpenMX和后处理）的关键。
    它首先调用 `poll_slurm_jobs` 更新所有被动监控的作业状态，
    然后分别检查 `QUEUE_OPENMX_WAITING` 和 `QUEUE_POST_WAITING` 队列，
    将已完成的作业对应任务推向下一个阶段。
    """
    # 1. 更新所有被监控的Slurm作业状态到Redis缓存
    poll_slurm_jobs()
    
    # 2. 处理OpenMX等待队列中的任务
    openmx_tasks_processed = process_openmx_waiting_tasks()
    
    # 3. 处理后处理等待队列中的任务
    postprocess_tasks_processed = process_postprocess_waiting_tasks()
    
    return (f"OpenMX完成并流转: {openmx_tasks_processed}个; "
            f"后处理完成并流转: {postprocess_tasks_processed}个。")


def process_openmx_waiting_tasks():
    """
    处理 `QUEUE_OPENMX_WAITING` 队列中的任务。
    检查每个任务关联的Slurm作业是否完成，如果完成则移入 `QUEUE_HAMGNN_WAITING`。
    """
    openmx_tasks = redis_client.hgetall(QUEUE_OPENMX_WAITING)
    if not openmx_tasks:
        return 0
    
    tasks_processed = 0
    for task_id, task_data_json in openmx_tasks.items():
        task_data = json.loads(task_data_json)
        openmx_job_id = task_data.get('openmx_job_id')
        
        if not openmx_job_id:
            logger.error(f"任务 {task_id} 在OpenMX等待队列中但没有 'openmx_job_id'，跳过。")
            continue
            
        # 从缓存中检查Slurm作业状态
        state = redis_client.hget('slurm_job_status_cache', openmx_job_id)
        
        if state and "COMPLETED" in state:
            logger.info(f"OpenMX作业 {openmx_job_id} (任务 {task_id}) 已完成。")
            task_data['status'] = 'openmx_completed'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'openmx_completed',
                'message': 'OpenMX预处理已完成'
            })
            # 从运行和监控集合中移除
            redis_client.srem('running_preprocess_jobs', openmx_job_id)
            redis_client.srem('monitored_slurm_jobs', openmx_job_id)
            # 移动到下一个队列
            if _move_task(task_id, QUEUE_OPENMX_WAITING, QUEUE_HAMGNN_WAITING, task_data):
                tasks_processed += 1
                logger.info(f"任务 {task_id} 已移至HamGNN等待队列。")
            else:
                logger.error(f"移动任务 {task_id} 到HamGNN等待队列失败")
            
        elif state and any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
            logger.error(f"OpenMX作业 {openmx_job_id} (任务 {task_id}) 失败，状态: {state}。")
            task_data['status'] = 'openmx_failed'
            task_data['error'] = f"OpenMX作业失败，状态: {state}"
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'openmx_failed',
                'message': f'OpenMX预处理作业失败, 状态: {state}'
            })
            _write_failure_file(
                workdir=task_data.get('workdir'),
                stage_name='1/4: OpenMX预处理失败',
                details=f'Slurm作业 {openmx_job_id} 失败，状态: {state}'
            )
            # 从运行和监控集合中移除
            redis_client.srem('running_preprocess_jobs', openmx_job_id)
            redis_client.srem('monitored_slurm_jobs', openmx_job_id)
            # 将失败任务移到最终队列
            _move_task(task_id, QUEUE_OPENMX_WAITING, QUEUE_COMPLETED, task_data)
            
    return tasks_processed

def process_postprocess_waiting_tasks():
    """
    处理 `QUEUE_POST_WAITING` 队列中的任务。
    检查每个任务关联的Slurm作业是否完成，如果完成则移入 `QUEUE_COMPLETED`。
    """
    postprocess_tasks = redis_client.hgetall(QUEUE_POST_WAITING)
    if not postprocess_tasks:
        return 0
    
    tasks_processed = 0
    for task_id, task_data_json in postprocess_tasks.items():
        task_data = json.loads(task_data_json)
        postprocess_job_id = task_data.get('postprocess_job_id')
        
        if not postprocess_job_id:
            # 如果没有作业ID，说明任务还未被 `dispatch_postprocess_tasks` 提交，直接跳过
            continue

        # 检查Slurm作业状态    
        state = redis_client.hget('slurm_job_status_cache', postprocess_job_id)
        
        if state and "COMPLETED" in state:
            logger.info(f"后处理作业 {postprocess_job_id} (任务 {task_id}) 已完成。")
            task_data['status'] = 'postprocess_completed'
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'postprocess_completed',
                'message': '后处理已完成'
            })
            # 从监控列表中移除作业ID
            redis_client.srem('running_postprocess_jobs', postprocess_job_id)
            redis_client.srem('monitored_slurm_jobs', postprocess_job_id)
            
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
                logger.error(f"写入 SUCCESS.json 到 {task_data.get('workdir')} 时出错: {file_error}")
            
            # 标记任务为已完成(用于可能的幂等性检查)，并设置7天过期
            redis_client.set(f"completed:{task_id}", "1", ex=7*24*60*60)
            
            # 移动到最终队列
            if _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data):
                tasks_processed += 1
                logger.info(f"任务 {task_id} 已全部完成，并移至完成队列。")

        elif state and any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
            # 作业失败,更新任务状态
            logger.error(f"后处理作业 {postprocess_job_id} (任务 {task_id}) 失败，状态: {state}。")
            task_data['status'] = 'postprocess_failed'
            task_data['error'] = f"后处理作业失败，状态: {state}"
            task_data['status_log'].append({
                'timestamp': time.time(),
                'status': 'postprocess_failed',
                'message': f'后处理作业失败, 状态: {state}'
            })

            # 从监控列表中移除作业ID
            redis_client.srem('running_postprocess_jobs', postprocess_job_id)
            redis_client.srem('monitored_slurm_jobs', postprocess_job_id)

            # 写入失败标记文件
            _write_failure_file(
                workdir=task_data.get('workdir'),
                stage_name='4/4: 后处理失败',
                details=f'Slurm作业 {postprocess_job_id} 失败，状态: {state}'
            )
            
            # 将失败任务移到最终队列
            _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
            
    return tasks_processed

Max_Hamgnn_Batch_Size = config.get('concurrency', {}).get('max_hamgnn_batch_size', 12)
@celery_app.task
def dispatch_hamgnn_tasks_async():
    """
    【F3 定时任务 - 异步启动器】
    这是一个同步的Celery任务，其唯一职责是启动并运行 `run_hamgnn_dispatcher` 这个异步函数。
    这种模式允许在同步的Celery工作流中嵌入高性能的异步IO操作。
    """
    try:
        # `asyncio.run()` 会创建一个新的事件循环，运行指定的协程直到完成，然后关闭循环。
        result_message = asyncio.run(run_hamgnn_dispatcher())
        logger.info(result_message)
        return result_message
    except Exception as e:
        # 捕获异步流程中可能未被处理的顶层异常
        logger.error(f"HamGNN异步调度任务执行时发生顶层错误: {traceback.format_exc()}")
        return f"执行错误: {str(e)}"

async def run_hamgnn_dispatcher():
    """
    【异步调度器核心逻辑】
    - 检查HamGNN并发限制。
    - 从 `QUEUE_HAMGNN_WAITING` 获取任务。
    - 使用 `httpx.AsyncClient` 和 `aioredis` 并发地向 `hamgnnServer` 提交所有任务。
    - 等待所有异步请求完成后返回。
    """
    # 在异步函数内部创建和管理异步客户端，确保它们属于同一个事件循环
    redis_cli = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    try:
        max_concurrent = config.get('concurrency', {}).get('max_hamgnn_jobs', 5)
        current_running_key = 'running_hamgnn_jobs'
        
        await redis_cli.setnx(current_running_key, 0) # 确保计数器存在
        current_running = int(await redis_cli.get(current_running_key) or 0)
        
        slots_available = max(0, max_concurrent - current_running)
        submission_limit = min(slots_available, Max_Hamgnn_Batch_Size)
        
        if submission_limit <= 0:
            return f"HamGNN并发数已达上限({max_concurrent})，当前运行: {current_running}。"
            
        hamgnn_tasks = await redis_cli.hgetall(QUEUE_HAMGNN_WAITING)
        if not hamgnn_tasks:
            return "没有等待中的HamGNN任务。"

        # 筛选并排序任务
        sorted_tasks_meta = []
        for task_id, task_data_json in hamgnn_tasks.items():
            try:
                task_data = json.loads(task_data_json)
                sorted_tasks_meta.append((task_id, task_data, task_data.get('created_at', 0)))
            except json.JSONDecodeError:
                continue
        tasks_to_process = heapq.nsmallest(submission_limit, sorted_tasks_meta, key=lambda x: x[2])
        
        if not tasks_to_process:
            return "没有可处理的HamGNN任务。"

        # 使用 `async with` 管理异步HTTP客户端的生命周期
        async with httpx.AsyncClient(timeout=TIMEOUT) as http_client:
            # 为每个任务创建一个协程
            coroutines = [
                process_single_hamgnn_task(http_client, redis_cli, task_id, task_data)
                for task_id, task_data, _ in tasks_to_process
            ]
            # `asyncio.gather` 并发执行所有协程
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        return f"异步处理了 {len(results)} 个HamGNN任务, 成功 {success_count} 个。"
    finally:
        # 确保在使用后关闭Redis连接池
        await redis_cli.close()




async def process_single_hamgnn_task(http_client: httpx.AsyncClient, redis_cli: aioredis.Redis, task_id: str, task_data: dict):
    """
    处理单个HamGNN任务的协程。
    这是一个完全异步的函数，负责与`hamgnnServer`通信并将完成的任务移到下一个队列。

    Args:
        http_client (httpx.AsyncClient): 共享的异步HTTP客户端。
        redis_cli (aioredis.Redis): 共享的异步Redis客户端。
        task_id (str): 任务ID。
        task_data (dict): 任务数据。

    Returns:
        bool: 成功返回True，失败返回False或异常。
    """
    task_lock_key = f"hamgnn_processing_lock:{task_id}"
    current_running_key = 'running_hamgnn_jobs'
    workdir = task_data.get('workdir')

    if not await redis_cli.set(task_lock_key, "1", nx=True, ex=600):
        logger.info(f"HamGNN任务 {task_id} 正在被其他进程处理，跳过。")
        return False

    try:
        if not await redis_cli.hexists(QUEUE_HAMGNN_WAITING, task_id):
            logger.info(f"任务 {task_id} 不在HamGNN队列中，可能已被处理。")
            return False

        task_data['status'] = 'hamgnn_processing'
        task_data['status_log'].append({
            'timestamp': time.time(),
            'status': 'hamgnn_processing',
            'message': f'Worker {os.getpid()} locked task and is starting HTTP request.'
        })

        predict_url = get_server_url("hamgnn") + "/predict"
        graph_data_path = os.path.join(workdir, "graph_data.npz")
        
        job_ticket = {
            "request_id": f"{task_id}_{int(time.time())}",
            "graph_data_path": str(graph_data_path),
            "output_path": workdir,
            "evaluate_loss": task_data.get('workflow_params', {}).get('evaluate_loss', False)
        }
        
        await redis_cli.incr(current_running_key) # 原子性地增加运行计数
        
        logger.info(f"异步提交任务 {task_id} 到HamGNN服务器: {predict_url}")
        response = await http_client.post(predict_url, json=job_ticket)
        response.raise_for_status()
        
        response_data = response.json()
        workdir = response_data.get('workdir')
        workflow_params = task_data.get('workflow_params', {})
        ifscf=workflow_params.get('ifscf', False)
        if ifscf:
            hamiltonian_path = os.path.join(workdir, "scf_hamiltonian.npy")
        else:
            hamiltonian_path = os.path.join(workdir, "prediction_hamiltonian.npy")

        task_data['hamiltonian_path'] = hamiltonian_path
        task_data['status'] = 'hamgnn_completed'
        task_data['status_log'].append({'timestamp': time.time(), 'status': 'hamgnn_completed', 'message': 'HamGNN预测已完成'})
        
        if await _async_move_task(redis_cli, task_id, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, task_data):
            logger.info(f"任务 {task_id} 的HamGNN预测已完成(异步)，并移至后处理等待队列。")
            await redis_cli.set(f"hamgnn_processed:{task_id}", "1", ex=7*24*60*60)
            return True
        else:
            logger.error(f"异步移动任务 {task_id} 到后处理等待队列失败。")
            return False

    except Exception as e:
        logger.error(f"处理异步HamGNN任务 {task_id} 时出错: {e}")
        await _async_handle_hamgnn_task_failure(redis_cli, task_id, task_data, workdir, str(e), traceback.format_exc())
        return e # 返回异常对象
    finally:
        await redis_cli.decr(current_running_key) # 原子性地减少运行计数
        await redis_cli.delete(task_lock_key) # 释放锁

async def _async_handle_hamgnn_task_failure(redis_cli: aioredis.Redis, task_id: str, task_data: dict, workdir: str, error_message: str, status_message: str):
    """
    处理HamGNN任务失败的异步辅助函数。
    """
    task_data['status'] = 'hamgnn_failed'
    task_data['error'] = error_message
    task_data['status_log'] = task_data.get('status_log', [])
    task_data['status_log'].append({
        'timestamp': time.time(),
        'status': 'hamgnn_failed',
        'message': f'HamGNN预测失败: {status_message}'
    })
    
    # 文件写入是同步IO，但在异步函数中可以直接调用
    _write_failure_file(
        workdir=workdir,
        stage_name='2/4: HamGNN预测失败',
        details=f'HamGNN预测失败: {error_message}'
    )
    
    # 异步地将失败任务移到完成队列
    await _async_move_task(redis_cli, task_id, QUEUE_HAMGNN_WAITING, QUEUE_COMPLETED, task_data)


   
Max_PostProcess_Batch_Size= config.get('concurrency', {}).get('max_postprocess_batch_size', 16)
@celery_app.task
def dispatch_postprocess_tasks():
    """
    【F4 定时任务】调度后处理任务。
    - 检查后处理并发限制。
    - 从 `QUEUE_POST_WAITING` 获取尚未提交的任务。
    - 向 `postprocessServer` 提交计算请求，获取Slurm作业ID。
    - **注意**：与OpenMX不同，这里任务提交后 **仍留在** `QUEUE_POST_WAITING` 队列中，
      只是更新其数据，添加 `postprocess_job_id`。
    - 后续由 `poll_slurm_and_dispatch` 任务来监控其完成状态并移动到 `QUEUE_COMPLETED`。
    """
    try:
        max_concurrent = config.get('concurrency', {}).get('max_postprocess_jobs', 10)
        current_running = redis_client.scard('running_postprocess_jobs')
        
        slots_available = max(0, max_concurrent - current_running)
        submission_limit = min(slots_available, Max_PostProcess_Batch_Size)
        
        if submission_limit <= 0:
            logger.info(f"后处理并发数已达上限({max_concurrent})，当前运行: {current_running}，跳过调度。")
            return f"后处理并发数已达上限({max_concurrent})，跳过调度。"
        
        postprocess_tasks = redis_client.hgetall(QUEUE_POST_WAITING)
        if not postprocess_tasks:
            return "没有等待中的后处理任务。"
        
        # 筛选出尚未提交给Slurm的任务（即没有 'postprocess_job_id' 字段）
        sorted_tasks = []
        for task_id, task_data_json in postprocess_tasks.items():
            try:
                task_data = json.loads(task_data_json)
                if 'postprocess_job_id' in task_data:
                    continue # 已提交，跳过
                sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
            except json.JSONDecodeError:
                logger.error(f"任务 {task_id} 数据格式无效，将移至完成队列。")
                _write_failure_file(
                    workdir=json.loads(task_data_json).get('workdir'),
                    stage_name='4/4: 后处理',
                    details=f'任务数据格式无效: {task_data_json}'
                )

                # 将格式错误的任务直接移走
                _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, json.loads(task_data_json))
                continue
        
        tasks_to_process = heapq.nsmallest(submission_limit, sorted_tasks, key=lambda x: x[2])
        
        tasks_processed = 0
        for task_id, task_data, _ in tasks_to_process:

            if tasks_processed >= submission_limit:
                break

            #为每个任务创建一个唯一的锁
            task_lock_key = f"postprocess_processing_lock:{task_id}"
            # 尝试获取锁
            if not redis_client.set(task_lock_key, "1", nx=True, ex=600):
                logger.info(f"后处理任务 {task_id} 正在被其他进程锁定处理，跳过。")
                continue
            
            try:
                if not redis_client.hexists(QUEUE_POST_WAITING, task_id):
                    logger.info(f"任务 {task_id} 在加锁后已不在后处理队列，可能已被处理。")
                    continue
                
                # 提取参数并准备提交
                hamiltonian_path = task_data.get('hamiltonian_path')
                workdir = task_data.get('workdir')
                workflow_params = task_data.get('workflow_params', {})
                if not hamiltonian_path:
                    raise ValueError(f"任务 {task_id} 缺少 'hamiltonian_path'。")
                
                # 自动选择分区
                ncpus = workflow_params.get('ncpus', 4)
                workflow_params_postprocess = workflow_params.copy()
                if workflow_params.get('partition') == 'auto':
                    logger.info(f"任务 {task_id}: 后处理阶段自动选择分区...")
                    best_partition = _get_best_partition(ncpus=ncpus)
                    workflow_params_postprocess['partition'] = best_partition
                    logger.info(f"任务 {task_id}: 已自动选择分区 -> {best_partition}")

                # 更新任务状态
                task_data['status'] = 'submitting_to_postprocess'
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'submitting_to_postprocess',
                    'message': '正在提交给后处理服务',
                    'worker_id': os.getpid()
                })
                
                # 构建请求
                postprocess_url = get_server_url("postprocess") + "/band_cal"
                graph_data_path = os.path.join(workdir, "graph_data.npz")

                job_ticket = {
                    "request_id": f"{task_id}_{int(time.time())}",
                    "hamiltonian_path": str(hamiltonian_path),
                    "graph_data_path": str(graph_data_path),
                    "band_para": workflow_params_postprocess,
                    "output_path": workdir
                }
                
                # 提交请求
                logger.info(f"提交任务 {task_id} 到后处理服务器: {postprocess_url}")
                response = requests.post(postprocess_url, json=job_ticket, timeout=TIMEOUT)
                response.raise_for_status()
                
                # 解析响应，获取作业ID
                postprocess_job_id = response.json()['job_id']
                
                # 更新任务数据，添加作业ID
                task_data['postprocess_job_id'] = postprocess_job_id
                task_data['status'] = 'submitted_to_postprocess'
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'submitted_to_postprocess',
                    'message': f'已提交给后处理服务器, 作业ID: {postprocess_job_id}'
                })
                
                # 将作业ID添加到运行和监控集合
                redis_client.sadd('running_postprocess_jobs', postprocess_job_id)
                redis_client.sadd('monitored_slurm_jobs', postprocess_job_id)
                
                # 在原地更新队列中的任务数据
                redis_client.hset(QUEUE_POST_WAITING, task_id, json.dumps(task_data))
                
                logger.info(f"任务 {task_id} 已成功提交到后处理 (Slurm作业ID: {postprocess_job_id})。")
                tasks_processed += 1
                
            except Exception as e:
                logger.error(f"提交任务 {task_id} 到后处理时出错: {e}")
                task_data['status'] = 'postprocess_failed'
                task_data['error'] = str(e)
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'postprocess_failed',
                    'message': f'提交到后处理服务器失败: {str(e)}'
                })
                _write_failure_file(
                    workdir=task_data.get('workdir'),
                    stage_name='4/4: 后处理提交失败',
                    details=f'提交到后处理服务器失败: {str(e)}'
                )
                _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
            
            finally:
                redis_client.delete(task_lock_key)
        
        return f"成功调度了 {tasks_processed} 个后处理任务。"

    except Exception as e:
        logger.error(f"后处理调度任务执行时发生未知错误: {e}")
        return f"执行错误: {str(e)}"

# --- 6. 清理和维护任务 ---
# --- 清理任务 ---
@celery_app.task
def cleanup_stale_locks_and_counters():
    """
    【定时维护任务】定期清理可能因worker异常退出而残留的过期锁和不准确的计数器。
    这是一个健壮性设计，用于系统的自我修复。
    """
    logger.info("开始清理过期的锁和不一致的计数器...")
    
    # 1. 清理孤立的任务锁
    # 查找所有任务处理锁
    processing_locks = []
    # 精确匹配模式，避免误删其他锁
    for key in redis_client.scan_iter(match="*_processing_lock:*"):
        processing_locks.append(key)
    
    # 获取当前所有活动队列中的任务ID集合
    active_tasks_in_queues = set()
    active_tasks_in_queues.update(redis_client.hkeys(QUEUE_PENDING))
    active_tasks_in_queues.update(redis_client.hkeys(QUEUE_OPENMX_WAITING))
    active_tasks_in_queues.update(redis_client.hkeys(QUEUE_HAMGNN_WAITING))
    active_tasks_in_queues.update(redis_client.hkeys(QUEUE_POST_WAITING))
    
    # 如果一个锁对应的任务ID已经不在任何活动队列中，说明该锁是过期的
    for lock_key in processing_locks:
        try:
            task_id = lock_key.split(":")[-1]
            if task_id not in active_tasks_in_queues:
                redis_client.delete(lock_key)
                logger.warning(f"清理了孤立的任务锁: {lock_key}")
        except IndexError:
            logger.warning(f"发现格式不正确的锁: {lock_key}，已忽略。")

    # 2. 修正各个阶段的运行计数器
    logger.info("开始修正各阶段的运行计数器...")

    # A. OpenMX计数器修正 (基于Set)
    running_openmx_jobs_in_redis = redis_client.smembers('running_preprocess_jobs')
    tasks_in_openmx_queue = redis_client.hgetall(QUEUE_OPENMX_WAITING)
    # 从队列中的任务数据里提取出实际应该在运行的 openmx_job_id
    actual_running_openmx_ids = {
        json.loads(v).get('openmx_job_id') 
        for v in tasks_in_openmx_queue.values() 
        if json.loads(v).get('openmx_job_id')
    }
    
    # 找出在计数器集合中，但其对应任务已不在等待队列的ID
    stale_openmx_ids = running_openmx_jobs_in_redis - actual_running_openmx_ids
    if stale_openmx_ids:
        logger.warning(f"发现过期的OpenMX运行ID: {stale_openmx_ids}，正在从计数器中移除...")
        redis_client.srem('running_preprocess_jobs', *stale_openmx_ids)

    # B. 后处理计数器修正 (基于Set，逻辑同OpenMX)
    running_postprocess_jobs_in_redis = redis_client.smembers('running_postprocess_jobs')
    tasks_in_postprocess_queue = redis_client.hgetall(QUEUE_POST_WAITING)
    # 从队列中的任务数据里提取出实际应该在运行的 postprocess_job_id
    actual_running_postprocess_ids = {
        json.loads(v).get('postprocess_job_id') 
        for v in tasks_in_postprocess_queue.values() 
        if json.loads(v).get('postprocess_job_id')
    }

    # 找出在计数器集合中，但其对应任务已不在等待队列或已完成的ID
    stale_postprocess_ids = running_postprocess_jobs_in_redis - actual_running_postprocess_ids
    if stale_postprocess_ids:
        logger.warning(f"发现过期的后处理运行ID: {stale_postprocess_ids}，正在从计数器中移除...")
        redis_client.srem('running_postprocess_jobs', *stale_postprocess_ids)
        
    # C. HamGNN计数器修正 (基于简单的数字计数器)
    # 这种计数器无法通过ID比对，最佳修正方式是重新计算当前被锁定的任务数量
    hamgnn_lock_prefix = "hamgnn_processing_lock:"
    tasks_in_hamgnn_queue = redis_client.hgetall(QUEUE_HAMGNN_WAITING)
    actual_running_hamgnn_count = 0
    for task_id in tasks_in_hamgnn_queue.keys():
        # 如果一个在HamGNN等待队列中的任务有关联的锁，我们认为它正在被处理
        if redis_client.exists(f"{hamgnn_lock_prefix}{task_id}"):
            actual_running_hamgnn_count += 1
            
    current_hamgnn_counter_val = int(redis_client.get('running_hamgnn_jobs') or 0)
    
    # 如果Redis中的计数值与实际锁定的任务数不符，则进行修正
    if current_hamgnn_counter_val != actual_running_hamgnn_count:
        logger.warning(
            f"HamGNN运行计数器不一致。当前值: {current_hamgnn_counter_val}, "
            f"实际锁定任务数: {actual_running_hamgnn_count}。正在修正..."
        )
        redis_client.set('running_hamgnn_jobs', actual_running_hamgnn_count)
        
    logger.info("维护清理任务完成。")
    return "清理完成。"

stuck_threshold = config.get('workflow', {}).get('stuck_task_threshold', 3600)  # 默认1小时
@celery_app.task
def recover_stuck_tasks():
    """
    【定时维护任务】恢复处理中卡住的任务。
    检查所有被锁住但长时间没有状态更新的任务，并释放它们的锁，
    以便调度器可以重新处理它们。
    """
    logger.info("开始检查并恢复可能卡住的任务...")
    recovered_count = 0
    
    # 定义要检查的队列和对应的锁前缀
    queues_to_check = [
        ("OpenMX", QUEUE_OPENMX_WAITING, "openmx_processing_lock:"),
        ("HamGNN", QUEUE_HAMGNN_WAITING, "hamgnn_processing_lock:"),
        ("PostProcess", QUEUE_POST_WAITING, "postprocess_processing_lock:")
    ]

    for stage_name, queue_name, lock_prefix in queues_to_check:
        tasks_in_queue = redis_client.hgetall(queue_name)
        
        for task_id, task_data_json in tasks_in_queue.items():
            lock_key = f"{lock_prefix}{task_id}"
            
            # 如果任务存在锁
            if redis_client.exists(lock_key):
                try:
                    task_data = json.loads(task_data_json)
                    last_update_time = 0
                    
                    # 遍历状态日志，找到最近的更新时间戳
                    status_logs = task_data.get('status_log', [])
                    if status_logs:
                        last_update_time = max(log.get('timestamp', 0) for log in status_logs)
                    else:
                        last_update_time = task_data.get('created_at', 0)

                    # 如果距离上次更新超过一个阈值，则认为任务卡住了
                    if (time.time() - last_update_time) > stuck_threshold:
                        logger.warning(f"发现卡住的 {stage_name} 任务: {task_id}。上次更新在 {time.ctime(last_update_time)}。正在释放锁...")
                        
                        # 释放锁，让任务可以被重新调度
                        redis_client.delete(lock_key)
                        
                        # （可选）如果任务失败导致计数器未减少，这里可以尝试修复
                        # 例如，HamGNN使用的是incr/decr，如果它在请求后但在decr前崩溃，计数器会偏高
                        if stage_name == "HamGNN":
                            redis_client.decr('running_hamgnn_jobs')
                        
                        recovered_count += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"解析卡住的任务 {task_id} 数据时出错: {e}。将释放锁以防万一。")
                    redis_client.delete(lock_key)
    
    return f"卡住任务恢复检查完成，共处理了 {recovered_count} 个任务。"
# --- 清理任务 ---
suffixes_to_delete=config.get('workflow', {}).get('cleanup_suffixes', ('.scfout', '.npz', '.npy', ".xyz", ".UCell", ".std", ".input"))
@celery_app.task
def cleanup_completed_task_files():
    """
    【定时任务】定期清理`completed_tasks`队列中任务的工作目录。
    删除指定的临时文件（如.scfout, .npz等），以释放磁盘空间。
    清理完毕后，将任务从`completed_tasks`队列中移除。
    **请谨慎开启此任务，因为它会删除所有已完成任务的工作目录中的指定文件。**
    """
    logger.info("开始执行已完成任务的文件清理工作...")
    completed_tasks = redis_client.hgetall(QUEUE_COMPLETED)
    if not completed_tasks:
        return "完成队列中没有需要清理的任务。"

    tasks_cleaned = 0
    files_deleted_count = 0
    for task_id, task_data_json in completed_tasks.items():
        try:
            task_data = json.loads(task_data_json)
            workdir = task_data.get('workdir')

            if not workdir or not os.path.isdir(workdir):
                logger.warning(f"任务 {task_id} 的工作目录 '{workdir}' 无效，将直接从完成队列移除。")
                redis_client.hdel(QUEUE_COMPLETED, task_id)
                continue

            logger.debug(f"正在清理任务 {task_id} 的工作目录: {workdir}")
            
            # 使用 os.scandir() 高效遍历目录
            for entry in os.scandir(workdir):
                if entry.is_file():
                    # 如果文件后缀匹配或文件没有后缀，则删除
                    if entry.name.endswith(suffixes_to_delete) or '.' not in entry.name:
                        try:
                            os.remove(entry.path)
                            logger.info(f"已删除文件: {entry.path}")
                            files_deleted_count += 1
                        except OSError as e:
                            logger.error(f"删除文件 {entry.path} 时失败: {e}")

            # 清理完文件后，从完成队列中移除该任务
            redis_client.hdel(QUEUE_COMPLETED, task_id)
            tasks_cleaned += 1
            logger.info(f"任务 {task_id} 的文件已清理，并已从完成队列移除。")

        except json.JSONDecodeError:
            logger.error(f"无法解析任务 {task_id} 的数据，将从队列移除以防循环错误。")
            redis_client.hdel(QUEUE_COMPLETED, task_id)
        except Exception as e:
            logger.error(f"清理任务 {task_id} 时发生未知错误: {e}")

    return f"清理完成。共处理了 {tasks_cleaned} 个任务，删除了 {files_deleted_count} 个文件。"

# --- 7. 定时任务注册 ---

# 在Celery应用配置完成后，注册所有周期性任务
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    设置Celery Beat周期性任务。
    此函数在Celery worker启动时（如果同时启动beat服务）自动运行。
    """
    periodic_tasks_config = config.get('periodic_tasks', {})
    
    # 任务1: Worker健康检查
    sender.add_periodic_task(30.0, worker_healthcheck.s(), name='worker healthcheck every 30s')
    
    # 任务2: 调度OpenMX任务 (F1)
    interval = periodic_tasks_config.get('dispatch_openmx_interval', 5.0)
    logger.info(f"注册OpenMX调度任务，执行间隔: {interval} 秒。")
    sender.add_periodic_task(interval, dispatch_openmx_tasks.s(), name=f'dispatch OpenMX tasks every {interval}s')
    
    # 任务3: 轮询Slurm状态并推进工作流 (F2)
    interval = periodic_tasks_config.get('poll_slurm_interval', 5.0)
    logger.info(f"注册Slurm轮询任务，执行间隔: {interval} 秒。")
    sender.add_periodic_task(interval, poll_slurm_and_dispatch.s(), name=f'poll Slurm and dispatch every {interval}s')
    
    # 任务4: 调度HamGNN任务 (F3 - 异步版)
    interval = periodic_tasks_config.get('dispatch_hamgnn_interval', 5.0)
    logger.info(f"注册HamGNN(异步)调度任务，执行间隔: {interval} 秒。")
    sender.add_periodic_task(interval, dispatch_hamgnn_tasks_async.s(), name=f'dispatch HamGNN tasks ASYNC every {interval}s')
    
    # 任务5: 调度后处理任务 (F4)
    interval = periodic_tasks_config.get('dispatch_postprocess_interval', 5.0)
    logger.info(f"注册后处理调度任务，执行间隔: {interval} 秒。")
    sender.add_periodic_task(interval, dispatch_postprocess_tasks.s(), name=f'dispatch postprocess tasks every {interval}s')

    # 任务6: 更新Slurm分区状态缓存
    interval = periodic_tasks_config.get('update_partition_interval', 1.0)
    logger.info(f"注册Slurm分区状态更新任务，执行间隔: {interval} 秒。")
    sender.add_periodic_task(interval, update_slurm_partition_status.s(), name=f'update slurm partition status every {interval}s')

    # 任务7和8: 任务维护（可选）
    if config.get('workflow', {}).get('task_maintenance', False):
        task_maintenance_interval = config.get('workflow', {}).get('task_maintenance_interval', 300)
        logger.info(f"注册任务维护任务，执行间隔: {task_maintenance_interval} 秒。")
        # 任务7: 清理过期的锁和计数器
        sender.add_periodic_task(task_maintenance_interval, cleanup_stale_locks_and_counters.s(), name=f'cleanup stale resources every {task_maintenance_interval}s')
        # 任务8: 恢复卡住的任务
        sender.add_periodic_task(task_maintenance_interval, recover_stuck_tasks.s(), name=f'recover stuck tasks every {task_maintenance_interval}s')

    # 任务9: 清理已完成任务的临时文件 (可选)
    if config.get('workflow', {}).get('cleanup', False):
        interval = config.get('workflow', {}).get('cleanup_completed_tasks_interval', 3600)
        logger.info(f"注册已完成任务文件清理任务，执行间隔: {interval} 秒。")
        sender.add_periodic_task(interval, cleanup_completed_task_files.s(), name=f'cleanup completed task files every {interval}s')

# --- 启动时初始化 ---
# 调用初始化函数，确保Redis中的计数器状态正确
initialize_redis_keys()



######################以下为已弃用代码######################

# def wait_for_slurm_job(job_id: str,
#                        first_time: int = config.get("slurm_monitor", {}).get("first_time", 10),
#                        poll_interval: int = config.get("slurm_monitor", {}).get("poll_interval", 1),
#                        timeout: int = 7200) -> bool:
#     """
#     [已弃用]对于大量并发请求会导致celery worker被阻塞，现改用异步队列模式管理
#     通过轮询Redis中的缓存来等待一个Slurm作业完成，而不是直接调用Slurm命令。
#     这种方式可以显著降低对Slurm控制器的压力。

#     Args:
#         job_id (str): 要监控的Slurm作业ID。
#         first_time (int): 首次查询前的初始等待时间（秒）。
#         poll_interval (int): 轮询间隔（秒）。
#         timeout (int): 最长等待时间（秒）。

#     Returns:
#         bool: 如果作业成功完成返回True，如果失败或超时返回False。
#     """
#     logger.info(f"开始通过Redis缓存监控Slurm作业 {job_id}...")
#     start_time = time.time()
    
#     # 将此作业ID添加到全局的被动监控集合中，由 `poll_slurm_jobs` 任务统一查询
#     redis_client.sadd('monitored_slurm_jobs', job_id)
    
#     # 初始等待，给Slurm一些时间来调度和更新作业状态
#     time.sleep(first_time)
    
#     while (time.time() - start_time) < timeout:
#         logger.info(f"正在查询作业 {job_id} 的状态...")
#         state = redis_client.hget('slurm_job_status_cache', job_id)
        
#         if state:
#             state = state.strip()
#             if "COMPLETED" in state:
#                 logger.info(f"从缓存中检测到作业 {job_id} 成功完成。")
#                 redis_client.srem('monitored_slurm_jobs', job_id)
#                 redis_client.hdel('slurm_job_status_cache', job_id)
#                 return True
#             elif any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
#                 logger.error(f"从缓存中检测到作业 {job_id} 失败，状态为: {state}。")
#                 redis_client.srem('monitored_slurm_jobs', job_id)
#                 redis_client.hdel('slurm_job_status_cache', job_id)
#                 return False
        
#         # 等待下一个轮询周期
#         time.sleep(poll_interval)
        
#     logger.error(f"等待作业 {job_id} 超时 ({timeout}秒)。")
#     redis_client.srem('monitored_slurm_jobs', job_id)
#     redis_client.hdel('slurm_job_status_cache', job_id)
#     return False

# import concurrent
# from concurrent.futures import ThreadPoolExecutor, as_completed
# @celery_app.task
# def dispatch_hamgnn_tasks():
#     """
#     【已废弃的同步版本】F3定时任务: 将HamGNN等待队列的任务提交给hamgnnServer。

#     .. deprecated::
#        此函数已被基于 `asyncio` 和 `httpx` 的异步版本 `dispatch_hamgnn_tasks_async` 替代。
#        当前版本使用 `ThreadPoolExecutor` 来并发处理阻塞式的 `requests` 网络请求。
#        虽然能够实现并发，但在高吞吐量下，线程切换的开销比异步IO更大。
#        保留此代码仅为历史参考和技术演进对比。

#     主要逻辑:
#     1. 检查全局并发数限制。
#     2. 从 `QUEUE_HAMGNN_WAITING` 队列中获取一批任务。
#     3. 为每个任务尝试获取一个分布式锁，以确保只有一个Worker能处理它。
#     4. 将成功锁定的任务提交到一个线程池中，由线程池中的线程执行HTTP请求。
#     5. 使用 `as_completed` 异步地收集已完成的请求结果。
#     6. 根据请求结果（成功/失败），将任务移动到下一个队列或标记为失败。
#     7. 在任务处理完成后（无论成功或失败），务必释放锁并更新并发计数器。
#     """
#     try:
#         # --- 1. 检查并发限制 ---
#         max_concurrent = config.get('concurrency', {}).get('max_hamgnn_jobs', 5)
#         current_running_key = 'running_hamgnn_jobs'
        
#         # 如果计数器键不存在，则原子性地设置为0
#         redis_client.setnx(current_running_key, 0)
        
#         current_running = int(redis_client.get(current_running_key) or 0)
        
#         # 计算本次可调度的任务空位数
#         slots_available = max(0, max_concurrent - current_running)
#         submission_limit = min(slots_available, Max_Hamgnn_Batch_Size)
        
#         if submission_limit <= 0:
#             logger.info(f"HamGNN并发数已达上限({max_concurrent})，当前运行: {current_running}，跳过本轮调度。")
#             return f"HamGNN并发数已达上限({max_concurrent})，跳过调度"
        
#         # --- 2. 从等待队列获取一批任务 ---
#         hamgnn_tasks = redis_client.hgetall(QUEUE_HAMGNN_WAITING)
#         if not hamgnn_tasks:
#             return "没有等待中的HamGNN任务"
        
#         # 解析任务数据，并按创建时间排序，实现先进先出
#         sorted_tasks = []
#         for task_id, task_data_json in hamgnn_tasks.items():
#             try:
#                 task_data = json.loads(task_data_json)
#                 sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
#             except json.JSONDecodeError:
#                 logger.error(f"任务 {task_id} 的数据格式无效(非JSON)，跳过。")
#                 continue
        
#         # 使用heapq高效地获取最早创建的、数量不超过上限的一批任务
#         sorted_tasks = heapq.nsmallest(submission_limit, sorted_tasks, key=lambda x: x[2])

#         # 初始化统计变量
#         tasks_processed_successfully = 0
#         tasks_found_locked = 0
#         tasks_already_completed = 0
        
#         # --- 3. 使用线程池并发处理任务 ---
#         # 线程池用于并发执行网络请求(IO密集型操作)，避免阻塞Celery主进程。
#         # max_workers 设为可提交任务数和实际任务数的较小值，避免创建不必要的线程。
#         with ThreadPoolExecutor(max_workers=min(submission_limit, len(sorted_tasks))) as executor:
#             futures = {}  # 字典: {Future对象 -> (任务ID, 任务数据, 工作目录, 锁的键名)}
            
#             # --- 4. 遍历任务，尝试加锁并提交到线程池 ---
#             for task_id, task_data, _ in sorted_tasks:
#                 task_lock_key = f"hamgnn_processing_lock:{task_id}"
                
#                 # a. 幂等性检查: 防止重复处理已成功的任务
#                 processed_key = f"hamgnn_processed:{task_id}"
#                 if redis_client.exists(processed_key):
#                     logger.info(f"HamGNN任务 {task_id} 已被标记为处理过，跳过。")
#                     tasks_already_completed += 1
#                     continue
                
#                 # b. 分布式锁: 确保任务的唯一处理
#                 # `nx=True` (set if not exist) 是原子操作，保证只有一个worker能成功加锁
#                 # `ex=600` 设置10分钟的锁过期时间，防止worker崩溃导致死锁
#                 if not redis_client.set(task_lock_key, "1", nx=True, ex=600):
#                     logger.info(f"HamGNN任务 {task_id} 正在被其他进程处理(已被锁定)，跳过。")
#                     tasks_found_locked += 1
#                     continue
                
#                 # c. 提交任务到线程池
#                 try:
#                     # 在加锁后，再次确认任务是否还在队列中，防止在并发下已被移走
#                     if not redis_client.hexists(QUEUE_HAMGNN_WAITING, task_id):
#                         logger.info(f"任务 {task_id} 在加锁后已不在HamGNN队列中，释放锁并跳过。")
#                         redis_client.delete(task_lock_key)
#                         continue
                    
#                     workdir = task_data.get('workdir')
#                     workflow_params = task_data.get('workflow_params', {})
                    
#                     # 准备请求参数
#                     predict_url = get_server_url("hamgnn") + "/predict"
#                     graph_data_path = os.path.join(workdir, "graph_data.npz")
#                     request_id = f"{task_id}_{int(time.time())}"
#                     job_ticket = {
#                         "request_id": request_id,
#                         "graph_data_path": str(graph_data_path), 
#                         "output_path": workdir, 
#                         "evaluate_loss": workflow_params.get('evaluate_loss', False)
#                     }
                    
#                     # 关键: 先原子性地增加全局并发计数器
#                     redis_client.incr(current_running_key)
                    
#                     # d. 提交任务到线程池，执行`submit_request`函数
#                     future = executor.submit(submit_request, predict_url, job_ticket)
#                     futures[future] = (task_id, task_data, workdir, task_lock_key)
                    
#                 except Exception as e:
#                     # 如果在准备阶段就出错，也必须回滚计数器和锁
#                     redis_client.decr(current_running_key)
#                     redis_client.delete(task_lock_key)
#                     logger.error(f"准备提交任务 {task_id} 到HamGNN时出错: {e}")
#                     handle_hamgnn_task_failure(task_id, task_data, workdir, str(e), "准备提交任务时出错")
            
#             # --- 5. 异步获取已完成任务的结果 ---
#             # `as_completed` 会在任何一个future完成后立即返回它，实现高效处理
#             # 设置一个动态超时，防止线程池永久阻塞
#             timeout = max(600, min(30 * len(futures), 3600))  # 动态超时: 最少10分钟，最多1小时

#             try:
#                 for future in as_completed(futures, timeout=timeout):
#                     task_id, task_data, workdir, task_lock_key = futures[future]
                    
#                     try:
#                         # a. 获取线程执行结果，如果线程中发生异常，.result()会重新抛出
#                         response_data = future.result()
#                         hamiltonian_path = response_data.get('output_file')
                        
#                         if not hamiltonian_path:
#                             raise ValueError("预测结果中未包含哈密顿量文件路径。")
                            
#                         # b. 任务成功处理逻辑
#                         task_data['hamiltonian_path'] = hamiltonian_path
#                         task_data['status'] = 'hamgnn_completed'
                        
#                         # 移动到下一个队列
#                         if _move_task(task_id, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, task_data):
#                             logger.info(f"任务 {task_id} HamGNN预测成功，已移至后处理队列。")
#                             tasks_processed_successfully += 1
#                             # 设置幂等性标记，7天后过期
#                             redis_client.set(f"hamgnn_processed:{task_id}", "1", ex=7*24*60*60)
#                         else:
#                             logger.error(f"任务 {task_id} 成功预测但移动到下一队列失败。")
                            
#                     except Exception as e:
#                         # c. 任务失败处理逻辑
#                         logger.error(f"HamGNN请求执行失败，任务ID: {task_id}, 错误: {e}")
#                         handle_hamgnn_task_failure(task_id, task_data, workdir, str(e), "HamGNN请求执行失败")
                    
#                     finally:
#                         # d. 关键: 无论成功或失败，都必须减少计数器和释放锁
#                         redis_client.decr(current_running_key)
#                         redis_client.delete(task_lock_key)

#             except concurrent.futures.TimeoutError:
#                 # e. 处理线程池整体超时的情况
#                 logger.error(f"线程池等待结果超时({timeout}秒)，可能存在卡死的HamGNN请求。")
#                 for future in [f for f in futures if not f.done()]:
#                     future.cancel() # 尝试取消未完成的线程
#                     task_id, task_data, workdir, task_lock_key = futures[future]
#                     logger.error(f"任务 {task_id} 因超时被取消。")
#                     handle_hamgnn_task_failure(task_id, task_data, workdir, "处理超时", "任务因整体超时被取消")
#                     # 同样需要减少计数器和释放锁
#                     redis_client.decr(current_running_key)
#                     redis_client.delete(task_lock_key)
    
#     except Exception as e:
#         logger.error(f"HamGNN调度任务执行时发生顶层未知错误: {e}", exc_info=True)
#         return f"执行错误: {str(e)}"
    
#     # 返回一个清晰的处理结果摘要
#     return (f"成功处理 {tasks_processed_successfully} 个HamGNN任务; "
#             f"{tasks_found_locked} 个任务被其他进程锁定; "
#             f"{tasks_already_completed} 个任务已被处理过。")


# def handle_hamgnn_task_failure(task_id, task_data, workdir, error_message, status_message):
#     """
#     处理HamGNN任务失败的辅助函数，将失败处理逻辑集中于此。

#     Args:
#         task_id (str): 失败的任务ID。
#         task_data (dict): 任务数据。
#         workdir (str): 工作目录。
#         error_message (str): 简短的错误信息，存入'error'字段。
#         status_message (str): 详细的状态信息，用于日志和FAILURE.json。
#     """
#     task_data['status'] = 'hamgnn_failed'
#     task_data['error'] = error_message
#     task_data['status_log'] = task_data.get('status_log', [])
#     task_data['status_log'].append({
#         'timestamp': time.time(),
#         'status': 'hamgnn_failed',
#         'message': f'HamGNN预测失败: {status_message}'
#     })
    
#     # 在工作目录写入失败信息文件
#     _write_failure_file(
#         workdir=workdir,
#         stage_name='2/4: HamGNN预测',
#         details=f'HamGNN预测失败: {status_message}'
#     )
    
#     # 将失败的任务直接移到最终的完成队列
#     _move_task(task_id, QUEUE_HAMGNN_WAITING, QUEUE_COMPLETED, task_data)