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
TIMEOUT=180

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
        self.update_state(
            state='FAILURE',
            meta={
                'stage_code': 'FAILED',
                'stage_name': '0/4: 初始化', 
                'details': f'创建任务失败: {str(e)}'
            }
        )
        raise

# --- 四个独立的定时任务函数 ---
@celery_app.task
def dispatch_openmx_tasks():
    """
    F1定时任务: 检查openmx并发限制,将待处理队列的任务转移到openmx等待队列并提交给openmxServer
    """
    # 从配置中获取最大并发数
    max_concurrent = config.get('concurrency', {}).get('max_openmx_jobs', 10)
    
    # 检查当前运行的预处理作业数
    current_running = redis_client.scard('running_preprocess_jobs')
    
    # 计算可提交的任务数量
    slots_available = max(0, max_concurrent - current_running)
    
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
        task_data = json.loads(task_data_json)
        sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
    
    sorted_tasks.sort(key=lambda x: x[2])  # 按创建时间排序
    
    # 处理可提交的任务
    tasks_processed = 0
    for task_id, task_data, _ in sorted_tasks:
        if tasks_processed >= slots_available:
            break
            
        try:
            
            # 提取任务参数
            structure_file_path = task_data.get('structure_file_path')
            workflow_params = task_data.get('workflow_params', {})
            logger.info(f"work_para:{workflow_params}")
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
                    "graph_para": workflow_params,
                    "output_path": output_path,
                    "timeout": 120
                }
            )
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            preprocess_job_id = response_data['job_id']
            workdir = response_data['workdir']
            
            # 更新任务数据
            task_data['openmx_job_id'] = preprocess_job_id
            task_data['workdir'] = workdir
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
            # 将失败的任务直接移到完成队列
            _move_task(task_id, QUEUE_PENDING, QUEUE_COMPLETED, task_data)
    
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
    logger.info(f"轮询Slurm作业状态结果: {poll_result}")
    
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
                failure_info = {
                    'stage_code': 'FAILED',
                    'stage_name': '4/4: 后处理', 
                    'details': task_data['error'],
                    'workdir': str(task_data.get('workdir'))
                }
                with open(os.path.join(task_data.get('workdir'), 'FAILURE.json'), 'w', encoding='utf-8') as f:
                    json.dump(failure_info, f, ensure_ascii=False, indent=4)
            except Exception as file_error:
                logger.error(f"写入失败信息到 {task_data.get('workdir')} 时出错: {file_error}")
            
            # 将失败的任务直接移到完成队列
            _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
            
    return tasks_processed

@celery_app.task
def dispatch_hamgnn_tasks():
    """
    F3定时任务: 将hamgnn等待队列的任务提交给hamgnnServer,完成后转移到后处理等待队列
    此版本允许多个进程并行处理不同的任务，每个进程只处理未被锁定的任务
    """
    try:
        # 从配置中获取最大并发数
        max_concurrent = config.get('concurrency', {}).get('max_hamgnn_jobs', 5)
        
        # 创建一个Redis键来跟踪当前运行的HamGNN作业数
        current_running_key = 'running_hamgnn_jobs'
        
        # 使用Redis的原子操作初始化计数器(如果不存在)
        redis_client.setnx(current_running_key, 0)
        
        # 获取当前运行的HamGNN作业数
        current_running = int(redis_client.get(current_running_key) or 0)
        
        # 计算可提交的任务数量
        slots_available = max(0, max_concurrent - current_running)
        
        if slots_available <= 0:
            logger.info(f"HamGNN并发数已达上限({max_concurrent}),当前运行: {current_running}")
            return f"HamGNN并发数已达上限({max_concurrent}),跳过调度"
        
        # 获取hamgnn等待队列中的所有任务
        hamgnn_tasks = redis_client.hgetall(QUEUE_HAMGNN_WAITING)
        
        if not hamgnn_tasks:
            return "没有等待中的HamGNN任务"
        
        # 按创建时间排序(先进先出)
        sorted_tasks = []
        for task_id, task_data_json in hamgnn_tasks.items():
            try:
                task_data = json.loads(task_data_json)
                sorted_tasks.append((task_id, task_data, task_data.get('created_at', 0)))
            except json.JSONDecodeError:
                logger.error(f"任务 {task_id} 的数据格式无效，跳过")
                continue
        
        sorted_tasks.sort(key=lambda x: x[2])  # 按创建时间排序
        
        # 存储成功处理的任务数量
        tasks_processed = 0
        tasks_locked = 0
        tasks_already_processed = 0
        
        # 创建线程池 - 只使用当前可用的槽位数量的线程
        with ThreadPoolExecutor(max_workers=min(slots_available, len(sorted_tasks))) as executor:
            futures = {}  # 使用字典存储future -> task_info的映射
            task_locks = []  # 存储已获取的任务锁，便于后续释放
            
            # 尝试锁定并提交任务 - 注意这里不限制处理的任务数量为slots_available
            # 而是尝试锁定所有任务，只有成功锁定的才会被提交，从而实现细粒度控制
            for task_id, task_data, _ in sorted_tasks:
                # 如果已经提交的任务达到可用槽位，则不再继续尝试
                if len(futures) >= slots_available:
                    break
                    
                # 为每个任务创建一个锁，确保同一任务不会被并发处理
                task_lock_key = f"hamgnn_processing_lock:{task_id}"
                
                # 检查任务是否已处理过(幂等性检查)
                processed_key = f"hamgnn_processed:{task_id}"
                if redis_client.exists(processed_key):
                    logger.info(f"HamGNN任务 {task_id} 已处理过，跳过")
                    tasks_already_processed += 1
                    continue
                
                # 尝试获取锁，如果已被锁定则跳过 - 这里是细粒度控制的关键
                # nx=True确保只有一个进程能获取锁，其他进程会立即返回False而不是阻塞
                if not redis_client.set(task_lock_key, "1", nx=True, ex=600):  # 10分钟锁
                    logger.info(f"HamGNN任务 {task_id} 正在被其他进程处理，跳过")
                    tasks_locked += 1
                    continue
                
                # 记录已获取的锁
                task_locks.append(task_lock_key)
                
                try:
                    # 再次检查任务是否在队列中(可能在获取锁的过程中被其他进程移除)
                    if not redis_client.hexists(QUEUE_HAMGNN_WAITING, task_id):
                        logger.info(f"任务 {task_id} 不在HamGNN队列中，可能已被处理")
                        redis_client.delete(task_lock_key)  # 释放锁
                        task_locks.remove(task_lock_key)
                        continue
                    
                    # 提取任务参数
                    workdir = task_data.get('workdir')
                    workflow_params = task_data.get('workflow_params', {})
                    
                    # 构建预测URL和请求参数
                    predict_url = get_server_url("hamgnn") + "/predict"
                    graph_data_path = os.path.join(workdir, "graph_data.npz")
                    output_path = workdir
                    logger.info(f"任务 {task_id} 的图数据路径: {graph_data_path}, 输出路径: {output_path}")
                    evaluate_loss = workflow_params.get('evaluate_loss', False)
                    
                    # 更新任务状态
                    task_data['status'] = 'submitting_to_hamgnn'
                    task_data['status_log'] = task_data.get('status_log', [])
                    task_data['status_log'].append({
                        'timestamp': time.time(),
                        'status': 'submitting_to_hamgnn',
                        'message': '正在提交给HamGNN预测服务',
                        'worker_id': os.getpid()  # 记录处理该任务的worker ID
                    })
                    
                    # 更新Redis中的任务数据
                    redis_client.hset(QUEUE_HAMGNN_WAITING, task_id, json.dumps(task_data))
                    
                    # 创建请求参数(添加请求ID确保幂等性)
                    request_id = f"{task_id}_{int(time.time())}"
                    job_ticket = {
                        "request_id": request_id,
                        "graph_data_path": str(graph_data_path), 
                        "output_path": output_path, 
                        "evaluate_loss": evaluate_loss
                    }
                    
                    # 提交给HamGNN服务器
                    logger.info(f"提交任务 {task_id} 到HamGNN服务器: {predict_url}")
                    
                    # 增加运行计数(使用原子操作)
                    redis_client.incr(current_running_key)
                    
                    # 将请求提交到线程池
                    future = executor.submit(submit_request, predict_url, job_ticket)
                    futures[future] = (task_id, task_data, workdir, task_lock_key)
                    
                except Exception as e:
                    # 释放锁
                    redis_client.delete(task_lock_key)
                    if task_lock_key in task_locks:
                        task_locks.remove(task_lock_key)
                    
                    # 减少运行计数
                    redis_client.decr(current_running_key)
                    
                    logger.error(f"准备提交任务 {task_id} 到HamGNN服务器时出错: {e}")
                    handle_hamgnn_task_failure(task_id, task_data, workdir, str(e), "准备提交到HamGNN服务器时出错")
            
            # 处理完成的任务，按照完成顺序处理(而非提交顺序)
            # 设置超时，防止线程池阻塞
            timeout = max(600, min(30 * len(futures), 3600))  # 最少10分钟，最多1小时
            
            try:
                for future in as_completed(futures, timeout=timeout):
                    task_id, task_data, workdir, task_lock_key = futures[future]
                    
                    try:
                        # 获取请求结果
                        response_data = future.result()
                        hamiltonian_path = response_data.get('output_file', None)
                        
                        if not hamiltonian_path:
                            raise ValueError("预测结果中未包含哈密顿量文件路径。请检查预测服务的输出。")
                            
                        # 更新任务数据
                        task_data['hamiltonian_path'] = hamiltonian_path
                        task_data['status'] = 'hamgnn_completed'
                        task_data['status_log'].append({
                            'timestamp': time.time(),
                            'status': 'hamgnn_completed',
                            'message': 'HamGNN预测已完成'
                        })
                        
                        # 将任务从hamgnn等待队列移动到后处理等待队列
                        if _move_task(task_id, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, task_data):
                            logger.info(f"任务 {task_id} 的HamGNN预测已完成,已移至后处理等待队列")
                            tasks_processed += 1
                            
                            # 标记任务为已处理(用于幂等性检查)
                            # 设置较长的过期时间，比如7天
                            redis_client.set(f"hamgnn_processed:{task_id}", "1", ex=7*24*60*60)
                        else:
                            logger.error(f"移动任务 {task_id} 到后处理等待队列失败")
                            
                    except Exception as e:
                        # 处理请求失败
                        logger.error(f"HamGNN请求执行失败: {e}")
                        handle_hamgnn_task_failure(task_id, task_data, workdir, str(e), "HamGNN请求执行失败")
                    
                    finally:
                        # 无论成功还是失败，都要减少运行计数和释放锁
                        redis_client.decr(current_running_key)
                        redis_client.delete(task_lock_key)
                        if task_lock_key in task_locks:
                            task_locks.remove(task_lock_key)
            except concurrent.futures.TimeoutError:
                # 处理整体超时情况
                logger.error(f"线程池执行超时，可能有HamGNN任务未完成")
                # 取消所有未完成的任务
                for future in [f for f in futures if not f.done()]:
                    future.cancel()
                    task_id, task_data, workdir, task_lock_key = futures[future]
                    logger.error(f"任务 {task_id} 在超时时间内未完成，已取消")
                    handle_hamgnn_task_failure(task_id, task_data, workdir, "处理超时", "任务在超时时间内未完成，已取消")
                    # 减少计数器和释放锁
                    redis_client.decr(current_running_key)
                    redis_client.delete(task_lock_key)
    
    except Exception as e:
        logger.error(f"HamGNN调度任务执行时发生未知错误: {e}")
        return f"执行错误: {str(e)}"
    
    return f"处理了 {tasks_processed} 个HamGNN任务，{tasks_locked} 个任务正在被其他进程处理，{tasks_already_processed} 个任务已处理过"

def handle_hamgnn_task_failure(task_id, task_data, workdir, error_message, status_message):
    """处理HamGNN任务失败的辅助函数，集中处理失败逻辑"""
    # 更新任务状态为失败
    task_data['status'] = 'hamgnn_failed'
    task_data['error'] = error_message
    task_data['status_log'] = task_data.get('status_log', [])
    task_data['status_log'].append({
        'timestamp': time.time(),
        'status': 'hamgnn_failed',
        'message': f'HamGNN预测失败: {status_message}'
    })
    
    # 将失败的任务直接移到完成队列
    _move_task(task_id, QUEUE_HAMGNN_WAITING, QUEUE_COMPLETED, task_data)

@celery_app.task
def dispatch_postprocess_tasks():
    """
    F4定时任务: 将后处理等待队列的任务提交给postprocessServer,得到job_id后留在队列中等待作业完成
    与OpenMX处理方式相似，只负责提交任务，完成状态的检查由poll_slurm_and_dispatch处理
    """
    try:
        # 从配置中获取最大并发数
        max_concurrent = config.get('concurrency', {}).get('max_postprocess_jobs', 10)
        
        # 检查当前运行的后处理作业数
        current_running = redis_client.scard('running_postprocess_jobs')
        
        # 计算可提交的任务数量
        slots_available = max(0, max_concurrent - current_running)
        
        if slots_available <= 0:
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
                continue
        
        sorted_tasks.sort(key=lambda x: x[2])  # 按创建时间排序
        
        # 处理可提交的任务
        tasks_processed = 0
        for task_id, task_data, _ in sorted_tasks:
            if tasks_processed >= slots_available:
                break
                
            try:
                # 为每个任务创建一个锁，确保同一任务不会被并发处理
                task_lock_key = f"processing_lock:{task_id}"
                
                # 尝试获取锁，如果已被锁定则跳过
                if not redis_client.set(task_lock_key, "1", nx=True, ex=600):  # 10分钟锁
                    logger.info(f"任务 {task_id} 正在被其他进程处理，跳过")
                    continue
                
                try:
                    # 再次检查任务是否在队列中(可能在获取锁的过程中被其他进程移除)
                    if not redis_client.hexists(QUEUE_POST_WAITING, task_id):
                        logger.info(f"任务 {task_id} 不在后处理队列中，可能已被处理")
                        redis_client.delete(task_lock_key)  # 释放锁
                        continue
                    
                    # 提取任务参数
                    hamiltonian_path = task_data.get('hamiltonian_path')
                    workdir = task_data.get('workdir')
                    workflow_params = task_data.get('workflow_params', {})
                    output_path = workdir
                    
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
                        'worker_id': os.getpid()  # 记录处理该任务的worker ID
                    })
                    
                    # 创建请求参数(添加请求ID确保幂等性)
                    request_id = f"{task_id}_{int(time.time())}"
                    job_ticket = {
                        "request_id": request_id,
                        "hamiltonian_path": str(hamiltonian_path),
                        "graph_data_path": str(graph_data_path),
                        "band_para": workflow_params,
                        "output_path": output_path
                    }
                    
                    # 提交给后处理服务器
                    logger.info(f"提交任务 {task_id} 到后处理服务器: {postprocess_url}")
                    response = requests.post(postprocess_url, json=job_ticket, timeout=TIMEOUT)
                    response.raise_for_status()
                    
                    # 解析响应
                    response_data = response.json()
                    postprocess_job_id = response_data['job_id']  # 假设后处理服务器返回job_id
                    
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
                    
                    # 更新队列中的任务数据(注意不移动队列)
                    redis_client.hset(QUEUE_POST_WAITING, task_id, json.dumps(task_data))
                    
                    logger.info(f"任务 {task_id} 已提交给后处理服务器,Slurm作业ID: {postprocess_job_id}")
                    tasks_processed += 1
                    
                finally:
                    # 释放锁
                    redis_client.delete(task_lock_key)
                    
            except Exception as e:
                logger.error(f"提交任务 {task_id} 到后处理服务器时出错: {e}")
                # 更新任务状态为失败
                task_data['status'] = 'postprocess_failed'
                task_data['error'] = str(e)
                task_data['status_log'].append({
                    'timestamp': time.time(),
                    'status': 'postprocess_failed',
                    'message': f'提交到后处理服务器失败: {str(e)}'
                })
                # 写入失败标记文件
                try:
                    failure_info = {
                        'stage_code': 'FAILED',
                        'stage_name': '4/4: 后处理', 
                        'details': str(e),
                        'workdir': str(workdir)
                    }
                    with open(os.path.join(workdir, 'FAILURE.json'), 'w', encoding='utf-8') as f:
                        json.dump(failure_info, f, ensure_ascii=False, indent=4)
                except Exception as file_error:
                    logger.error(f"写入失败信息到 {workdir} 时出错: {file_error}")
                
                # 将失败的任务直接移到完成队列
                _move_task(task_id, QUEUE_POST_WAITING, QUEUE_COMPLETED, task_data)
        
        return f"处理了 {tasks_processed} 个后处理任务"
    
    except Exception as e:
        logger.error(f"后处理调度任务执行时发生未知错误: {e}")
        return f"执行错误: {str(e)}"

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
        ("postprocess", "processing_lock:", QUEUE_POST_WAITING)
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
    dispatch_hamgnn_interval = periodic_tasks_config.get('dispatch_hamgnn_interval', 5.0)
    logger.info(f"设置HamGNN调度定时任务,执行间隔: {dispatch_hamgnn_interval} 秒。")
    sender.add_periodic_task(
        dispatch_hamgnn_interval, 
        dispatch_hamgnn_tasks.s(), 
        name=f'dispatch HamGNN tasks every {dispatch_hamgnn_interval}s'
    )
    
    # F4: 调度后处理任务
    dispatch_postprocess_interval = periodic_tasks_config.get('dispatch_postprocess_interval', 5.0)
    logger.info(f"设置后处理调度定时任务,执行间隔: {dispatch_postprocess_interval} 秒。")
    sender.add_periodic_task(
        dispatch_postprocess_interval, 
        dispatch_postprocess_tasks.s(), 
        name=f'dispatch postprocess tasks every {dispatch_postprocess_interval}s'
    )
    
    # 定期清理任务
    sender.add_periodic_task(
        300.0,  # 5分钟
        cleanup_stale_locks_and_counters.s(),
        name='cleanup stale resources every 5min'
    )
    
    # 恢复卡住任务
    sender.add_periodic_task(
        300.0,  # 5分钟
        recover_stuck_tasks.s(),
        name='recover stuck tasks every 5min'
    )