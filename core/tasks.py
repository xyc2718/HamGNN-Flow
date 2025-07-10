# core/tasks.py

import os
import yaml
import json
import random
import logging
import subprocess
import time
from pathlib import Path
from celery import Celery
import requests
import redis
import traceback
from .utils import get_package_path,get_server_url
import yaml
# --- 初始化与配置 ---

# 初始化日志，建议在Celery worker启动时也配置好日志级别
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TASK_CONFIG_PATH = get_package_path('task_basic_config.yaml')

config=yaml.safe_load(open(TASK_CONFIG_PATH, 'r', encoding='utf-8'))
logger.info(f"加载任务配置: {TASK_CONFIG_PATH}")
# 初始化一个全局的Redis客户端，用于我们自定义的状态缓存操作
# decode_responses=True 确保我们从Redis获取的值是字符串而不是字节
try:
    # 假设Redis在本地运行，如果不是，请修改host
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Celery任务模块成功连接到Redis。")
except redis.exceptions.ConnectionError as e:
    logger.error(f"无法连接到Redis，请确保Redis服务正在运行: {e}")
    # 在无法连接到Redis时，抛出异常，阻止worker启动
    raise

# 初始化Celery应用
# 'tasks'是当前模块的名称
# broker是“任务订单栏”的地址，backend是“任务状态显示屏”的地址
celery_app = Celery('tasks',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0')

# --- 定时任务：专职的“信息员”，批量查询Slurm作业状态 ---

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    设置Celery Beat周期性任务。
    这个函数会在Celery worker启动时（如果也启动了beat服务）自动运行。
    """
    # 从配置文件加载执行间隔，而不是作为函数参数
    try:
        # 假设配置文件与tasks.py在同一目录下的configs文件夹中
        with open(TASK_CONFIG_PATH, 'r') as f:
            task_config = yaml.safe_load(f)
        # 从配置中获取轮询间隔，如果找不到则使用一个安全的默认值60秒
        interval = task_config.get('periodic_poller', {}).get('run_interval_seconds', 60.0)
    except (FileNotFoundError, KeyError, TypeError):
        # 如果配置文件或键不存在，使用一个安全的默认值
        interval = 60.0
    
    logging.info(f"设置批量轮询定时任务，执行间隔: {interval} 秒。")
    
    # 使用从配置中加载到的interval来设置定时任务
    sender.add_periodic_task(interval, poll_slurm_jobs.s(), name=f'poll slurm jobs every {interval}s')

@celery_app.task
def poll_slurm_jobs():
    """
    【信息员任务】 - 定时运行
    1. 从Redis获取所有需要监控的作业ID。
    2. 用一次`sacct`调用批量查询它们的状态。
    3. 将最新状态更新回Redis的缓存中。
    这极大地降低了对Slurm控制器的压力。
    """
    # 我们用一个Redis的"Set"数据结构来存储所有独一无二的、正在监控的作业ID
    monitored_jobs = redis_client.smembers('monitored_slurm_jobs')
    if not monitored_jobs:
        return "当前没有需要监控的作业。"

    job_ids_str = ",".join(monitored_jobs)
    logger.info(f"批量查询 {len(monitored_jobs)} 个作业的状态...")
    
    try:
        command = ["sacct", "-j", job_ids_str, "-o", "JobID,State", "-n", "-P"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        pipe = redis_client.pipeline()
        updated_jobs = set()
        
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            try:
                job_id, state = line.strip().split('|')[:2]
                state = state.strip()
                # 更新Redis中的作业状态缓存（一个Hash结构，类似大字典）
                pipe.hset('slurm_job_status_cache', job_id, state)
                # .batch, .extern 等作业步骤也可能被查询到，我们只关心主作业ID
                updated_jobs.add(job_id.split('.')[0])
            except ValueError:
                continue
        
        pipe.execute()

        # 检查哪些作业已经终结，并从相关监控列表中移除
        finished_jobs = {job for job in updated_jobs if redis_client.hget('slurm_job_status_cache', job) in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]}
        if finished_jobs:
            # srem可以安全地尝试删除不存在的成员
            redis_client.srem('monitored_slurm_jobs', *finished_jobs)
            redis_client.srem('running_preprocess_jobs', *finished_jobs) # 同时清理并发控制列表
            logger.info(f"作业 {finished_jobs} 已完成，从所有相关监控列表移除。")

        return f"已更新 {len(updated_jobs)} 个作业的状态。"
    except Exception as e:
        logger.error(f"批量查询Slurm作业时出错: {e}")
        return f"批量查询失败: {e}"


def wait_for_slurm_job(job_id: str, first_time=5,poll_interval: int = 1, timeout: int = 7200) -> bool:
    """通过查询Redis缓存来等待一个Slurm作业完成。"""
    logger.info(f"开始通过Redis缓存监控作业 {job_id}...")
    start_time = time.time()
    
    redis_client.sadd('monitored_slurm_jobs', job_id)
    
    time.sleep(first_time) # 初始等待，给Slurm一些时间来更新状态
    while ((time.time() - start_time) < timeout):
        logger.info(f"正在查询作业 {job_id} 的状态...")
        state = redis_client.hget('slurm_job_status_cache', job_id)
        if state:
            state = state.strip()
            if "COMPLETED" in state:
                logger.info(f"从缓存中检测到作业 {job_id} 成功完成。")
                redis_client.hdel('slurm_job_status_cache', job_id)
                return True
            elif any(fail_state in state for fail_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]):
                logger.error(f"从缓存中检测到作业 {job_id} 失败，状态为: {state}。")
                redis_client.hdel('slurm_job_status_cache', job_id)
                return False
        
        time.sleep(poll_interval)
        
    logger.error(f"等待作业 {job_id} 超时 ({timeout}秒)。")
    redis_client.srem('monitored_slurm_jobs', job_id)
    redis_client.hdel('slurm_job_status_cache', job_id)
    return False

# --- 核心工作流任务 ---

@celery_app.task(
    bind=True,
    autoretry_for=(requests.exceptions.RequestException, ConnectionError),
    retry_kwargs={'max_retries': 1},
    default_retry_delay=10
)
def run_hamgnn_workflow(self,  structure: str,workflow_config: dict = {}):
    """
    【核心业务逻辑】 - 这个函数由Celery Worker在后台执行。
    它负责完整地调度一次从前处理到后处理的流程，并具备并发控制和详细状态报告。
    """
    work_dir = None # 先定义，以防在创建前就出错
    try:
        self.update_state(state='PROGRESS', meta={'stage_code': 'INITIALIZING', 'stage_name': '0/4: 初始化', 'details': '工作流已受理，正在准备环境...'})
        logger.info(f"开始处理结构体: {structure}")
        # --- 阶段一：预处理 ---
        current_stage_name = '1/4: 前处理排队中'
        max_concurrent = config.get('slurm_monitor', {}).get('max_concurrent_preprocess_jobs', 20)
        while redis_client.scard('running_preprocess_jobs') >= max_concurrent:
            wait_message = f"预处理作业并发数已达上限({max_concurrent})，正在等待空闲槽位..."
            self.update_state(state='PROGRESS', meta={'stage_code': 'QUEUED_FOR_PREPROCESSING', 'stage_name': current_stage_name, 'details': wait_message, 'work_dir': str(work_dir)})
            time.sleep(10)
        logger.info(f"当前并发预处理作业数: {redis_client.scard('running_preprocess_jobs')}, 准备提交新的预处理作业。")
        self.update_state(state='PROGRESS', meta={'stage_code': 'PREPROCESSING', 'stage_name': '2/4: 前处理', 'details': '正在提交预处理作业...', 'work_dir': str(work_dir)})


        preprocess_url = get_server_url("openmx") + "/pre_process"
        output_path=workflow_config.get('output_path', None)
        response = requests.post(preprocess_url, json={"structure": str(structure), "graph_para": workflow_config,"output_path":output_path ,"timeout": 120})
        logger.info(f"预处理服务响应: {response.status_code} - {response.json()}")
        response.raise_for_status()
        preprocess_job_id = response.json()['job_id']
        work_dir= response.json()['workdir']
        redis_client.sadd('running_preprocess_jobs', preprocess_job_id)
        current_stage_name = '2/4: 前处理'
        self.update_state(state='PROGRESS', meta={'stage_code': 'PREPROCESSING', 'stage_name': current_stage_name, 'details': f'等待Slurm作业 {preprocess_job_id} 完成...', 'work_dir': str(work_dir), 'slurm_job_id': preprocess_job_id})
        if not wait_for_slurm_job(preprocess_job_id):
            raise RuntimeError(f"预处理作业 {preprocess_job_id} 失败。")
        
        # --- 阶段二：预测 ---
        current_stage_name = '3/4: HamGNN预测'
        self.update_state(state='PROGRESS', meta={'stage_code': 'PREDICTION', 'stage_name': '3/4: HamGNN预测', 'details': '正在调用预测服务...', 'work_dir': str(work_dir)})
        # 这里可以加入我们之前设计的、针对预测服务的智能负载均衡逻辑
        predict_url = get_server_url("hamgnn") + "/predict" 
        graph_data_path = os.path.join(response.json()["workdir"], "graph_data.npz")  # 假设预处理生成了这个文件
        output_path = workflow_config.get('output_path', None)
        evaluate_loss = workflow_config.get('evaluate_loss', False)
        response = requests.post(predict_url, json={"graph_data_path": str(graph_data_path), "output_path": str(output_path), "evaluate_loss": evaluate_loss}, timeout=600)
        response.raise_for_status()
        
        # --- 阶段三：后处理 ---
        current_stage_name = '4/4: 后处理'
        self.update_state(state='PROGRESS', meta={'stage_code': 'POSTPROCESSING', 'stage_name': '4/4: 后处理', 'details': '正在提交能带计算作业...', 'work_dir': str(work_dir)})
        postprocess_url = get_server_url("postprocess") + "/band_cal" 
        hamiltonian_path = response.json().get('output_file', None)
        if not hamiltonian_path:
            raise ValueError("预测结果中未包含哈密顿量文件路径。请检查预测服务的输出。")
        response = requests.post(postprocess_url, json={"hamiltonian_path": str(hamiltonian_path),"graph_data_path":graph_data_path,"output_path":output_path}, timeout=120)
        response.raise_for_status()
        # --- 工作流完成 ---
        final_result = {'stage_code': 'COMPLETED', 'stage_name': '全部完成', 'details': '工作流所有阶段已成功执行完毕。', 'result_dir': str(work_dir)}
        return final_result
    except Exception as e:
        # 【修正后的错误处理逻辑】
        # 任何一步失败，都会被这里捕获
        failure_info = {
            'stage_code': 'FAILED',
            'stage_name': current_stage_name, # 使用局部变量来记录失败时所处的阶段
            'details': str(e),
            'work_dir': str(work_dir)
        }
        self.update_state(state='FAILURE', meta=failure_info)
        # 重新抛出异常，以便在Celery日志或Flower中看到详细的Traceback
        raise
