# core/orchestrator_server.py
from flask import Flask, request, jsonify, url_for
from waitress import serve
import argparse
from celery.result import AsyncResult
import logging
import socket
import json

# 从tasks.py中导入我们定义的Celery app和任务函数
# 使用相对导入,因为它们在同一个'core'包里
from .tasks import celery_app, start_workflow, redis_client
from .tasks import QUEUE_PENDING, QUEUE_OPENMX_WAITING, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, QUEUE_COMPLETED
import traceback
from .utils import get_package_path, find_free_port, write_server_info

LOGGING_LEVEL = logging.INFO

class OrchestratorAPI:
    """
    封装了调度服务器的所有API和运行逻辑。
    它扮演着"前台点餐员"的角色。
    """
    def __init__(self):
        """初始化调度服务器API。"""
        self.app = Flask(__name__)
        self.app.logger.setLevel(LOGGING_LEVEL)
        self._register_routes()

    def _register_routes(self):
        """注册所有API端点。"""
        
        @self.app.route("/health", methods=['GET'])
        def health_check():
            """
            健康检查API,返回服务器状态。
            主要用于检查服务器是否存活。
            """
            return jsonify({"status": "ok"}), 200

        @self.app.route("/submit", methods=['POST'])
        def submit_workflow():
            """
            提交一个新的工作流处理请求。
            请求体应包含:
            - structure_file_path: 结构文件路径
            - workflow_params: (可选)工作流参数
            """
            if not request.is_json:
                return jsonify({"error": "请求必须是JSON格式"}), 400
            
            data = request.get_json()
            structure_path = data.get('structure')
            workflow_params = data.get('config', {}) # 获取可选的、本次运行的自定义参数

            if not structure_path:
                return jsonify({"error": "请求中必须包含 'structure'"}), 400

            logging.info(f"收到新的工作流请求: {structure_path}")
            
            # 调用轻量级的启动任务,仅负责将任务放入待处理队列
            task = start_workflow.delay(
                structure_file_path=structure_path,
                workflow_params=workflow_params
            )
            logging.info(f"任务已提交到队列,任务ID: {task.id}")
            
            response_data = {
                "message": "工作流已受理,正在后台排队处理",
                "task_id": task.id,
                "status_url": url_for('get_workflow_status_route', task_id=task.id, _external=True)
            }
            return jsonify(response_data), 202

        @self.app.route("/status/<task_id>", methods=['GET'])
        def get_workflow_status_route(task_id):
            """
            根据任务ID,查询工作流的当前状态。
            会从两个来源获取信息:
            1. Celery的结果后台(任务本身的状态)
            2. Redis队列(任务在哪个处理阶段)
            """
            # 从Celery结果后台获取任务状态
            task_result = AsyncResult(task_id, app=celery_app)
            
            # 初始化响应数据
            response_data = {
                "task_id": task_id,
                "state": task_result.state, # PENDING, PROGRESS, SUCCESS, FAILURE, RETRY
                "info": task_result.info   # 我们用 update_state 设置的详细信息
            }

            # 如果任务失败,可以包含更详细的错误信息(traceback)
            if task_result.state == 'FAILURE':
                response_data['traceback'] = str(task_result.traceback)
            
            # 增强状态信息: 检查任务在哪个队列中
            for queue_name in [QUEUE_PENDING, QUEUE_OPENMX_WAITING, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, QUEUE_COMPLETED]:
                task_data_json = redis_client.hget(queue_name, task_id)
                if task_data_json:
                    task_data = json.loads(task_data_json)
                    response_data['queue'] = queue_name
                    response_data['queue_status'] = task_data.get('status', 'unknown')
                    response_data['details'] = task_data
                    break
            
            return jsonify(response_data)
            
        @self.app.route("/queue_stats", methods=['GET'])
        def get_queue_stats():
            """
            获取所有队列的统计信息,包括每个队列中的任务数量。
            """
            stats = {}
            
            # 获取每个队列的任务数量
            for queue_name in [QUEUE_PENDING, QUEUE_OPENMX_WAITING, QUEUE_HAMGNN_WAITING, QUEUE_POST_WAITING, QUEUE_COMPLETED]:
                stats[queue_name] = redis_client.hlen(queue_name)
            
            # 获取各种作业的当前运行数量
            stats['running_openmx_jobs'] = redis_client.scard('running_preprocess_jobs')
            stats['running_hamgnn_jobs'] = int(redis_client.get('running_hamgnn_jobs') or 0)
            stats['running_postprocess_jobs'] = int(redis_client.get('running_postprocess_jobs') or 0)
            
            return jsonify(stats)

    def run(self, host=None, port=None):
        """启动服务器。"""
        host = host or socket.getfqdn()
        port = port or find_free_port()
        logging.info(f"调度服务器正在启动,监听地址: http://{host}:{port}")
        info_file_path = get_package_path("server_info/orchestrator_server_info.json")
        write_server_info(host, port, "orchestrator", info_file_path)
        serve(self.app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HamGNN 工作流调度服务器')
    parser.add_argument('--host', help='服务器主机名')
    parser.add_argument('--port', type=int, help='服务器端口号')
    
    args = parser.parse_args()
    
    api_server = OrchestratorAPI()
    api_server.run(host=args.host, port=args.port)