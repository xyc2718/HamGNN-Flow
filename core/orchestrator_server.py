# core/orchestrator_server.py
from flask import Flask, request, jsonify, url_for
from waitress import serve
import argparse
from celery.result import AsyncResult
import logging
import socket

# 从tasks.py中导入我们定义的Celery app和任务函数
# 使用相对导入，因为它们在同一个'core'包里
from .tasks import celery_app, run_hamgnn_workflow
import traceback
from .utils import get_package_path,find_free_port,write_server_info
LOGGING_LEVEL= logging.DEBUG
class OrchestratorAPI:
    """
    封装了调度服务器的所有API和运行逻辑。
    它扮演着“前台点餐员”的角色。
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
            健康检查API，返回服务器状态。
            主要用于检查服务器是否存活。
            """
            return jsonify({"status": "ok"}), 200

        @self.app.route("/submit", methods=['POST'])
        def submit_workflow():
            """
            接收用户的计算请求，验证输入后，快速将任务提交到Celery队列，
            并立即返回一个任务ID给用户。
            """
            if not request.is_json:
                return jsonify({"error": "请求必须是JSON格式"}), 400
            
            data = request.get_json()
            structure_path = data.get('structure')
            workflow_config = data.get('config', {})
            if not structure_path:
                return jsonify({"error": "请求中必须包含 'structure'"}), 400

            logging.info(f"收到新的工作流请求，结构文件: {structure_path}")
            
            # 异步调用我们的任务。
            # .delay() 是快捷方式，它会将任务和参数发送到Redis队列中
            task = run_hamgnn_workflow.delay(
                workflow_config=workflow_config,
                structure=structure_path
            )

            logging.info(f"任务已提交到Celery，任务ID: {task.id}")
            
            # 立刻返回202 Accepted状态和任务ID
            # _external=True 会生成完整的URL，方便外部调用
            response_data = {
                "message": "工作流已受理，正在后台处理",
                "task_id": task.id,
                "status_url": url_for('get_workflow_status', task_id=task.id, _external=True)
            }
            return jsonify(response_data), 202

        @self.app.route("/status/<task_id>", methods=['GET'])
        def get_workflow_status(task_id):
            """
            根据任务ID，从Celery的结果后台（Redis）查询工作流的当前状态。
            """
            task_result = AsyncResult(task_id, app=celery_app)
            
            response_data = {
                "task_id": task_id,
                "state": task_result.state, # PENDING, PROGRESS, SUCCESS, FAILURE, RETRY
                "info": task_result.info   # 我们用 update_state 设置的详细信息
            }

            # 如果任务失败，可以包含更详细的错误信息（traceback）
            if task_result.state == 'FAILURE':
                # Celery的结果后台会保存异常信息
                response_data['traceback'] = str(task_result.traceback)

            return jsonify(response_data)

    def run(self):
        """启动服务器。"""
        host = socket.getfqdn()
        port = find_free_port()
        logging.info(f"调度服务器正在启动，监听地址: http://{host}:{port}")
        info_file_path = get_package_path("server_info/orchestrator_server_info.json")
        write_server_info(host, port, "orchestrator", info_file_path )
        serve(self.app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HamGNN 工作流调度服务器 (面向对象版)')
    
    args = parser.parse_args()
    
    api_server = OrchestratorAPI()
    api_server.run()