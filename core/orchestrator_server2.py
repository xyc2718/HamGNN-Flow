# core/orchestrator_server.py
from flask import Flask, request, jsonify, url_for
from waitress import serve
import argparse
from celery.result import AsyncResult
import logging
import socket

# 从tasks.py中导入Celery app和任务函数
from .tasks import celery_app, start_workflow
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
            健康检查API，返回服务器状态。
            主要用于检查服务器是否存活。
            """
            return jsonify({"status": "ok"}), 200
        
        @self.app.route("/submit", methods=['POST'])
        def submit_workflow():
            """
            提交新的工作流任务。
            接收结构文件路径和可选的工作流参数，提交给Celery队列处理。
            """
            if not request.is_json:
                return jsonify({"error": "请求必须是JSON格式"}), 400
            
            data = request.get_json()
            structure_path = data.get('structure')
            workflow_params = data.get('config', {})  # 获取可选的、本次运行的自定义参数

            if not structure_path:
                return jsonify({"error": "请求中必须包含 'structure'"}), 400

            logging.info(f"收到新的工作流请求: {structure_path}")
            
            # 调用轻量级的启动任务
            # task = start_workflow.delay(
            #     structure_file_path=structure_path,
            #     workflow_params=workflow_params
            # )
            task = start_workflow(
                structure_file_path=structure_path,
                workflow_params=workflow_params
            )
            logging.info(f"任务已提交到队列，任务ID: {task.id}")
            
            response_data = {
                "message": "工作流已受理，正在后台排队处理",
                "task_id": task.id,
                "status_url": url_for('get_workflow_status', task_id=task.id, _external=True)
            }
            return jsonify(response_data), 202
        
        @self.app.route("/status/<task_id>", methods=['GET'])
        def get_workflow_status(task_id):
            """
            根据任务ID，从Celery的结果后台(Redis)查询工作流的当前状态。
            """
            task_result = AsyncResult(task_id, app=celery_app)
            
            response_data = {
                "task_id": task_id,
                "state": task_result.state,  # PENDING, PROGRESS, SUCCESS, FAILURE, RETRY, ACCEPTED
                "info": task_result.info     # 我们用 update_state 设置的详细信息
            }

            # 如果任务失败，可以包含更详细的错误信息(traceback)
            if task_result.state == 'FAILURE':
                response_data['traceback'] = str(task_result.traceback)

            return jsonify(response_data)

    def run(self):
        """启动服务器。"""
        host = socket.getfqdn()
        port = find_free_port()
        logging.info(f"调度服务器正在启动，监听地址: http://{host}:{port}")
        info_file_path = get_package_path("server_info/orchestrator_server_info.json")
        write_server_info(host, port, "orchestrator", info_file_path)
        serve(self.app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HamGNN 工作流调度服务器')
    args = parser.parse_args()
    api_server = OrchestratorAPI()
    api_server.run()