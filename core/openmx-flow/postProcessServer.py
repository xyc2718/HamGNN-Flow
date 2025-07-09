import yaml
import logging
import time
import random
import subprocess
from pathlib import Path
import os
import json
import flask
import socket
import traceback
import warnings
import numpy as np
from flask import Flask, request, jsonify
from waitress import serve
from ..utils import write_server_info, find_free_port, get_package_path
from .utils_openmx.band_cal import band_cal
from ..communication import PostProcessCommunicator as Communicator, BaseCommunicator
import argparse
LOGGING_LEVEL= logging.DEBUG
POSTPROCESS_CONFIG_PATH = get_package_path("openmx-flow/postprocess_basic_config.yaml")
class PostProcessServer:
    def __init__(self, config_path=None):
        """
        对预测结果进行后处理的服务器。
        """
        self.app = Flask(__name__)
        self.app.logger.setLevel(LOGGING_LEVEL)
        self.app.logger.info("正在初始化 OpenMX 服务器...")
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB
        self.communicator: BaseCommunicator = Communicator()
        self.load_basic_config()
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB
        self._register_routes()
        self.type = "PostProcessServer"  # 服务器类型标识
        self.app.logger.info(f"PostProcess服务器配置已加载: {self.config}")
        self.app.logger.info(f"默认参数: {self.default_params}")

    def load_basic_config(self):
        """加载PostProcess配置文件。"""
        if not POSTPROCESS_CONFIG_PATH.exists():
            raise FileNotFoundError(f"PostProcess配置文件不存在: {POSTPROCESS_CONFIG_PATH}")
        self.app.logger.debug(f"PostProcess配置文件路径: {POSTPROCESS_CONFIG_PATH}")
        with open(POSTPROCESS_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config
        self.config_path= POSTPROCESS_CONFIG_PATH
        self.default_params = config.get("default_input_parameters", {})
        self.app.logger.info(f"get default_params: {self.default_params}")
        self.process_config = self.default_params.copy()
    def set_workdir(self, workdir=None):
        self.workdir = str(workdir) if workdir else str(get_package_path(""))
        self.process_config["save_dir"] = str(self.workdir)
    def set_process_config(self, process_config={}):
        """设置后处理配置。"""
        self.process_config.update(process_config)
        self.app.logger.debug(f"设置后处理配置为: {self.process_config}")    
    def _register_routes(self):
        """注册Flask路由，并将它们连接到类实例。"""
    
        @self.app.route("/health", methods=['GET'])
        def health_check():
            # 检查服务器是否存活以及模型是否已加载
            return jsonify({"status": "ok"})
        
        @self.app.route("/band_cal", methods=['POST'])
        def band_calculate():
            try:
                hamiltonian_path, graph_data_path, band_para, current_dir = self.communicator.unpack_request(request)
                if current_dir:
                    workdir = os.path.dirname(hamiltonian_path)
                else:
                    current_path = get_package_path("")
                    t = int(time.time())
                    workdir_name = os.path.join(current_path,"temp",f"band_{Path(hamiltonian_path).stem}_{t}{random.randint(100, 999)}")
                    try:
                        os.makedirs(workdir_name)
                    except FileExistsError:
                        logging.warning(f"工作目录已存在: {workdir_name}")
                self.set_workdir(workdir)
                self.app.logger.debug(f"接收到的输入: {input}")
                self.app.logger.debug(f"使用参数: {self.process_config}")
                self.app.logger.debug(f"工作目录: {self.workdir}")
                self.process_config["hamiltonian_path"] = hamiltonian_path
                self.process_config["graph_data_path"] = graph_data_path
                self.set_process_config(band_para)
                band_cal(self.process_config)
                self.app.logger.debug(f"process_config: {self.process_config}")
                self.app.logger.info(f"能带计算完成，结果保存在: {self.workdir}")
                return self.communicator.pack_response({"status": "success",  "workdir": str(self.workdir), "process_config": self.process_config})
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500
            
        @self.app.route("/api", methods=['GET'])
        def api():
            api_format= {
                "band_cal": {
                    "method": "POST",
                    "description": "执行能带计算的后处理操作。",
                    "parameters": {
                        "hamiltonian_path": "哈密顿量文件路径",
                        "graph_data_path": "图数据文件路径",
                        "band_para": "能带计算参数",
                        "current_dir": "是否使用当前目录（默认为True）"
                    },
                    "responses": {
                        "200": {
                            "description": "成功响应，包含工作目录和处理配置。",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "status": "success",
                                        "workdir": "/path/to/workdir",
                                        "process_config": {}
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "服务器内部错误"
                        }
                    }
                }
            }
            return jsonify(api_format)

        

    def run(self):
        """
        启动服务器，包括HPC的服务发现功能。
        这个方法对应于服务器的“运行”阶段。
        """

        info_file_path = get_package_path("server_info/postprocess_server_info.json")
        host = socket.getfqdn()
        port = find_free_port()
        self.app.logger.info(f"正在启动 Flask 服务器，地址: http://{host}:{port}")
        write_server_info(host, port, self.type,info_file_path)
        self.app.logger.debug(f"服务器信息已写入: {info_file_path}")
        # 使用生产级的Waitress服务器来运行应用
        serve(self.app, host="0.0.0.0", port=port)
        

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='OpenMX Server')
    argument_parser.add_argument('--config', default=PostProcessServer, type=str, help='OpenMX配置文件路径')
    args = argument_parser.parse_args()
    openmx_server = PostProcessServer(config_path=args.config)
    openmx_server.run()