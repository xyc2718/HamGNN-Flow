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
from functools import wraps
import argparse
import threading
import torch
LOGGING_LEVEL= logging.INFO
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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.app.logger.info(f"使用的计算设备: {self.device}")

        self.active_requests = 0
        self.lock = threading.Lock() # 线程锁，确保计数器在多线程环境下是安全的

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
        self.conda_source = config.get("conda_source")
        self.conda_env = config.get("conda_env")
        if not self.conda_source or not self.conda_env:
            raise ValueError("Conda环境配置不完整，请检查配置文件。")
    # def set_workdir(self, workdir=None):
    #     self.workdir = str(workdir) if workdir else str(get_package_path(""))
    #     self.process_config["save_dir"] = str(self.workdir)
    def set_process_config(self, process_config={}):
        """设置后处理配置。"""
        final_config = self.default_params.copy()
        final_config.update(process_config)
        self.app.logger.debug(f"设置后处理配置为: {final_config}")
        return final_config

    def track_load(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # --- 请求开始前 ---
            with self.lock:
                self.active_requests += 1
            logging.info(f"接收到新请求，当前活跃请求数: {self.active_requests}")
            # --- 执行原始的路由函数 ---
            try:
                result = f(*args, **kwargs)
            finally:
                # --- 请求结束后（无论成功或失败） ---
                with self.lock:
                    self.active_requests -= 1
                logging.info(f"请求处理完毕，当前活跃请求数: {self.active_requests}")
            return result
        return decorated_function
    
    def _submit_job(self, script_path: str) -> str:
        """辅助方法：提交一个sbatch脚本并返回作业ID。"""
        command = ["sbatch", script_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        self.app.logger.info(f"作业提交成功，Job ID: {job_id}")
        return job_id
    def run_band_cal_submit(self,process_config, workdir):
        config_path=os.path.join(workdir,"band_cal.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(process_config, f, default_flow_style=False, allow_unicode=True, 
              encoding='utf-8')

        sbatch_script = f"""#!/bin/sh
#SBATCH --job-name=band_cal               # Job name
#SBATCH --partition={process_config.get("partition", "chu")}                     # Partition (queue) name [xiang;yang;xu;chu]; adjust as needed
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                         # Total MPI tasks ntasks*cpu-per-task <= [64-xiang;64-yang;96-xu;96-chu]
#SBATCH --cpus-per-task={process_config.get("ncpus", 4)}                 # CPUs per task (OpenMP threads)
#SBATCH --time=48:00:00                   # Wall‐time limit (HH:MM:SS)
#SBATCH --output={os.path.join(workdir,"band_cal.%j.out")}              # STDOUT file
#SBATCH --error={os.path.join(workdir,"band_cal.%j.err")}               # STDERR file
set -euo pipefail
module purge
source {self.conda_source}
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
conda run -n {self.conda_env} python {get_package_path("openmx-flow/utils_openmx/band_cal.py")} \\
        --input {config_path}
        """

        with open(os.path.join(workdir, "run_band_cal.sh"), 'w') as f:
            f.write(sbatch_script)
        job_id = self._submit_job(os.path.join(workdir, "run_band_cal.sh"))
        self.app.logger.info(f"OpenMX计算作业已提交，Job ID: {job_id}")
        return job_id

       
    def _register_routes(self):
        """注册Flask路由，并将它们连接到类实例。"""
    
        @self.app.route("/health", methods=['GET'])
        def health_check():
            # 检查服务器是否存活以及模型是否已加载
            return jsonify({"status": "ok"})
        
        @self.app.route("/load_status", methods=['GET'])
        def load_status():
            return jsonify({
                "active_requests": self.active_requests,
            })
        
        
        @self.app.route("/band_cal_local", methods=['POST'])
        @self.track_load
        def band_calculate_local():
            start_time = time.time()
            workdir = None
            try:
                hamiltonian_path, graph_data_path, band_para, output_path = self.communicator.unpack_request(request)
                if output_path is not None and output_path != "./":
                    workdir = output_path
                    try:
                        os.makedirs(workdir)
                    except FileExistsError:
                        logging.warning(f"工作目录已存在: {workdir}")
                elif output_path == "./":
                    workdir = os.path.dirname(hamiltonian_path)
                else:
                    current_path = get_package_path("")
                    t = int(time.time())
                    workdir= os.path.join(current_path,"temp",f"band_{Path(hamiltonian_path).stem}_{t}{random.randint(100, 999)}")
                    try:
                        os.makedirs(workdir)
                    except FileExistsError:
                        logging.warning(f"工作目录已存在: {workdir}")

                process_config= self.set_process_config(band_para)
                self.app.logger.debug(f"接收到的输入: {hamiltonian_path}")
                self.app.logger.debug(f"使用参数: {process_config}")
                self.app.logger.debug(f"工作目录: {workdir}")
                process_config["hamiltonian_path"] = hamiltonian_path
                process_config["graph_data_path"] = graph_data_path
                process_config["save_dir"] = workdir
                band_cal(process_config)
                self.app.logger.debug(f"process_config: {process_config}")
                self.app.logger.info(f"能带计算完成，结果保存在: {workdir}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.app.logger.info(f"能带计算耗时: {elapsed_time:.2f}秒")
                return self.communicator.pack_response({"status": "success",  "workdir": str(workdir), "process_config": process_config})
            except Exception as e:
                warnings.warn(f"能带计算发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500
            
        @self.app.route("/band_cal", methods=['POST'])
        @self.track_load
        def band_calculate():
            workdir = None
            try:
                hamiltonian_path, graph_data_path, band_para, output_path = self.communicator.unpack_request(request)
                if output_path is not None and output_path != "./":
                    workdir = output_path
                    try:
                        os.makedirs(workdir)
                    except FileExistsError:
                        logging.warning(f"工作目录已存在: {workdir}")
                elif output_path == "./":
                    workdir = os.path.dirname(hamiltonian_path)
                else:
                    current_path = get_package_path("")
                    t = int(time.time())
                    workdir= os.path.join(current_path,"temp",f"band_{Path(hamiltonian_path).stem}_{t}{random.randint(100, 999)}")
                    try:
                        os.makedirs(workdir)
                    except FileExistsError:
                        logging.warning(f"工作目录已存在: {workdir}")

                process_config= self.set_process_config(band_para)
                self.app.logger.debug(f"接收到的输入: {hamiltonian_path}")
                self.app.logger.debug(f"使用参数: {process_config}")
                self.app.logger.debug(f"工作目录: {workdir}")
                process_config["hamiltonian_path"] = hamiltonian_path
                process_config["graph_data_path"] = graph_data_path
                process_config["save_dir"] = workdir
                job_id=self.run_band_cal_submit(process_config, workdir)
                self.app.logger.info(f"能带计算任务已提交，任务ID: {job_id}")
                return self.communicator.pack_response({"status": "success", "job_id": job_id, "workdir": str(workdir), "process_config": process_config})
            except Exception as e:
                warnings.warn(f"能带计算发生错误: {e}")
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
                        "output_path": "输出路径（可选）",
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

        

    def run(self,num_threads=4):
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
        serve(self.app, host="0.0.0.0", port=port,threads=num_threads)

# server = PostProcessServer()
# app = server.app

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description='OpenMX Server')
    argument_parser.add_argument('--config', default=POSTPROCESS_CONFIG_PATH, type=str, help='OpenMX配置文件路径')
    args = argument_parser.parse_args()
    openmx_server = PostProcessServer(config_path=args.config)
    num_threads = openmx_server.config.get('num_threads', 4)  # 从配置中获取线程数
    openmx_server.run(num_threads=num_threads)