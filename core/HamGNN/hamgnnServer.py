#server.py
"""
@author: ycxie
@date: 2025/07/02
@Last Modified: 2025/07/02
@Last Modified by: ycxie
基于Yang Zhong的HamGNN2.0main.py重构的保留预测部分的flask hamgnn服务器。
这个服务器使用Flask框架和Waitress服务器来处理客户端HTTP请求，
由客户端向server发送图数据路径或图数据本身，预测并返回哈密顿量结果。
免去了原有脚本中冷启动hamgnn模型的耗时过程，使得预测程序更灵活和高效。
"""
import torch
import traceback
import pprint
import warnings
import sys
import socket
import json
from pathlib import Path
import argparse
import os
from flask import Flask, request, jsonify
from waitress import serve
from torch_geometric.data import Data, Batch
# 导入HamGNN项目的相关组件
from .input.config_parsing import read_config
from .models.Model import Model
from .models.HamGNN.net import HamGNNTransformer, HamGNNConvE3, HamGNNPlusPlusOut
from types import SimpleNamespace
from ..communication import HamGNNCommunicator as Communicator, BaseCommunicator
import logging
import numpy as np
from flask import request, jsonify, Response as FlaskResponse
import pytorch_lightning as pl
from ..utils import write_server_info,delete_server_info
from ..utils import find_free_port, get_package_path
import threading
from functools import wraps
import time
import yaml
LOGGING_LEVEL = logging.INFO
BASIC_CONFIG_PATH = get_package_path("/HamGNN/hamgnn_basic_config.yaml")

# --- 服务器主类 ---
class HamGNNServer:
    def __init__(self, args):
        """
        初始化服务器，包括加载配置和模型。
        这个方法对应于服务器的“设置”或“准备”阶段。
        """
        self.app = Flask(__name__)
        self.app.logger.setLevel(LOGGING_LEVEL)
        self.app.logger.info("HamGNN服务器正在初始化...")
        self.args = args
        if not args.config:
            raise ValueError("配置文件路径不能为空，请使用 --config 参数指定。")
        config_path = args.config
        self.communicator: BaseCommunicator =  Communicator()
        self.config = self._load_config(config_path=config_path)
        self._setup_device()
        self._load_model()
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB"]
        self._register_routes()
        self.type = "HamGNNServer"  # 服务器类型标识
        

        self.max_concurrent_jobs = 12
        self.active_requests = 0
        self.lock = threading.Lock() # 线程锁，确保计数器在多线程环境下是安全的

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

    def _load_config(self, config_path: str):
        """加载YAML配置文件。"""
        # self.basic_config = yaml.safe_load(open(BASIC_CONFIG_PATH, 'r', encoding='utf-8'))
        # self.app.logger.debug(f"加载基本配置: {BASIC_CONFIG_PATH}")
        config = read_config(config_file_name=config_path)
        hostname = socket.getfqdn(socket.gethostname())
        config.setup.hostname = hostname
        if config.setup.ignore_warnings:
            warnings.filterwarnings('ignore')
        print("--- 服务器配置信息 ---")
        pprint.pprint(config)
        print("--------------------------")
        return config

    def _setup_device(self):
        """设置torch的计算设备和默认数据类型。"""
        self.dtype = torch.float32 if self.config.setup.precision == 32 else torch.float64
        torch.set_default_dtype(self.dtype)
        use_gpu = (
            torch.cuda.is_available() and
            self.config.setup.num_gpus is not None and
            self.config.setup.num_gpus > 0
        )
        
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.app.logger.info(f"使用的计算设备: {self.device}，计算精度: {self.dtype}")

    def _load_model(self):
        """构建模型结构并从检查点文件加载权重。"""
        self.app.logger.info("正在加载预训练模型用于推理...")

        # 这部分逻辑是从旧版脚本的全局作用域中迁移过来的
        graph_representation, output_module = self._build_model_architecture()
        try:
            is_strict = not self.args.no_strict_load 
        except:
            is_strict = True

        if is_strict:
            pl_model = Model.load_from_checkpoint(
            checkpoint_path=self.config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            losses=self.config.losses_metrics.losses,
            validation_metrics=self.config.losses_metrics.metrics,
            lr=0.0, # 推理时不需要学习率
        )
        else:
            self.app.logger.warning("使用非严格模式加载模型，这可能会忽略一些不匹配的参数。")
            pl_model = Model.load_from_checkpoint(
                checkpoint_path=self.config.setup.checkpoint_path,
                representation=graph_representation,
                output=output_module,
                losses=self.config.losses_metrics.losses,
                validation_metrics=self.config.losses_metrics.metrics,
                lr=0.0, # 推理时不需要学习率
                strict=False
            )

        # 从PyTorch Lightning的包装器中提取出核心的PyTorch模块
        self.gnn_model = pl_model.representation.to(self.device).eval()
        self.output_model = pl_model.output_module.to(self.device).eval()
        self.app.logger.info("模型加载成功。")

    def _build_model_architecture(self):
        """一个辅助方法，根据配置构建模型结构。"""
        # 这个函数与之前脚本中的版本完全相同
        config = self.config
        self.app.logger.info("正在构建模型结构...")
        if config.setup.GNN_Net.lower() in ['hamgnnconv', 'hamgnnpre', 'hamgnn_pre']:
            config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
            config.representation_nets.HamGNN_pre.setdefault('use_corr_prod', True)
            Gnn_net = HamGNNConvE3(config.representation_nets)
        elif config.setup.GNN_Net.lower() == 'hamgnntransformer':
            Gnn_net = HamGNNTransformer(config.representation_nets)
        else:
            raise NotImplementedError(f"网络类型: {config.setup.GNN_Net} 尚不支持!")

        if config.setup.property.lower() == 'hamiltonian':
            output_params = config.output_nets.HamGNN_out
            output_params.setdefault('add_H_nonsoc', False)
            output_params.setdefault('get_nonzero_mask_tensor', False)
            output_params.setdefault('zero_point_shift', False)
            output_module = HamGNNPlusPlusOut(irreps_in_node=Gnn_net.irreps_node_features, irreps_in_edge=Gnn_net.irreps_node_features, **output_params)
        else:
            raise NotImplementedError(f"本服务器仅支持 'hamiltonian' 属性的预测。")
        
        return Gnn_net, output_module

    def _preprocess_input(self, input_request):
        """
        将来自API的JSON输入转换为模型兼容的图数据对象。
        """
        return self.communicator.unpack_request(input_request, self.device)

    def _register_routes(self):
        """注册Flask路由，并将它们连接到类实例。"""
        
        @self.app.route("/health", methods=['GET'])
        def health_check():
            # 检查服务器是否存活以及模型是否已加载
            return jsonify({"status": "ok", "model_loaded": self.gnn_model is not None})
        
        @self.app.route("/load_status", methods=['GET'])
        def load_status():
            return jsonify({
                "active_requests": self.active_requests,
                "max_capacity": self.max_concurrent_jobs,
                "load_factor": self.active_requests / self.max_concurrent_jobs
            })

        @self.app.route("/predict", methods=['POST'])
        @self.track_load
        def predict():
            try:
                start_time = time.time()
                # 步骤1: 预处理输入数据
                graph, output_path = self._preprocess_input(request)
                self.app.logger.debug(f"预处理后的图数据: {graph}")
                # 步骤2: 在不计算梯度的模式下运行模型推理
                try:
                    with torch.no_grad():
                        representation = self.gnn_model(graph)
                        hamiltonian_output = self.output_model(
                    graph, representation
                )
                except Exception as e:
                    self.app.logger.error(f"模型推理过程中发生错误: {str(e)}")
                    return jsonify({"error": str(e)}), 500
                self.app.logger.debug(f"模型输出: {hamiltonian_output}")
                ifevaluate_loss = request.json.get("evaluate_loss", False)
                # self.app.logger.debug(graph)
                # self.app.logger.debug(list(graph))
                # self.app.logger.debug(graph.edge_index)
                # self.app.logger.debug(graph.x)
                # self.app.logger.debug(graph.hamiltonian)
                if ifevaluate_loss:
                    hamiltonian_true = graph.hamiltonian
                    hamiltonian_pred = hamiltonian_output["hamiltonian"]
                    # 计算L1和L2损失
                    l1_loss = torch.mean(torch.abs(hamiltonian_true - hamiltonian_pred)).item()
                    l2_loss = torch.mean((hamiltonian_true - hamiltonian_pred) ** 2).item()
                    hamiltonian_output["l1_loss"] = l1_loss
                    hamiltonian_output["l2_loss"] = l2_loss
                    self.app.logger.info(f"平均L1损失: {l1_loss}, 平均L2损失: {l2_loss}")
                if output_path is not None and output_path != "./":
                    try:
                        os.makedirs(output_path)
                    except FileExistsError:
                        self.app.logger.warning(f"输出目录已存在: {output_path}")
                elif output_path == "./":
                    output_path = os.path.dirname(request.json.get("graph_data_path", None))
                else:
                    current_path = get_package_path("")
                    t = int(time.time())
                    output_path = os.path.join(current_path, "temp", f"hamgnn_{t}{np.random.randint(100, 999)}")
                    try:
                        os.makedirs(output_path)
                    except FileExistsError:
                        self.app.logger.warning(f"输出目录已存在: {output_path}")
                    

                hamiltonian_output["output_path"] = output_path if output_path else None
                hamiltonian_output["return_directly"] = request.json.get("return_directly", False)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.app.logger.info(f"耗时: {elapsed_time:.2f}秒")
                return self.communicator.pack_response(hamiltonian_output)
                
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500
            
        @self.app.route("/shutdown")
        def shutdown():
            """
            处理服务器关闭请求。
            """
            info_file_path = get_package_path("server_info/hamgnn_server_info.json")
            self.app.logger.info("收到服务器关闭请求，正在进行关闭...")
            delete_server_info(os.getpid(), info_file_path)

    def run(self, num_threads=4):
        """
        启动服务器，包括HPC的服务发现功能。
        这个方法对应于服务器的“运行”阶段。
        """
        info_file_path = get_package_path("server_info/hamgnn_server_info.json")
        host = socket.getfqdn()
        port = find_free_port()
        self.app.logger.info(f"正在启动 Flask 服务器，地址: http://{host}:{port}")
        self.app.logger.info(f"正在注册服务器信息到: {info_file_path}")
        write_server_info(host=host, port=port, type=self.type, info_file=info_file_path, mode="a")
        serve(self.app, host="0.0.0.0", port=port, threads=num_threads)

   

if __name__ == '__main__':
    # 现在，主执行块变得极其简单和清晰。
    parser = argparse.ArgumentParser(description='HamGNN 预测服务器 (面向对象版)')
    parser.add_argument('--config', default='config.yaml', type=str, help='配置文件的路径')
    parser.add_argument(
        '--no-strict-load', 
        action='store_true', # 这表示如果出现此开关，则其值为True，否则为False
        help='【仅供测试使用】禁用严格的模型权重加载模式。'
    )
    args = parser.parse_args()
    BASIC_CONFIG_PATH= get_package_path("HamGNN/hamgnn_basic_config.yaml")
    basic_config= yaml.safe_load(open(BASIC_CONFIG_PATH, 'r', encoding='utf-8'))
    num_threads = basic_config.get('num_threads', 4)  # 从基本配置中获取线程数

    # 1. 创建一个服务器实例。所有的设置工作都在构造函数中完成。
    server = HamGNNServer(args)
    # 2. 运行服务器。
    server.run(num_threads=num_threads)