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
from ..communication import Communicator, BaseCommunicator
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
from flask import request, jsonify, Response as FlaskResponse
import pytorch_lightning as pl
# --- 服务器主类 ---
class HamGNNServer:
    def __init__(self, args):
        """
        初始化服务器，包括加载配置和模型。
        这个方法对应于服务器的“设置”或“准备”阶段。
        """
        logging.info("正在初始化 HamGNN 服务器...")
        self.args = args
        if not args.config:
            raise ValueError("配置文件路径不能为空，请使用 --config 参数指定。")
        config_path = args.config
        self.communicator: BaseCommunicator =  Communicator()
        self.config = self._load_config(config_path=config_path)
        self._setup_device()
        self._load_model()
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB"]
        self._register_routes()

    def _load_config(self, config_path: str):
        """加载YAML配置文件。"""
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
        logging.info(f"使用的计算设备: {self.device}，计算精度: {self.dtype}")

    def _load_model(self):
        """构建模型结构并从检查点文件加载权重。"""
        logging.info("正在加载预训练模型用于推理...")

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
            logging.warning("使用非严格模式加载模型，这可能会忽略一些不匹配的参数。")
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
        logging.info("模型加载成功。")

    def _build_model_architecture(self):
        """一个辅助方法，根据配置构建模型结构。"""
        # 这个函数与之前脚本中的版本完全相同
        config = self.config
        logging.info("正在构建模型结构...")
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

        @self.app.route("/predict", methods=['POST'])
        def predict():
            try:
                # 步骤1: 预处理输入数据
                graph, output_path = self._preprocess_input(request)
                logging.debug(f"预处理后的图数据: {graph}")
                # 步骤2: 在不计算梯度的模式下运行模型推理
                try:
                    with torch.no_grad():
                        representation = self.gnn_model(graph)
                        hamiltonian_output = self.output_model(
                    graph, representation
                )
                except Exception as e:
                    logging.error(f"模型推理过程中发生错误: {str(e)}")
                    return jsonify({"error": str(e)}), 500
                logging.debug(f"模型输出: {hamiltonian_output}")
                
                hamiltonian_output["output_path"] = output_path if output_path else None
                return self.communicator.pack_response(hamiltonian_output)
                
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500

    def run(self):
        """
        启动服务器，包括HPC的服务发现功能。
        这个方法对应于服务器的“运行”阶段。
        """
        {}
        info_file_path = Path(os.path.expanduser(os.path.join(os.path.dirname(os.path.abspath(__file__)),"hamgnn_server_info.json")))
        host = socket.getfqdn()
        port = self._find_free_port()
        logging.info(f"正在启动 Flask 服务器，地址: http://{host}:{port}")
        self._write_server_info(host, port, info_file_path)

        # 使用生产级的Waitress服务器来运行应用
        serve(self.app, host="0.0.0.0", port=port)

    # --- 用于HPC环境的辅助方法 ---
    @staticmethod
    def _find_free_port():
        """静态方法：动态查找一个未被占用的端口。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0)); return s.getsockname()[1]

    @staticmethod
    def _write_server_info(host, port, info_file):
        """静态方法：将服务器地址信息写入到共享文件。"""
        server_info = {"host": host, "port": port, "pid": os.getpid()}
        info_file.parent.mkdir(parents=True, exist_ok=True)
        with open(info_file, 'w') as f: json.dump(server_info, f)
        logging.info(f"服务器信息已写入: {info_file}")


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

    # 1. 创建一个服务器实例。所有的设置工作都在构造函数中完成。
    server = HamGNNServer(args)
    # 2. 运行服务器。
    server.run()