"""
server_oop.py

HamGNN预测服务器的面向对象实现版本。
该版本将服务器逻辑封装在一个类中，以实现更好的代码组织、状态管理和可扩展性。
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
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
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
        self.config = self._load_config(config_path=config_path)
        self._setup_device()
        self._load_model()
        self.app = Flask(__name__)
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

    def _preprocess_input(self, input_data: dict):
        """
        将来自API的JSON输入转换为模型兼容的图数据对象。

        """
        logging.info("正在预处理输入数据... ")
        try:
            graph_data_path=input_data.get("graph_data_path", None)
            output_path = input_data.get("output_path", None)             
            if not graph_data_path:
                raise ValueError("输入数据中必须包含 'graph_data_path' 键。")
            
                # 1. 从.npz文件中加载核心的Python对象
            with np.load(graph_data_path, allow_pickle=True) as npz_file:
                # 假设数据总是存在'graph'这个键下
                if 'graph' not in npz_file:
                    raise KeyError(f"在 {graph_data_path} 中找不到必需的 'graph' 数据。")
                loaded_object = npz_file['graph'].item()

            # 2. 根据加载对象的类型，进行智能适配
            
            # --- 情况1：加载的对象本身就是一个Batch对象 ---
            if isinstance(loaded_object, Batch):
                logging.info(f"检测到输入为Batch对象，包含 {loaded_object.num_graphs} 个图。直接使用。")
                final_batch_object = loaded_object

            # --- 情况2：加载的对象是一个Data对象的列表 ---
            elif isinstance(loaded_object, list):
                if not loaded_object:
                    raise ValueError("输入的图数据列表为空。")
                if not all(isinstance(g, Data) for g in loaded_object):
                    raise TypeError("列表中的元素必须都是PyG的Data对象。")
                logging.info(f"检测到输入为列表，包含 {len(loaded_object)} 个图。正在转换为Batch对象...")
                final_batch_object = Batch.from_data_list(loaded_object)

            # --- 情况3：加载的对象是一个Data对象的字典 ---
            elif isinstance(loaded_object, dict):
                if not loaded_object:
                    raise ValueError("输入的图数据字典为空。")
                
                list_of_graphs = list(loaded_object.values())
                
                if not all(isinstance(g, Data) for g in list_of_graphs):
                    raise TypeError("字典中的值必须都是PyG的Data对象。")

                logging.info(f"检测到输入为字典，包含 {len(list_of_graphs)} 个图。正在转换为Batch对象...")
                final_batch_object = Batch.from_data_list(list_of_graphs)
                
            # --- 情况4：加载的对象是一个单独的Data对象 ---
            elif isinstance(loaded_object, Data):
                logging.info("检测到输入为单个图（Data对象）。正在转换为包含一个图的Batch对象...")
                final_batch_object = Batch.from_data_list([loaded_object])
                
            # --- 其他无法识别的情况 ---
            else:
                raise TypeError(
                    f"不支持的已加载数据类型: {type(loaded_object)}。"
                    "只支持Batch, list[Data], dict[any, Data], 或单个Data对象。"
                )
                
            # 3. 最终确保返回的对象位于正确的计算设备上
            return final_batch_object.to(self.device), output_path
        except Exception as e:
            logging.error(f"预处理输入数据时发生错误: {e}")
            raise ValueError(f"无法处理输入数据: {e}")
    

    def _register_routes(self):
        """注册Flask路由，并将它们连接到类实例。"""
        
        @self.app.route("/health", methods=['GET'])
        def health_check():
            # 检查服务器是否存活以及模型是否已加载
            return jsonify({"status": "ok", "model_loaded": self.gnn_model is not None})

        @self.app.route("/predict", methods=['POST'])
        def predict():
            # 检查请求是否为JSON格式
            if not request.is_json:
                return jsonify({"error": "请求必须是JSON格式"}), 400
            
            input_data = request.get_json()
    
            try:
                # 步骤1: 预处理输入数据
                graph, output_path = self._preprocess_input(input_data)
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
                hamiltonian_tensor = hamiltonian_output['hamiltonian']
                # 步骤4: 现在可以安全地对这个张量进行后续处理了
     
                if output_path:
                    # 如果提供了输出路径，则将结果保存到指定位置
                    output_file = Path(output_path) / "prediction_hamiltonian.npy"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_file, hamiltonian_tensor.cpu().numpy())
                    logging.info(f"预测结果已保存到: {output_file}")
                    return jsonify({"output_file": str(output_file), "status": "success"}), 200
                else:
                    # 如果没有提供输出路径，则直接返回结果
                    result = hamiltonian_tensor.cpu().numpy().tolist()
                    return jsonify({"hamiltonian_matrix": result, "status": "success"}), 200
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500

    def run(self):
        """
        启动服务器，包括HPC的服务发现功能。
        这个方法对应于服务器的“运行”阶段。
        """
        info_file_path = Path(os.path.expanduser("~/.config/hamgnn_flow/server_info.json"))
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