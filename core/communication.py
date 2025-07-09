# communicators.py
"""
@author: ycxie
@file: communication.py
@time: 2025/07/02
@Last Modified: 2025/07/02
@Last Modified by: ycxie
本模块定义了客户端与服务器之间进行通信的“通信器”。
每个通信器都封装了特定数据格式的所有处理逻辑，
包括请求的打包/解包和响应的打包/解包，可供客户端和服务器双方共同使用。
"""
import abc
import io
import json
import h5py
import numpy as np
from flask import request, jsonify, Response as FlaskResponse
from requests import Response as RequestsResponse
from torch_geometric.data import Data, Batch
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class BaseCommunicator(abc.ABC):
    """通信器的抽象基类，定义了客户端和服务器所需的所有接口。"""

    # --- Server-Side Methods ---
    @abc.abstractmethod
    def unpack_request(self, input_data, device: str = 'cpu') -> tuple:
        """【服务器端】从Flask请求中解析出输入数据。"""
        pass

    @abc.abstractmethod
    def pack_response(self, hamiltonian_output: dict) -> FlaskResponse: 
        """【服务器端】将计算结果打包成一个Flask响应对象。"""
        pass

    # --- Client-Side Methods ---
    @abc.abstractmethod
    def pack_request(self, structure_data: dict) -> tuple:
        """【客户端】将结构数据打包成HTTP请求体和请求头。"""
        pass

    @abc.abstractmethod
    def unpack_response(self, requests_response: RequestsResponse):
        """【客户端】从Requests响应中解析出最终结果。"""
        pass


class HamGNNCommunicator(BaseCommunicator):
    """
    HamGNN Server的通信器,处理客户端和服务器之间的所有数据交换。
    methods:
        unpack_request: 从Flask请求中解析输入数据，返回一个Graph Batch对象和可选的输出路径。请求可以是JSON或multipart/form-data格式上传的graph(.npz)文件。
        pack_response: 将计算结果打包成一个Flask的json响应对象，如果提供output_path，则将结果保存到指定位置，否则直接返回结果。
        pack_request: 将结构数据打包成HTTP请求体和请求头，返回一个元组，包含请求体和请求头。
        unpack_response: 从Requests响应中解析出最终结果，返回哈密顿量
    """
    def __init__(self):
        self.type = "HamGNNCommunicator"  # 通信器类型标识
    def unpack_request(self, request, device: str = 'cpu'):
        """
        从Flask请求中解析输入数据，返回一个Graph Batch对象和可选的输出路径。
        :param request: Flask请求对象,json格式("graph_data_path": "path/to/graph.npz", "output_path": "path/to/output")
        :param device: 计算设备，默认为'cpu'。
        """
        logging.info("正在预处理输入数据... ")
        loaded_object=None
        try: 
            # 检查请求中是否直接包含 .npz 文件（通过 multipart/form-data 上传）
            if 'graph' in request.files:
                #TODO:改成通过2进制流发送graph_data
                #FIXME 现在似乎找不到这个文件，报错：
                #                 2025-07-02 00:05:38,224 - ERROR - 预处理输入数据时发生错误: 'NoneType' object has no attribute 'open'
                # Traceback (most recent call last):
                #   File "/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/communication.py", line 63, in unpack_request
                #     loaded_object = npz['graph'].item()
                #   File "/ssd/work/ycxie/.conda_envs/hamgnn/lib/python3.9/site-packages/numpy/lib/npyio.py", line 251, in __getitem__
                #     bytes = self.zip.open(key)
                # AttributeError: 'NoneType' object has no attribute 'open'

                # During handling of the above exception, another exception occurred:

                # Traceback (most recent call last):
                #   File "/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/HamGNN/server.py", line 156, in predict
                #     graph, output_path = self._preprocess_input(request)
                #   File "/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/HamGNN/server.py", line 142, in _preprocess_input
                #     return self.communicator.unpack_request(input_request, self.device)
                #   File "/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/communication.py", line 132, in unpack_request
                #     raise ValueError(f"无法处理输入数据: {e}")
                # ValueError: 无法处理输入数据: 'NoneType' object has no attribute 'open'
                
                npz_file = request.files['graph']
                # 修复：转换为可寻址的 BytesIO
                file_stream = io.BytesIO(npz_file.read())
                with np.load(file_stream, allow_pickle=True) as npz:
                    if 'graph' not in npz:
                        raise KeyError("上传的 .npz 文件中缺少 'graph' 数据")
                loaded_object = npz['graph'].item()
                output_path = request.get_json().get("output_path", None)
                logging.debug("communicator 获取graph.npz文件")
            # 否则从 JSON 中读取 graph_data_path
            else:
                input_data = request.get_json()
                if not input_data or 'graph_data_path' not in input_data:
                    raise ValueError("请求中必须包含上传的 .npz 文件或 graph_data_path")
                
                graph_data_path = input_data['graph_data_path']
                if not graph_data_path:
                    raise ValueError("graph_data_path 不能为空")   
                graph_data_path = input_data.get("graph_data_path", None)

                if not graph_data_path:
                    raise ValueError("输入数据中必须包含 'graph_data_path' 键。")
                with np.load(graph_data_path, allow_pickle=True) as npz:
                    # 假设数据总是存在'graph'这个键下
                    if 'graph' not in npz:
                        raise KeyError(f"在 {graph_data_path} 中找不到必需的 'graph' 数据。")
                    loaded_object = npz['graph'].item()
                logging.info(f"communicator 从路径加载 .npz 文件: {graph_data_path}")
                output_path = input_data.get("output_path", None)

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
            return final_batch_object.to(device), output_path
        except Exception as e:
            logging.error(f"预处理输入数据时发生错误: {e}")
            raise ValueError(f"无法处理输入数据: {e}")
    
    

    def pack_response(self, hamiltonian_output: dict):
        """
        将计算结果打包成一个Flask的json响应对象。
        :param hamiltonian_output: 包含计算结果的字典，必须包含 'hamiltonian' 键,对应hamiltonian张量。
        以及可选的 'output_path' 键，指定结果保存的路径。
        :return: Flask响应对象，包含预测结果或错误信息。
        """
        output_path = hamiltonian_output.get('output_path', None)
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

    def pack_request(self, structure_data: dict) -> tuple:
        # JSON请求体就是字典本身，请求头可以为空或指定application/json
        headers = {'Content-Type': 'application/json'}
        return structure_data, headers

    def unpack_response(self, requests_response: RequestsResponse):
        response_data = requests_response.json()
        if "output_file" in response_data:
            return np.load(response_data['output_file'], allow_pickle=True).item()
        elif "hamiltonian_matrix" in response_data:
            return np.array(response_data['hamiltonian_matrix'])
        else:
            raise ValueError("响应中不包含有效的预测结果。请检查服务器日志以获取更多信息。")
    
class OpenmxCommunicator(BaseCommunicator):
    """
    Openmx Server的通信器,处理客户端和服务器之间的所有数据交换。
    methods:
    """
    def __init__(self):
        self.type = "OpenmxCommunicator"  # 通信器类型标识
    def unpack_request(self, request):
        """
        从Flask请求中解析输入数据，返回一个Graph Batch对象和可选的输出路径。
        :param request: Flask请求对象,json格式("graph_data_path": "path/to/graph.npz", "output_path": "path/to/output")
        :param device: 计算设备，默认为'cpu'。
        """
        logging.info("正在预处理输入数据... ")
        
        try: 
            json_data = request.get_json()
            if not json_data or 'structure' not in json_data:
                return jsonify({"error": "请求中必须包含 'structure' 键。"}), 400
            structure = json_data['structure']
            graph_para = json_data.get('graph_para', {})
            current_dir = json_data.get('current_dir', False)
            return structure, graph_para, current_dir
        except Exception as e:
            logging.error(f"预处理输入数据时发生错误: {e}")
            return jsonify({"error": f"无法处理输入数据: {e}"}), 400        
    
    

    def pack_response(self,response_data: dict):
        """
        将计算结果打包成一个Flask的json响应对象。
        """
        return jsonify(response_data), 200
        
    def pack_request(self, structure_data: dict) -> tuple:
        headers = {'Content-Type': 'application/json'}
        return structure_data, headers

    def unpack_response(self, requests_response: RequestsResponse):
        return requests_response.json()
class PostProcessCommunicator(BaseCommunicator):
    """
    PostProcess Server的通信器,处理客户端和服务器之间的所有数据交换。
    methods:
    """
    def __init__(self):
        self.type = "PostProcessCommunicator"  # 通信器类型标识
    def unpack_request(self, request):
        """
        从Flask请求中解析输入数据
        """
        logging.info("正在预处理输入数据... ")
        try: 
            json_data = request.get_json()
            if not json_data or 'hamiltonian_path' not in json_data:
                return jsonify({"error": "请求中必须包含 'hamiltonian_path' 键。"}), 400
            hamiltonian_path = json_data['hamiltonian_path']
            graph_data_path = json_data['graph_data_path']
            if not json_data or 'graph_data_path' not in json_data:
                return jsonify({"error": "请求中必须包含 'graph_data_path' 键。"}), 400
            if type(hamiltonian_path) is not str:
                try:
                    hamiltonian_path = str(hamiltonian_path)
                except Exception as e:
                    logging.error(f"转换 hamiltonian_path 时发生错误: {e}")
                    raise ValueError("hamiltonian_path 必须是字符串类型。")
            if not Path(hamiltonian_path).exists():
                 raise ValueError(f"指定的哈密顿量文件路径不存在: {hamiltonian_path}")
            if type(json_data['graph_data_path']) is not str:
                try:
                    graph_data_path = str(json_data['graph_data_path'])
                except Exception as e:
                    logging.error(f"转换 graph_data_path 时发生错误: {e}")
                    raise ValueError("graph_data_path 必须是字符串类型。")
            if not Path(graph_data_path).exists():
                raise ValueError(f"指定的图数据文件路径不存在: {graph_data_path}")
            band_para = json_data.get('band_para', {})
            current_dir = json_data.get('current_dir',True)
            return hamiltonian_path, graph_data_path, band_para, current_dir
        except Exception as e:
            logging.error(f"预处理输入数据时发生错误: {e}")
            return jsonify({"error": f"无法处理输入数据: {e}"}), 400        

    def pack_response(self,response_data: dict):
        """
        将计算结果打包成一个Flask的json响应对象。
        """
        return jsonify(response_data), 200
    def pack_request(self, structure_data: dict) -> tuple:
        return jsonify(structure_data)

    def unpack_response(self, requests_response: RequestsResponse):
        return requests_response.json()

    
