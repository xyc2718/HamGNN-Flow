# communicators.py
"""
本模块定义了客户端与服务器之间进行通信的“通信器”。
每个通信器都封装了特定数据格式（如JSON, HDF5）的所有处理逻辑，
包括请求的打包/解包和响应的打包/解包，可供客户端和服务器双方共同使用。
"""
import abc
import io
import json
import h5py
import numpy as np
from flask import request, jsonify, Response as FlaskResponse
from requests import Response as RequestsResponse

class BaseCommunicator(abc.ABC):
    """通信器的抽象基类，定义了客户端和服务器所需的所有接口。"""

    # --- Server-Side Methods ---
    @abc.abstractmethod
    def unpack_request(self, flask_request: request):
        """【服务器端】从Flask请求中解析出输入数据。"""
        pass

    @abc.abstractmethod
    def pack_response(self, result_data: dict, app) -> FlaskResponse:
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


class JSONCommunicator(BaseCommunicator):
    """使用JSON格式进行通信的通信器。"""

    # --- Server-Side Implementations ---
    def unpack_request(self, flask_request: request):
        if not flask_request.is_json:
            raise ValueError("请求的内容类型必须是JSON")
        return flask_request.get_json()

    def pack_response(self, result_data: dict, app) -> FlaskResponse:
        return jsonify(result_data)

    # --- Client-Side Implementations ---
    def pack_request(self, structure_data: dict) -> tuple:
        # JSON请求体就是字典本身，请求头可以为空或指定application/json
        headers = {'Content-Type': 'application/json'}
        return structure_data, headers

    def unpack_response(self, requests_response: RequestsResponse):
        result_data = requests_response.json()
        # 假设我们关心的是hamiltonian_matrix
        return result_data.get('hamiltonian_matrix')


class HDF5Communicator(BaseCommunicator):
    """使用HDF5格式进行通信的通信器。"""

    # --- Server-Side Implementations ---
    def unpack_request(self, flask_request: request):
        if flask_request.content_type not in ['application/x-hdf5', 'application/octet-stream']:
            raise ValueError("请求的内容类型必须是application/x-hdf5或application/octet-stream")
        
        with io.BytesIO(flask_request.get_data()) as hdf5_in_memory:
            with h5py.File(hdf5_in_memory, 'r') as f:
                input_data = {key: f[key][:] for key in f.keys()}
        return input_data

    def pack_response(self, result_data: dict, app) -> FlaskResponse:
        with io.BytesIO() as hdf5_out_memory:
            with h5py.File(hdf5_out_memory, 'w') as f:
                if 'prediction_result' not in result_data or not isinstance(result_data['prediction_result'], np.ndarray):
                    raise TypeError("HDF5响应数据必须包含一个键为'prediction_result'的numpy数组")
                f.create_dataset('hamiltonian_matrix', data=result_data['prediction_result'], compression="gzip")
            
            response_bytes = hdf5_out_memory.getvalue()
        
        return FlaskResponse(response=response_bytes, status=200, mimetype='application/x-hdf5')

    # --- Client-Side Implementations ---
    def pack_request(self, structure_data: dict) -> tuple:
        with io.BytesIO() as hdf5_in_memory:
            with h5py.File(hdf5_in_memory, 'w') as f:
                # 假设输入的structure_data是一个字典，值为numpy数组或列表
                for key, value in structure_data.items():
                    f.create_dataset(key, data=np.array(value))
            
            request_bytes = hdf5_in_memory.getvalue()
        
        headers = {'Content-Type': 'application/x-hdf5'}
        return request_bytes, headers

    def unpack_response(self, requests_response: RequestsResponse):
        with io.BytesIO(requests_response.content) as hdf5_out_memory:
            with h5py.File(hdf5_out_memory, 'r') as f:
                hamiltonian_matrix = f['hamiltonian_matrix'][:]
        return hamiltonian_matrix