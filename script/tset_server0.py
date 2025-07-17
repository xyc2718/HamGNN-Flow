# tests/test_server.py
"""
针对 server_oop.py 的集成测试。
这个测试会真实地启动一个服务器子进程，并通过HTTP请求进行测试。
"""
import pytest
import requests
import json
import subprocess
import time
import os
import sys
from pathlib import Path
import torch
import numpy as np
import sys
import shutil
from pathlib import Path
# 导入通信器，用于在客户端侧打包和解包
from core.communication import JSONCommunicator, HDF5Communicator, BaseCommunicator

# --- 参数化设置，让测试同时覆盖'json'和'hdf5'两种模式 ---
@pytest.fixture(scope="module", params=["json"])
def server_process(request):
    """
    一个参数化的fixture。它会为每种通信模式（json, hdf5）
    分别启动一次服务器，并在测试结束后关闭它。
    """
    # communicator_type = request.param # 获取参数 'json' 或 'hdf5'

    # # --- 创建虚拟的配置文件和模型文件 ---
    # test_dir = Path("./test_temp_server")
    # test_dir.mkdir(exist_ok=True)
    # config_path = test_dir / f"test_config_{communicator_type}.yaml"
    # checkpoint_path = test_dir / "dummy_model.ckpt"
    # default_info_path = Path(os.path.expanduser("~/.config/hamgnn_flow/server_info.json"))

    # torch.save({"state_dict": {}}, checkpoint_path)

    communicator_type = request.param
    
    # --- 创建虚拟的配置文件和模型文件 ---
    test_dir = Path("./test_temp_server")
    if test_dir.exists():
        shutil.rmtree(test_dir) # 先清理，防止上次失败的残留
    test_dir.mkdir(exist_ok=True)
    
    config_path = test_dir / f"test_config_{communicator_type}.yaml"
    checkpoint_path = test_dir / "dummy_model.ckpt"
    default_info_path = Path(os.path.expanduser("~/.config/hamgnn_flow/server_info.json"))

    torch.save({"state_dict": {}}, checkpoint_path)
    config_content = f"""
setup:
  GNN_Net: HamGNNConv
  property: hamiltonian
  precision: 32
  num_gpus: 0
  checkpoint_path: {checkpoint_path}
  ignore_warnings: true
  stage: test

representation_nets:
  HamGNN_pre:
    # --- 根据您的示例配置，补全所有已知的必需参数 ---
    num_types: 128
    num_layers: 3
    num_radial: 128
    num_heads: 4
    correlation: 2
    num_hidden_features: 32
    
    rbf_func: bessel
    cutoff: 6.0
    cutoff_func: cos
    
    irreps_edge_sh: "0e+1o+2e" # 简化版，能通过即可
    irreps_node_features: "64x0e+16x1o+8x2e" # 简化版
    
    edge_sh_normalize: true
    edge_sh_normalization: component

    radial_MLP: [128, 128]
    use_corr_prod: True
    use_kan: False
    set_features: true
    
    radius_scale: 1.01
    build_internal_graph: False  # <--- 新增此行，修复当前错误
    
    # --- 以下是旧虚拟配置中的参数，有些可能与上面重复，保留以防万一 ---
    radius: 6.0 
    max_l: 2
    num_node_features: 16
    num_edge_features: 16
    num_kernel: 1
    num_radial_basis: 128
    num_interaction_layers: 3


output_nets:
  HamGNN_out:
    ham_type: openmx        
    nao_max: 26             
    add_H0: True            
    symmetrize: True        
    calculate_band_energy: False
    num_k: 0
    k_path: ''
    band_num_control: []
    soc_switch: '0000'
    nonlinearity_type: 'gate'
    add_H0: False
    spin_constrained: False
    collinear_spin: 'none'
    minMagneticMoment: 0.1

losses_metrics:
  losses: []
  metrics: []
"""
    with open(config_path, 'w') as f: f.write(config_content)
    
    # --- 关键修改：将日志重定向到文件 ---
    stdout_log_path = test_dir / "stdout.log"
    stderr_log_path = test_dir / "stderr.log"

    with open(stdout_log_path, 'w') as f_out, open(stderr_log_path, 'w') as f_err:
        project_root = Path(__file__).parent.parent
        server_command = [
            sys.executable,
            "-m", "core.HamGNN.server",
            "--config", str(config_path),
            "--no-strict-load"
        ]
        
        process = subprocess.Popen(
            server_command,
            stdout=f_out, # 重定向到文件
            stderr=f_err, # 重定向到文件
            cwd=project_root
        )

    # --- 等待服务器就绪 (逻辑不变) ---
    server_url = None
    for _ in range(40): 
        time.sleep(0.5)
        if default_info_path.exists() and default_info_path.stat().st_size > 0:
            try:
                with open(default_info_path, 'r') as f: info = json.load(f)
                server_url = f"http://127.0.0.1:{info['port']}"
                requests.get(f"{server_url}/health", timeout=1)
                print(f"\nServer for '{communicator_type}' started at {server_url}")
                break
            except requests.ConnectionError:
                # 如果连接被拒绝，说明服务器可能已崩溃，继续等待或最后统一处理
                continue
            except (json.JSONDecodeError):
                continue
    
    # fixture的`yield`部分将在try...finally...块中执行
    try:
        if server_url is None:
            pytest.fail(f"Server for '{communicator_type}' failed to start.")
        
        # 确定communicator实例
        if communicator_type == "json":
            communicator = JSONCommunicator()
        else:
            communicator = HDF5Communicator()
            
        yield server_url, communicator

    finally:
        # --- 清理工作 ---
        print(f"\nShutting down server for '{communicator_type}'...")
        process.terminate()
        process.wait(timeout=5)
        
        # 打印日志文件的内容，以便调试
        with open(stdout_log_path, 'r') as f:
            print("\n--- Server STDOUT ---")
            print(f.read())
        with open(stderr_log_path, 'r') as f:
            print("\n--- Server STDERR ---")
            print(f.read())
            
        if default_info_path.exists():
            default_info_path.unlink()
        if test_dir.exists():
            shutil.rmtree(test_dir)


    

# --- 测试用例 ---

def test_health_check(server_process):
    """测试 /health 端点，对所有模式都应有效。"""
    server_url, _ = server_process
    response = requests.get(f"{server_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction(server_process):
    """
    通过参数化，这个测试会分别用JSON和HDF5运行两次。
    """
    server_url, communicator = server_process
    
    # 准备客户端数据
    structure_data = {
        "atomic_numbers": [14, 14],
        "cell": np.random.rand(3, 3).tolist(), # tolist()使其对JSON友好
        "positions": np.random.rand(2, 3).tolist()
    }

    # 使用通信器打包请求
    payload, headers = communicator.pack_request(structure_data)
    
    # 发送请求
    if isinstance(payload, dict): # JSON case
        response = requests.post(f"{server_url}/predict", json=payload, headers=headers)
    else: # HDF5 case
        response = requests.post(f"{server_url}/predict", data=payload, headers=headers)
    
    assert response.status_code == 200

    # 使用通信器解包响应
    result = communicator.unpack_response(response)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10) # 假设从虚拟模型我们知道输出形状