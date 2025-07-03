# tests/test_server.py
"""
针对 server_oop.py 的最终版集成测试。
该测试模拟了“基于路径”的API调用方式。
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
import shutil

# 导入通信器，主要用于测试json格式的“任务单”
from core.communication import Communicator
# 导入PyG的Data对象，用于创建虚拟的图数据
from torch_geometric.data import Data

# --- 参数化设置 ---
# 注意：由于我们最终采用了“发送路径”的API设计，其载荷(payload)是JSON格式。
# 因此，我们主要测试JSON通信器。如果未来您为这个API也设计了HDF5的“任务单”格式，
# 才需要将'hdf5'加回来。
@pytest.fixture(scope="module", params=["json"])
def server_process(request):
    """
    一个参数化的fixture。它会为每种通信模式启动一次服务器，
    并在测试结束后关闭它。
    """
    communicator_type = request.param
    
    # --- 创建虚拟的配置文件和模型文件 ---
    test_dir = Path("./test_temp_server_final")
    if test_dir.exists():
        shutil.rmtree(test_dir) # 先清理，防止上次失败的残留
    test_dir.mkdir(exist_ok=True)
    
    config_path = test_dir / f"test_config_{communicator_type}.yaml"
    checkpoint_path = test_dir / "dummy_model.ckpt"
    default_info_path = Path(os.path.expanduser("/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/HamGNN/hamgnn_server_info.json"))

    torch.save({"state_dict": {}}, checkpoint_path)

    # 使用我们最终调试成功的、最完整的虚拟配置
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
    num_types: 128
    num_layers: 3
    num_radial: 128
    num_heads: 4
    correlation: 2
    num_hidden_features: 32
    rbf_func: bessel
    cutoff: 6.0
    cutoff_func: cos
    irreps_edge_sh: "0e+1o+2e"
    irreps_node_features: "64x0e+16x1o+8x2e"
    edge_sh_normalize: true
    edge_sh_normalization: component
    radial_MLP: [128, 128]
    use_corr_prod: True
    use_kan: False
    set_features: true
    radius_scale: 1.01
    build_internal_graph: False
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
    ham_only: true
    calculate_band_energy: False
    num_k: 0
    k_path: ''
    band_num_control: []
    soc_switch: '0000'
    nonlinearity_type: 'gate'
    spin_constrained: False
    collinear_spin: 'none'
    minMagneticMoment: 0.2
    add_H_nonsoc: False

losses_metrics:
  losses: []
  metrics: []
"""
    with open(config_path, 'w') as f: f.write(config_content)
    
    # --- 启动服务器子进程 ---
    # (这部分逻辑已经验证是正确的，保持不变)
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
            stdout=f_out, stderr=f_err, cwd=project_root
        )

    # --- 等待服务器就绪 ---
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
                continue
            except (json.JSONDecodeError):
                continue
    
    # --- yield/finally 清理逻辑 ---
    try:
        if server_url is None:
            # 如果服务器启动失败，读取并打印日志
            with open(stdout_log_path, 'r') as f: print("\n--- Server STDOUT ---\n" + f.read())
            with open(stderr_log_path, 'r') as f: print("\n--- Server STDERR ---\n" + f.read())
            pytest.fail(f"Server for '{communicator_type}' failed to start.")
        
        # 传递服务器URL和临时目录的路径，方便测试用例在其中创建文件
        yield server_url, test_dir

    finally:
        print(f"\nShutting down server for '{communicator_type}'...")
        process.terminate()
        process.wait(timeout=5)
        if test_dir.exists():
            shutil.rmtree(test_dir)
        if default_info_path.exists():
            default_info_path.unlink()

# --- 测试用例 ---

def test_health_check(server_process):
    """测试 /health 端点，确认服务器已启动并健康。"""
    server_url, _ = server_process
    response = requests.get(f"{server_url}/health")
    assert response.status_code == 200, f"健康检查失败，内容: {response.text}"
    assert response.json()["status"] == "ok"

def test_prediction_from_path(server_process):
    """
    测试核心的“基于路径”的预测功能。
    """
    server_url, test_dir = server_process
    
    # 1. 在临时测试目录中，定义输入和输出文件的路径
    input_path = test_dir / "graph.npz"
    output_path = test_dir / "result.npz"

    # 2. 创建一个虚拟的输入 .npz 文件
    #    这个文件的结构必须与服务器端 _preprocess_input 方法期望的一致
    graph_obj = Data(
        z=torch.tensor([14, 14], dtype=torch.long),
        pos=torch.rand(2, 3, dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        # 这里可以根据需要添加更多真实的图属性
    )
    # 我们将一个字典保存在.npz文件的'graph'键下
    data_to_save = {0: graph_obj}
    np.savez_compressed(input_path, graph=data_to_save)
    assert input_path.exists(), "测试用的输入文件未能成功创建"

    # 3. 准备 "任务单" JSON，使用绝对路径
    job_ticket = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve())
    }

    # 4. 发送POST请求到 /predict_from_path 端点
    response = requests.post(f"{server_url}/predict_from_path", json=job_ticket)
    
    # 5. 断言服务器返回了成功状态码和正确的JSON响应
    assert response.status_code == 200, f"预测请求失败，内容: {response.text}"
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["output_path"] == str(output_path.resolve())

    # 6. 最关键的断言：检查服务器是否真的在指定路径创建了结果文件
    assert output_path.exists(), "服务器没有在指定的输出路径创建结果文件"

    # (可选) 还可以加载结果文件，检查其内容
    result_data = np.load(output_path)
    assert 'hamiltonian' in result_data
    print(f"\n测试成功，结果文件已创建，包含键: {list(result_data.keys())}")