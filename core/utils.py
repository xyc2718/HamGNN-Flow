import os
import json
import logging
from pathlib import Path
from typing import Union
import socket
from filelock import FileLock, Timeout
import requests
from typing import Dict, List, Any
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

corePath=Path(__file__).parent
def find_free_port():
    """静态方法：动态查找一个未被占用的端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]
def get_package_path(path):
    return corePath / path

def write_server_info(host: str, port: int, type: str, info_file: Union[str, Path],mode: str = 'w'):
    """
    以进程安全的方式将服务器地址信息写入到共享文件。
    
    此函数使用文件锁来确保在多进程（或多线程）环境下，
    文件写入操作是原子的，避免竞态条件。

    Args:
        host (str): 服务器主机地址。
        port (int): 服务器端口。
        type (str): 服务器类型。
        info_file (str | Path): 要写入的目标文件路径。
    """
    # 确保 info_file 是 Path 对象，方便操作
    info_file_path = Path(info_file)
    # 基于目标文件名创建一个唯一的锁文件名，例如 'server.json' -> 'server.json.lock'
    lock_file_path = info_file_path.with_suffix(info_file_path.suffix + '.lock')

    # 初始化文件锁，设置一个5秒的超时，防止无限期等待
    lock = FileLock(lock_file_path, timeout=5)

    try:
        # 使用 'with' 语句来自动获取和释放锁
        # 这会阻塞，直到锁被获取或超时
        with lock:
            # --------- 关键代码段开始 ---------
            # 只有获得锁的进程才能执行这里的代码

            server_info = {
                "host": host,
                "port": port,
                "pid": os.getpid(),
                "type": type
            }
            
            # 确保目录存在
            info_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 以写入模式打开文件并写入 JSON 数据
            with open(info_file_path, mode, encoding='utf-8') as f:
                json.dump(server_info, f, ensure_ascii=False)
                f.write('\n') 
                
            logging.info(f"成功获取锁，服务器信息已写入: {info_file_path}")
    except Timeout:
        logging.error(f"获取文件锁超时 ({lock_file_path})。另一个进程可能持有锁太久。")
    except Exception as e:
        logging.error(f"写入文件时发生未知错误: {e}")


def delete_server_info(pid: int, info_file: Union[str, Path]):
    """
    以进程安全的方式从共享文件中删除指定的服务器信息记录。

    此函数使用文件锁来确保在多进程环境下，文件读取、修改和
    写入操作是原子的。它通过 pid 来识别要删除的记录。

    Args:
        pid (int): 要删除的服务器记录的进程 ID。
        info_file (str | Path): 目标文件路径。
    """
    info_file_path = Path(info_file)
    # 确保锁文件与写入/添加函数使用的锁文件一致
    lock_file_path = info_file_path.with_suffix(info_file_path.suffix + '.lock')

    # 初始化文件锁
    lock = FileLock(lock_file_path, timeout=5)

    try:
        # 使用 'with' 语句自动获取和释放锁
        with lock:
            # --------- 关键代码段开始 (Read-Modify-Write) ---------

            # 如果文件不存在或为空，则无需执行任何操作
            if not info_file_path.exists() or info_file_path.stat().st_size == 0:
                logging.info(f"文件不存在或为空 ({info_file_path})，无需删除。")
                return

            # 1. 读取文件中的所有记录
            try:
                with open(info_file_path, 'r', encoding='utf-8') as f:
                    all_records = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logging.error(f"无法读取或解析 JSON 文件: {info_file_path}。")
                return
            
            # 确保数据是列表格式
            if not isinstance(all_records, list):
                logging.error(f"文件 {info_file_path} 的内容不是预期的列表格式。")
                return

            # 2. 筛选出不包含指定 pid 的记录
            records_to_keep = [record for record in all_records if record.get("pid") != pid]

            if len(records_to_keep) == len(all_records):
                logging.warning(f"未找到 PID 为 {pid} 的记录，文件未作修改。")
                return
            # 3. 将筛选后的记录写回文件
            with open(info_file_path, 'w', encoding='utf-8') as f:
                json.dump(records_to_keep, f, ensure_ascii=False)    
            logging.info(f"成功删除 PID 为 {pid} 的记录，文件已更新: {info_file_path}")

    except Timeout:
        logging.error(f"获取文件锁超时 ({lock_file_path})。另一个进程可能持有锁太久。")
    except Exception as e:
        logging.error(f"删除记录时发生未知错误: {e}")


SERVER_INFO_PATH = get_package_path('server_info')

SERVER_INFO_FILE_DICT = {"openmx":Path(os.path.expanduser("/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/server_info/openmx_server_info.json")),
                "postprocess":Path(os.path.expanduser("/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/server_info/postprocess_server_info.json")),
                "hamgnn":Path(os.path.expanduser("/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/server_info/hamgnn_server_info.json")),
                "orchestrator":Path(os.path.expanduser("/ssd/work/ycxie/hamgnn/testopenmx/HamGNN-Flow/core/server_info/orchestrator_server_info.json")) }
def get_server_url(type: str) -> str:
    """
    从共享文件中读取服务器地址和端口。
    对于 'hamgnn' 类型，此函数会执行以下操作：
    1. 从文件中读取一个服务器地址列表。
    2. 通过健康检查 API (`/health`) 筛选出可用的服务器。
    3. 查询每个健康服务器的负载状态 API (`/load_status`)。
    4. 返回负载最小的服务器的 URL。

    对于其他类型，它会读取包含单个服务器信息的 JSON 文件。
    """
    if type not in SERVER_INFO_FILE_DICT:
        raise ValueError(f"未知的服务器类型: {type}。可用类型: {list(SERVER_INFO_FILE_DICT.keys())}")

    server_info_file = SERVER_INFO_FILE_DICT[type]
    if not server_info_file.exists():
        raise FileNotFoundError(f"服务器信息文件不存在: {type}:{server_info_file}\n请确认服务器作业是否已成功运行。")

    if type == 'hamgnn':
        # --- hamgnn 的特殊处理逻辑 ---
        
        # 1. 读取所有服务器地址
        with open(server_info_file, 'r') as f:
            # 过滤掉空行
            server_info = [line.strip() for line in f if line.strip()]

        if not server_info:
            raise ValueError(f"服务器列表文件为空: {server_info_file}")

        # 2. 健康检查，构建可访问服务器列表
        healthy_servers: List[str] = []
        for server_info_line in server_info:
            try:
                info= json.loads(server_info_line)
                address = f"{info['host']}:{info['port']}"
                health_check_url = f"http://{address}/health"
                # 设置一个较短的超时时间
                response = requests.get(health_check_url, timeout=3)
                if response.status_code == 200:
                    healthy_servers.append(f"http://{address}")
                else:
                    logging.warning(f"警告: 服务器 {address} 健康检查失败 (状态码: {response.status_code})。")
            except requests.exceptions.RequestException as e:
                # 忽略连接失败的服务器
                logging.warning(f"警告: 无法连接到服务器 {address} 进行健康检查: {e}")
                continue
        
        if not healthy_servers:
            raise ConnectionError("在 'hamgnn' 服务器列表中，没有找到任何健康且可访问的服务器。")

        # 3. 查询每个健康服务器的负载
        server_loads: Dict[str, float] = {}
        for server_url in healthy_servers:
            try:
                load_status_url = f"{server_url}/load_status"
                response = requests.get(load_status_url, timeout=5)
                # 确保请求成功 (状态码 2xx)
                response.raise_for_status()
                load_value = response.json().get('load_factor')
                if load_value is not None:
                    server_loads[server_url] = float(load_value)
                else:
                    logging.warning(f"警告: 服务器 {server_url} 的负载信息格式不正确。")

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
                logging.warning(f"警告: 无法从 {server_url} 获取或解析负载信息: {e}")
                continue

        if not server_loads:
            raise ConnectionError("无法从任何健康的 'hamgnn' 服务器获取负载信息。")

        # 4. 返回负载最小的服务器 URL
        min_load_url = min(server_loads, key=server_loads.get)
        logging.info(f"发现 {len(server_loads)} 个健康服务器。选择的最低负载服务器: {min_load_url} (负载: {server_loads[min_load_url]})")
        return min_load_url
    else:
        # --- 其他服务器类型的原始逻辑 ---
        with open(server_info_file, 'r') as f:
            info = json.load(f)
        return f"http://{info['host']}:{info['port']}"