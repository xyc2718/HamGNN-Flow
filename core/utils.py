import os
import json
import logging
from pathlib import Path
from typing import Union
import socket
from filelock import FileLock, Timeout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

corePath=Path(__file__).parent
def find_free_port():
    """静态方法：动态查找一个未被占用的端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]
def get_package_path(path):
    return corePath / path

def write_server_info(host: str, port: int, type: str, info_file: Union[str, Path]):
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
            with open(info_file_path, 'w', encoding='utf-8') as f:
                json.dump(server_info, f, ensure_ascii=False)
                
            logging.info(f"成功获取锁，服务器信息已写入: {info_file_path}")
    except Timeout:
        logging.error(f"获取文件锁超时 ({lock_file_path})。另一个进程可能持有锁太久。")
    except Exception as e:
        logging.error(f"写入文件时发生未知错误: {e}")