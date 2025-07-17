import os
from .utils import get_package_path
from pathlib import Path
import shutil
os.remove(get_package_path('server_info/hamgnn_server_info.json'))
print("已删除服务器信息文件: hamgnn_server_info.json")
os.remove(get_package_path('server_info/hamgnn_server_info.json.lock'))
print("已删除服务器信息锁文件: hamgnn_server_info.json.lock")
shutil.rmtree(get_package_path("").parent/"log/HamGNNServer")
print("已删除日志目录: log/hamgnn_server")