import os
from .utils import get_package_path
from pathlib import Path
import shutil
try:
    os.remove(get_package_path('server_info/hamgnn_server_info.json'))
    print("已删除服务器信息文件: hamgnn_server_info.json")
except:
    print("服务器信息文件不存在: hamgnn_server_info.json")
try:
    os.remove(get_package_path('server_info/hamgnn_server_info.json.lock'))
    print("已删除服务器信息锁文件: hamgnn_server_info.json.lock")
except:
    print("服务器信息锁文件不存在: hamgnn_server_info.json.lock")
try:
    shutil.rmtree(get_package_path("").parent/"log/HamGNNServer")
    print("已删除日志目录: log/HamGNNServer")
except:
    print("删除日志失败")