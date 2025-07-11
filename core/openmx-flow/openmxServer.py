import yaml
import logging
import time
import random
import subprocess
from pathlib import Path
import os
import json
import flask
import socket
import traceback
import warnings
import numpy as np
from flask import Flask, request, jsonify
from waitress import serve
from ..utils import write_server_info, find_free_port, get_package_path
from .utils_openmx.graph_data_gen import graph_data_gen
from .utils_openmx.poscar2openmx import poscar_to_openmxfile
from ..communication import OpenmxCommunicator as Communicator, BaseCommunicator
import argparse
LOGGING_LEVEL= logging.INFO
OPENMX_CONFIG_PATH = get_package_path("openmx-flow/openmx_basic_config.yaml")
class OpenMXServer:
    def __init__(self, config_path=None):
        """
        初始化OpenMX服务器，包括加载配置和模型。
        这个方法对应于服务器的“设置”或“准备”阶段。
        """
        self.app = Flask(__name__)
        self.app.logger.setLevel(LOGGING_LEVEL)
        self.app.logger.info("正在初始化 OpenMX 服务器...")
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB
        self.communicator: BaseCommunicator = Communicator()
        if config_path is None:
            self.openmx_config_path = OPENMX_CONFIG_PATH
        else:
            self.openmx_config_path = Path(config_path)
            if not self.openmx_config_path.exists():
                raise FileNotFoundError(f"OpenMX配置文件不存在: {self.openmx_config_path}")
        self.app.logger.debug(f"OpenMX配置文件路径: {self.openmx_config_path}")
        self.load_basic_config()
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大请求体大小为100MB
        self._register_routes()
        self.type = "OpenMXServer"  # 服务器类型标识
        self.app.logger.info(f"OpenMX服务器配置已加载: {self.config}")
        self.app.logger.info(f"默认参数: {self.default_params}")

    def load_basic_config(self):
        """加载OpenMX配置文件。"""
        with open(self.openmx_config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config
        gsl_default = str(get_package_path("gsl"))
        openmx_postprocess_default= str(get_package_path("bin/openmx_postprocess"))
        read_openmx_default= str(get_package_path("bin/read_openmx"))
        openmx_default= str(get_package_path("bin/openmx"))
        conda_env_default = 'hamgnn'
        conda_source_default = '/ssd/app/anaconda3/etc/profile.d/conda.sh'
        openmx_module_default = 'compiler/oneAPI/2023.2.0'
        DATA_PATH_DEFAULT = get_package_path("openmx-flow/DFT_DATA19")

        gsl_path=config.get("gsl_path")
        self.gsl= gsl_default if gsl_path is None else gsl_path
        read_openmx_path = config.get("read_openmx_path")
        self.read_openmx = read_openmx_default if read_openmx_path is None else read_openmx_path
        openmx_postprocess_path = config.get("openmx_postprocess_path")
        self.openmx_postprocess = openmx_postprocess_default if openmx_postprocess_path is None else openmx_postprocess_path
        openmx_path = config.get("openmx_path")
        self.openmx = openmx_default if openmx_path is None else openmx_path
        conda_env = config.get("conda_env")
        self.conda_env = conda_env_default if conda_env is None else conda_env
        conda_source = config.get("conda_source")
        openmx_module = config.get("openmx_module", openmx_module_default)
        self.openmx_module = openmx_module if openmx_module else openmx_module_default
        self.conda_source = conda_source_default if conda_source is None else conda_source
        DATA_PATH= config.get("DATA_PATH")
        self.DATA_PATH = DATA_PATH_DEFAULT if DATA_PATH is None else DATA_PATH
        self.app.logger.info(f"gsl_path: {self.gsl}, read_openmx_path: {self.read_openmx}, "
                             f"openmx_postprocess_path: {self.openmx_postprocess}, "
                             f"openmx_path: {self.openmx}, "
                             f"conda_env: {self.conda_env}, conda_source: {self.conda_source}, "
                             f"DATA_PATH: {self.DATA_PATH}")





        self.default_params = config.get("default_input_parameters", {})
        self.app.logger.info(f"get default_params: {self.default_params}")
        self.process_config = self.default_params.copy()
    def set_workdir(self, workdir=None):
        self.workdir = Path(workdir) if workdir else Path(get_package_path(""))
        
    def set_process_config(self, process_config={}):
        """设置后处理配置。"""
        self.process_config.update(process_config)
        self.app.logger.debug(f"设置后处理配置为: {self.process_config}")
    def set_structure(self, structure,output_path=None):
        """设置结构体，当前目录默认为工作目录。"""
        self.structure = structure
        if output_path is not None and output_path != "./":
            try:
                os.mkdir(output_path)
            except FileExistsError:
                logging.warning(f"工作目录已存在: {output_path}")
            self.set_workdir(output_path)
        elif output_path == "./":
            self.set_workdir(os.path.dirname(self.structure))
        else:
            # current_path = os.path.dirname(os.path.abspath(__file__))
            current_path = get_package_path("")
            t = int(time.time())
            workdir_name = os.path.join(current_path,"temp",f"openmx_{Path(structure).stem}_{t}{random.randint(100, 999)}")
            try:
                os.makedirs(workdir_name)
            except FileExistsError:
                logging.warning(f"工作目录已存在: {workdir_name}")
            self.set_workdir(workdir_name)
        self.app.logger.info(f"设置结构体为: {self.structure}，工作目录为: {self.workdir}")
    def transform_structure(self, structure=None):
        """转换结构体为OpenMX输入文件。"""
        if structure is None:
            structure = self.structure
        if not structure:
            raise ValueError("请先设置结构体。")
        output_file = self.workdir / f"{Path(structure).stem}.dat"
        poscar_to_openmxfile(structure,
                        system_name=self.process_config.get("system_name", "SystemName"),
                        filename=output_file,
                        DosKgrid=self.process_config.get("DosKgrid", (4, 4, 4)),
                        ScfKgrid=self.process_config.get("ScfKgrid", (4, 4, 4)),
                        SpinPolarization=self.process_config.get("SpinPolarization", 'off'),
                        XcType=self.process_config.get("XcType", 'GGA-PBE'),
                        ElectronicTemperature=self.process_config.get("ElectronicTemperature", 100),
                        energycutoff=self.process_config.get("energycutoff", 150),
                        maxIter=self.process_config.get("maxIter", 1),
                        ScfCriterion=self.process_config.get("ScfCriterion", 1.0e-6)
                    )
        self.app.logger.debug(f"转换完成，OpenMX输入文件已保存到: {output_file}")
        self.openmx_input_file = output_file
        return output_file

    def _submit_job(self, script_path: str) -> str:
        """辅助方法：提交一个sbatch脚本并返回作业ID。"""
        command = ["sbatch", script_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        self.app.logger.info(f"作业提交成功，Job ID: {job_id}")
        return job_id

    def run_openmx_submit(self):
        """运行OpenMX计算。"""
        if not hasattr(self, 'openmx_input_file'):
            raise ValueError("请先转换结构体为OpenMX输入文件。")
        sbatch_script = f"""#!/bin/sh
#SBATCH --job-name=openmx_postprocess               # Job name
#SBATCH --partition={self.process_config.get("openmx_partition", "chu")}                     # Partition (queue) name [xiang;yang;xu;chu]; adjust as needed
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                         # Total MPI tasks ntasks*cpu-per-task <= [64-xiang;64-yang;96-xu;96-chu]
#SBATCH --cpus-per-task={self.process_config.get("openmx_ncpus", 4)}                 # CPUs per task (OpenMP threads)
#SBATCH --time=48:00:00                   # Wall‐time limit (HH:MM:SS)
#SBATCH --output={os.path.join(self.workdir,"openmx.%j.out")}              # STDOUT file (%j = JobID)
#SBATCH --error={os.path.join(self.workdir,"openmx.%j.err")}               # STDERR file
set -euo pipefail
module purge
source {self.conda_source}
module load {self.openmx_module}  
export LD_LIBRARY_PATH={self.gsl}/lib:$LD_LIBRARY_PATH
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job and environment info for logging/debugging
cat << EOF
====================== Job Information ======================
Job ID:           $SLURM_JOB_ID
Job Name:         $SLURM_JOB_NAME
Partition:        $SLURM_JOB_PARTITION
Total Nodes:      $SLURM_JOB_NUM_NODES
Total MPI Tasks:  $SLURM_NTASKS
CPUs per Task:    $SLURM_CPUS_PER_TASK
Node List:        $SLURM_JOB_NODELIST
OMP Threads:      $OMP_NUM_THREADS
Job Start Time:   $(date +"%Y-%m-%d %H:%M:%S")
============================================================

EOF

# Launch OpenMX (standard build)
cd {self.workdir}
mpirun -np {self.process_config.get("openmx_ncpus", 4)} {self.openmx_postprocess} {self.openmx_input_file} > {self.process_config.get("system_name","SystemName")}.std

conda run -n hamgnn python {get_package_path("openmx-flow/utils_openmx/graph_data_gen.py")} \\
        --graph_data_save_path {self.workdir / "graph_data.npz"} \\
        --dat_file_name {self.openmx_input_file} \\
        --scf_path {self.workdir} \\
        --nao_max {self.process_config.get("nao_max")} \\
        --soc_switch {self.process_config.get("soc_switch")} \\
        --dat_file_name {str(self.openmx_input_file)}
        """
        with open(self.workdir / "run_openmx.sh", 'w') as f:
            f.write(sbatch_script)
        job_id = self._submit_job(self.workdir / "run_openmx.sh")
        self.app.logger.info(f"OpenMX计算作业已提交，Job ID: {job_id}")
        return job_id
    
    def run_openmx_scf(self,gen_graph=True):
        """运行OpenMX计算。"""
        if not hasattr(self, 'openmx_input_file'):
            raise ValueError("请先转换结构体为OpenMX输入文件。")
        sbatch_script = f"""#!/bin/sh
#SBATCH --job-name=openmx_scf                # Job name
#SBATCH --partition={self.process_config.get("openmx_partition", "chu")}                     # Partition (queue) name [xiang;yang;xu;chu]; adjust as needed
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                         # Total MPI tasks ntasks*cpu-per-task <= [64-xiang;64-yang;96-xu;96-chu]
#SBATCH --cpus-per-task={self.process_config.get("openmx_ncpus", 16)}                 # CPUs per task (OpenMP threads)
#SBATCH --time=48:00:00                   # Wall‐time limit (HH:MM:SS)
#SBATCH --output={os.path.join(self.workdir,"openmx.%j.out")}              # STDOUT file (%j = JobID)
#SBATCH --error={os.path.join(self.workdir,"openmx.%j.err")}               # STDERR file
set -euo pipefail
module purge
source {self.conda_source}
module load {self.openmx_module}  
export LD_LIBRARY_PATH={self.gsl}/lib:$LD_LIBRARY_PATH
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job and environment info for logging/debugging
cat << EOF
====================== Job Information ======================
Job ID:           $SLURM_JOB_ID
Job Name:         $SLURM_JOB_NAME
Partition:        $SLURM_JOB_PARTITION
Total Nodes:      $SLURM_JOB_NUM_NODES
Total MPI Tasks:  $SLURM_NTASKS
CPUs per Task:    $SLURM_CPUS_PER_TASK
Node List:        $SLURM_JOB_NODELIST
OMP Threads:      $OMP_NUM_THREADS
Job Start Time:   $(date +"%Y-%m-%d %H:%M:%S")
============================================================

EOF

# Launch OpenMX (standard build)
cd {self.workdir}
mpirun -np {self.process_config.get("openmx_ncpus", 16)} {self.openmx} {self.openmx_input_file}  > {self.process_config.get("system_name","SystemName")}.std
mpirun -np {self.process_config.get("openmx_ncpus", 16)} {self.openmx_postprocess} {self.openmx_input_file}
        """
        if gen_graph:
            sbatch_script += f"""\n
conda run -n hamgnn python {get_package_path("openmx-flow/utils_openmx/graph_data_gen.py")} \\
        --graph_data_save_path {self.workdir / "graph_data.npz"} \\
        --dat_file_name {self.openmx_input_file}    \\
        --scf_path {self.workdir} \\
        --nao_max {self.process_config.get("nao_max")} \\
        --soc_switch {self.process_config.get("soc_switch")} \\
        --system_name {self.process_config.get("system_name","SystemName")} \\
        --ifscf True
        """
        with open(self.workdir / "run_openmx.sh", 'w') as f:
            f.write(sbatch_script)
        job_id = self._submit_job(self.workdir / "run_openmx.sh")
        self.app.logger.info(f"OpenMX计算作业已提交，Job ID: {job_id}")
        return job_id
    
    def run_openmx(self):
        """运行OpenMX计算。"""
        #FIXME:由于无法指定openmx_postprocess和read_openmx的保存目录位置，该方法暂时无法使用
        if not hasattr(self, 'openmx_input_file'):
            raise ValueError("请先转换结构体为OpenMX输入文件。")
        os.chdir(self.workdir) #FIXME: openmx_postprocess和read_openmx似乎都不支持设置保存目录位置，这对于flaskserver是不被允许的。
        os.system(f"mpirun -np {self.process_config.get('openmx_ncpus', 16)} {self.openmx_postprocess} {self.openmx_input_file}")
        input= {"graph_data_save_path": str(self.workdir / "graph_data.npz"),
        "dat_file_name": str(self.openmx_input_file),
        "scf_path": str(self.workdir),
        "nao_max": 26,
        "soc_switch": False
    }
        self.app.logger.info(f"开始生成图数据，参数: {input}")
        # 调用图数据生成函数
        graph_data_gen(
            input)
        self.app.logger.info(f"图数据生成完成，保存路径: {self.workdir / 'graph_data.npz'}")

    def _register_routes(self):
        """注册Flask路由，并将它们连接到类实例。"""
    
        @self.app.route("/health", methods=['GET'])
        def health_check():
            # 检查服务器是否存活以及模型是否已加载
            return jsonify({"status": "ok"})
        
        @self.app.route("/pre_process", methods=['POST'])
        def pre_process():
            try:
               structure, graph_para, output_path = self.communicator.unpack_request(request)
               self.set_structure(structure, output_path=output_path)
               self.app.logger.debug(f"接收到的结构体: {structure}")
               self.app.logger.debug(f"接收到的图参数: {graph_para}")
               self.set_process_config(graph_para)
               self.app.logger.debug(f"使用图参数: {self.process_config}")
               self.transform_structure()
               job_id=self.run_openmx_submit()
               return self.communicator.pack_response({"job_id": job_id, "workdir": str(self.workdir),
                                                       "process_config": self.process_config,"job_type": "post_process"})
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500

        @self.app.route("/graph", methods=['POST'])
        def gen_graph():
            try:
               structure, graph_para, output_path = self.communicator.unpack_request(request)
               self.set_structure(structure, output_path=output_path)
               self.app.logger.debug(f"接收到的结构体: {structure}")
               self.app.logger.debug(f"接收到的图参数: {graph_para}")
               self.set_process_config(graph_para)
               self.app.logger.debug(f"使用图参数: {self.process_config}")
               graph_data_gen(
                   input={
                       "graph_data_save_path": self.workdir / "graph_data.npz",
                       "dat_file_name": self.openmx_input_file,
                       "scf_path": self.workdir,
                       "nao_max": self.process_config.get("nao_max", 26),
                       "soc_switch": self.process_config.get("soc_switch", False)
                   }
               )
               self.app.logger.info(f"图数据生成完成，保存路径: {self.workdir / 'graph_data.npz'}")
               return self.communicator.pack_response({"job_id":None, "workdir": str(self.workdir),
                                                       "process_config": self.process_config,"job_type": "graph"})
            except Exception as e:
                warnings.warn(f"openmx处理过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500
            
        @self.app.route("/scf", methods=['POST'])
        def scf():
            try:
                structure, graph_para, output_path = self.communicator.unpack_request(request)
                self.set_structure(structure, output_path=output_path)
                self.app.logger.debug(f"接收到的结构体: {structure}")
                self.app.logger.debug(f"接收到的图参数: {graph_para}")
                self.set_process_config(graph_para)
                self.transform_structure()
                job_id=self.run_openmx_scf(gen_graph=self.process_config.get("gen_graph", True))

                return self.communicator.pack_response({"job_id": job_id, "workdir": str(self.workdir),
                                                        "process_config": self.process_config,"job_type": "scf"})
            except Exception as e:
                warnings.warn(f"预测过程中发生错误: {e}")
                traceback.print_exc() 
                return jsonify({"error": "服务器内部错误，请查看服务器日志了解详情。", "error_type": str(type(e).__name__)}), 500
        @self.app.route("/api", methods=['GET'])
        def api():
            return jsonify({
                "endpoints": {
                    "/health": "健康检查，返回服务器状态",
                    "/pre_process": "预处理请求，转换结构体为OpenMX输入文件并提交计算",
                    "/graph": "生成图数据请求，转换结构体为图数据",
                    "/scf": "运行OpenMX SCF计算请求"
                }
            }), 200

    def run(self,num_threads):
        """
        启动服务器，包括HPC的服务发现功能。
        这个方法对应于服务器的“运行”阶段。
        """

        info_file_path = get_package_path("server_info/openmx_server_info.json")
        host = socket.getfqdn()
        port = find_free_port()
        self.app.logger.info(f"正在启动 Flask 服务器，地址: http://{host}:{port}")
        write_server_info(host, port, self.type,info_file_path)
        self.app.logger.debug(f"服务器信息已写入: {info_file_path}")
        # 使用生产级的Waitress服务器来运行应用
        serve(self.app, host="0.0.0.0", port=port,threads=num_threads)
        

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='OpenMX Server')
    argument_parser.add_argument('--config', default=OPENMX_CONFIG_PATH, type=str, help='OpenMX配置文件路径')
    args = argument_parser.parse_args()
    openmx_server = OpenMXServer(config_path=args.config)
    num_threads = openmx_server.config.get('num_threads', 4)  # 从配置中获取线程数
    openmx_server.run(num_threads)  # 默认使用4个线程