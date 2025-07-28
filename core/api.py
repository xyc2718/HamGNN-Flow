from flask import jsonify

def postprocess_api(self):
        info = {
        "server_name": "PostProcess Server for Band Calculation",
        "version": "1.0",
        "description": "提供基于哈密顿量的能带结构计算后处理服务。支持同步本地计算和异步Slurm作业提交两种模式。",
        "config_file": str(self.config_path.absolute()),
        "default_parameters": self.default_params,
        "endpoints": [
            {
                "path": "/api",
                "method": "GET",
                "description": "获取所有可用的API端点信息和使用说明。",
                "response": {
                    "content_type": "application/json",
                    "description": "包含API详细信息的JSON对象。",
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "健康检查端点，用于确认服务器是否正在运行。",
                "response": {
                    "content_type": "application/json",
                    "example": {"status": "ok"}
                }
            },
            {
                "path": "/load_status",
                "method": "GET",
                "description": "获取服务器当前的活跃请求数量。",
                "response": {
                    "content_type": "application/json",
                    "example": {"active_requests": 2}
                }
            },
            {
                "path": "/band_cal",
                "method": "POST",
                "description": "【异步】通过Slurm提交一个能带计算作业。此请求会立即返回一个作业ID，不会等待计算完成。",
                "request": {
                    "content_type": "application/json",
                    "body": {
                        "description": "包含计算所需文件路径和参数的JSON对象。",
                        "parameters": [
                            {"name": "hamiltonian_path", "type": "string", "required": True, "description": "指向哈密顿量文件（.npz 或 .npy）的绝对路径。"},
                            {"name": "graph_data_path", "type": "string", "required": True, "description": "指向原始图结构数据文件（.npz）的绝对路径。"},
                            {"name": "output_path", "type": "string", "required": False, "description": "指定用于保存所有输出（包括sbatch脚本和计算结果）的目录。如果未提供，将自动创建临时目录。"},
                            {"name": "band_para", "type": "object", "required": False, "description": "一个包含能带计算参数的字典，用于覆盖服务器配置文件中的默认值。例如: {'k_path': 'G-M-K-G', 'ncpus': 8}"}
                        ],
                        "example": {
                            "hamiltonian_path": "/path/to/material_A/hamiltonian.npz",
                            "graph_data_path": "/path/to/material_A/graph.npz",
                            "output_path": "/path/to/material_A/band_output",
                            "band_para": {
                                "partition": "yang",
                                "ncpus": 16,
                                "mem": 32
                            }
                        }
                    }
                },
                "response": {
                    "content_type": "application/json",
                    "description": "成功提交后，返回作业ID和工作目录。",
                    "example": {
                        "status": "success",
                        "job_id": "123456",
                        "workdir": "/path/to/material_A/band_output",
                        "process_config": {
                            "k_path": "G-M-K-G", "ncpus": 16, "mem": 32, "partition": "yang", 
                            "hamiltonian_path": "/path/to/material_A/hamiltonian.npz",
                            "graph_data_path": "/path/to/material_A/graph.npz",
                            "save_dir": "/path/to/material_A/band_output"
                        }
                    }
                }
            },
            {
                "path": "/band_cal_local",
                "method": "POST",
                "description": "【同步】直接在服务器上执行能带计算。这是一个阻塞式请求，会等待计算完成后再返回结果。请仅用于快速测试或小型计算。",
                "request": {
                    "content_type": "application/json",
                    "body": "请求体结构与 /band_cal 完全相同。"
                },
                "response": {
                    "content_type": "application/json",
                    "description": "成功计算后，返回状态和工作目录。",
                    "example": {
                        "status": "success",
                        "workdir": "/path/to/temp/band_material_A_1678886400123",
                        "process_config": {
                            "k_path": "G-M-K-G", "ncpus": 4, "mem": 16, "partition": "chu",
                            "hamiltonian_path": "/path/to/material_A/hamiltonian.npz",
                            "graph_data_path": "/path/to/material_A/graph.npz",
                            "save_dir": "/path/to/temp/band_material_A_1678886400123"
                        }
                    }
                }
            }
        ]
    }
        return jsonify(info)


def hamgnn_api(self):

    info = {
    "server_name": "HamGNN Prediction Server",
    "version": "1.0",
    "description": "提供基于图神经网络(GNN)的哈密顿量(Hamiltonian)预测服务。",
    "endpoints": [
        {
            "path": "/api",
            "method": "GET",
            "description": "获取所有可用的API端点信息和使用说明。",
            "request": None,
            "response": {
                "content_type": "application/json",
                "description": "包含API详细信息的JSON对象。",
                "example": "你正在查看的内容"
            }
        },
        {
            "path": "/health",
            "method": "GET",
            "description": "健康检查端点，用于确认服务器是否正在运行以及模型是否成功加载。",
            "request": None,
            "response": {
                "content_type": "application/json",
                "description": "返回服务器和模型的状态。",
                "example": {
                    "status": "ok",
                    "model_loaded": True
                }
            }
        },
        {
            "path": "/load_status",
            "method": "GET",
            "description": "获取服务器当前的负载状态。",
            "request": None,
            "response": {
                "content_type": "application/json",
                "description": "返回当前活跃请求数、最大容量和负载因子。",
                "example": {
                    "active_requests": 1,
                    "max_capacity": 10,
                    "load_factor": 0.1
                }
            }
        },
        {
            "path": "/predict",
            "method": "POST",
            "description": "核心预测端点。接收图结构数据，返回预测的哈密顿量。",
            "request": {
                "content_type": "application/json",
                "body": {
                    "description": "包含图数据路径和其他可选参数的JSON对象。",
                    "parameters": [
                        {
                            "name": "graph_data_path",
                            "type": "string",
                            "required": True,
                            "description": "指向图数据文件（如 .npz 格式）的绝对或相对路径。服务器必须有权限访问此路径。"
                        },
                        {
                            "name": "output_path",
                            "type": "string",
                            "required": False,
                            "description": "指定用于保存输出结果的目录路径。如果未提供，将自动创建临时目录。如果设为'./'，则会使用`graph_data_path`所在的目录。"
                        },
                        {
                            "name": "evaluate_loss",
                            "type": "boolean",
                            "required": False,
                            "default": False,
                            "description": "如果为true，并且输入数据中包含真实的哈密顿量，则计算并返回L1和L2损失。"
                        },
                        {
                            "name": "return_directly",
                            "type": "boolean",
                            "required": False,
                            "default": False,
                            "description": "如果为true，则将包含哈密顿量数组的完整结果直接在响应体中返回。如果为false（默认），结果将保存到`output_path`指定的文件中，响应体只包含文件路径信息。"
                        }
                    ],
                    "example": {
                        "graph_data_path": "/path/to/your/graph_data.npz",
                        "output_path": "/path/to/your/output_dir",
                        "evaluate_loss": True,
                        "return_directly": True
                    }
                }
            },
            "response": {
                "content_type": "application/json",
                "description": "响应可以是直接的结果，也可以是结果文件的路径，具体取决于`return_directly`参数。",
                "example_direct_return": {
                    "hamiltonian": [[0.1, 0.2], [0.3, 0.4]],
                    "l1_loss": 0.05,
                    "l2_loss": 0.003,
                    "output_path": "/path/to/your/output_dir",
                    "return_directly": True
                },
                "example_file_return": {
                    "message": "Result saved to /path/to/your/output_dir/result.json",
                    "output_path": "/path/to/your/output_dir"
                }
            }
        }
    ]
}
    return jsonify(info)


def openmx_api(self):
    """
    返回API的基本信息和使用说明。
    """
    info = {
        "server_name": "OpenMX Calculation Server",
        "version": "1.0",
        "description": "提供一个接口，用于通过Slurm作业调度系统提交和管理OpenMX DFT计算。所有计算任务都是异步执行的。",
        "config_file": str(self.openmx_config_path.absolute()),
        "default_parameters": self.default_params,
        "endpoints": [
            {
                "path": "/api",
                "method": "GET",
                "description": "获取所有可用的API端点信息、服务器配置和使用说明。",
                "response": {
                    "content_type": "application/json",
                    "description": "包含API详细信息的JSON对象。",
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "健康检查端点，用于确认服务器是否正在运行。",
                "response": {
                    "content_type": "application/json",
                    "example": {"status": "ok"}
                }
            },
            {
                "path": "/scf",
                "method": "POST",
                "description": "【异步】提交一个完整的自洽场（SCF）计算作业。此端点用于从头开始进行标准的DFT计算。",
                "request": {
                    "content_type": "application/json",
                    "body": {
                        "description": "包含晶体结构文件路径和可选计算参数的JSON对象。",
                        "parameters": [
                            {"name": "structure", "type": "string", "required": True, "description": "指向晶体结构文件（如VASP POSCAR格式）的绝对路径。服务器必须能访问此文件。"},
                            {"name": "output_path", "type": "string", "required": False, "description": "指定用于保存所有输出（如 .dat 输入文件, .std 输出日志, sbatch脚本和计算结果）的目录。如果未提供，将自动创建唯一的临时目录。"},
                            {"name": "graph_para", "type": "object", "required": False, "description": "一个包含计算参数的字典，用于覆盖服务器配置文件中的默认值。这些参数将用于生成OpenMX输入文件和sbatch脚本。"},
                            {"name": "graph_para.gen_graph", "type": "boolean", "required": False, "default": True, "description": "计算完成后是否自动运行脚本生成图数据（graph_data.npz）。"}
                        ],
                        "example": {
                            "structure": "/path/to/your/material.poscar",
                            "output_path": "/path/to/your/scf_run",
                            "graph_para": {
                                "maxIter": 100,
                                "ScfKgrid": [8, 8, 8],
                                "energycutoff": 200,
                                "partition": "chu",
                                "ncpus": 32,
                                "gen_graph": True
                            }
                        }
                    }
                },
                "response": {
                    "content_type": "application/json",
                    "description": "成功提交后，返回作业ID、工作目录、最终使用的计算参数以及作业类型。",
                    "example": {
                        "job_id": "123457",
                        "workdir": "/path/to/your/scf_run",
                        "process_config": {
                            "maxIter": 100, "ScfKgrid": [8, 8, 8], "energycutoff": 200, "partition": "chu", "ncpus": 32, "gen_graph": True,
                            "system_name": "SystemName" # ... 其他默认或计算出的参数
                        },
                        "job_type": "scf"
                    }
                }
            },
            {
                "path": "/pre_process",
                "method": "POST",
                "description": "【异步】提交一个轻量级的OpenMX计算作业，主要用于快速生成图神经网络所需的输入数据（如哈密顿量和重叠矩阵）。通常只执行一次迭代（默认 maxIter: 1）。",
                    "request": {
                    "content_type": "application/json",
                    "body": {
                        "description": "请求体结构与 /scf 类似，但通常使用更少的计算资源和迭代次数。",
                        "parameters": [
                            {"name": "structure", "type": "string", "required": True, "description": "指向晶体结构文件（如VASP POSCAR格式）的绝对路径。"},
                            {"name": "output_path", "type": "string", "required": False, "description": "指定用于保存所有输出的目录。"},
                            {"name": "graph_para", "type": "object", "required": False, "description": "计算参数字典。对于此端点，'maxIter'通常应保持为1。"}
                        ],
                        "example": {
                            "structure": "/path/to/your/material.poscar",
                            "output_path": "/path/to/your/graph_gen_run",
                            "graph_para": {
                                "maxIter": 1,
                                "ncpus": 4,
                                "nao_max": 26,      # GNN图生成所需参数
                                "soc_switch": False   # GNN图生成所需参数
                            }
                        }
                    }
                },
                "response": {
                    "content_type": "application/json",
                    "description": "成功提交后，返回作业ID、工作目录、最终使用的计算参数以及作业类型。",
                    "example": {
                        "job_id": "123458",
                        "workdir": "/path/to/your/graph_gen_run",
                        "process_config": {
                            "maxIter": 1, "ncpus": 4, "nao_max": 26, "soc_switch": False,
                            "system_name": "SystemName" # ... 其他默认或计算出的参数
                        },
                        "job_type": "post_process"
                    }
                }
            }
        ]
    }
    return jsonify(info)

def orchestrator_api(self):

    info = {
        "server_name": "HamGNN Workflow Orchestrator",
        "version": "1.0",
        "description": "工作流调度服务器。这是所有计算任务的统一入口。它接收请求，将其放入分布式任务队列（Celery），并返回一个任务ID用于状态追踪。",
        "endpoints": [
            {
                "path": "/api",
                "method": "GET",
                "description": "获取所有可用的API端点、默认参数和使用说明。"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "健康检查端点，用于确认服务器是否正在运行。",
                "response": {"example": {"status": "ok"}}
            },
            {
                "path": "/submit",
                "method": "POST",
                "description": "提交一个新的、完整的计算工作流。这是一个异步请求，服务器将立即返回一个任务ID用于后续状态查询。",
                "request": {
                    "content_type": "application/json",
                    "body": {
                        "parameters": [
                            {"name": "structure", "type": "string", "required": True, "description": "指向初始晶体结构文件（如POSCAR）的绝对路径。"},
                            {"name": "config", "type": "object", "required": False, "description": "一个扁平化的字典，包含覆盖默认配置的参数。所有工作流阶段（OpenMX, HamGNN, PostProcess）的参数都直接放在这个对象里。"},
                            {"name": "config.output_path", "type": "string", "required": False, "description": "【通用】指定用于保存所有阶段结果的根目录。如果未提供(null)，系统将在temp目录下自动创建一个唯一的文件夹。"},
                            {"name": "config.partition", "type": "string", "required": False, "description": "【OpenMX/后处理】指定Slurm分区。特别地，设为 'auto' 时系统将根据当前负载自动选择最优分区。"},
                            {"name": "config.ncpus", "type": "integer", "required": False, "description": "【OpenMX/后处理】为Slurm作业申请的CPU核心数。"},
                            {"name": "config.mem", "type": "integer", "required": False, "description": "【OpenMX/后处理】为Slurm作业申请的内存大小（GB）。"},
                            {"name": "config.ifscf", "type": "boolean", "required": False, "description": "【OpenMX】是否执行自洽场计算。默认为False。"},
                            {"name": "config.evaluate_loss", "type": "boolean", "required": False, "description": "【HamGNN】在模型推理时，如果输入数据包含真值，是否计算并保存L1/L2损失。"},
                            {"name": "config.other_params_of_band_cal_or_postprocess", "description": "若需要修改各阶段的默认参数也可以直接放在config中。"},
                        ],
                        "example": {
                            "structure": "/path/to/si.poscar",
                            "config": {
                                "output_path": "/user/home/my_si_calculation",
                                "partition": "auto",
                                "ncpus": 32,
                                "maxIter": 100,
                                "evaluate_loss": True,
                                "ifscf": True,
                                "nk":120,
                                "save_fig": True
                            }
                        }
                    }
                },
                "response": {
                    "status_code": 202,
                    "description": "请求已被受理。响应中包含用于查询状态的任务ID和URL。",
                    "example": {
                        "message": "工作流已受理,正在后台排队处理",
                        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                        "status_url": "http://<server_address>:<port>/status/a1b2c3d4-e5f6-7890-1234-567890abcdef"
                    }
                }
            },
            {
                "path": "/status/<task_id>",
                "method": "GET",
                "description": "根据任务ID查询特定工作流的当前状态和进度。建议客户端轮询此端点以获取更新。",
                "response": {
                    "description": "返回一个包含详细状态信息的JSON对象。",
                    "possible_states": ["PENDING", "PROGRESS", "SUCCESS", "FAILURE", "RETRY"],
                    "example_in_progress": {
                        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                        "state": "PROGRESS", # Celery 状态
                        "info": {"current_step": "Running OpenMX SCF", "progress": 25}, # 任务自定义信息
                        "queue": "openmx_waiting_queue", # 当前所处的自定义队列
                        "queue_status": "running", # 在队列中的细分状态
                        "details": { # 队列中存储的完整任务数据
                            "structure_file_path": "/path/to/si.poscar",
                            "status": "running",
                            "job_id": "slurm_job_54321"
                        }
                    },
                    "example_success": {
                        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                        "state": "SUCCESS",
                        "info": {"final_output_path": "/user/home/my_si_calculation/final_results"},
                        "queue": "completed_queue",
                        "queue_status": "finished",
                        "details": {
                                "final_output_path": "/user/home/my_si_calculation/final_results",
                                "status": "finished"
                        }
                    }
                }
            },
            {
                "path": "/queue_stats",
                "method": "GET",
                "description": "获取系统中所有任务队列的统计信息。主要用于监控和管理。",
                "response": {
                    "description": "返回每个处理阶段的排队任务数和正在运行的任务数。",
                    "example": {
                        "pending_queue": 5,
                        "openmx_waiting_queue": 2,
                        "hamgnn_waiting_queue": 0,
                        "postprocess_waiting_queue": 0,
                        "completed_queue": 150,
                        "running_openmx_jobs": 1,
                        "running_hamgnn_jobs": 0,
                        "running_postprocess_jobs": 0
                    }
                }
            }
        ]
    }
    return jsonify(info)