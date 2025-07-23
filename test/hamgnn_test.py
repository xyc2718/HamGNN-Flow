import requests
import json
import os
from pathlib import Path
import numpy as np
from core.utils import get_server_url,get_package_path
import asyncio
import httpx
import json
import time
import importlib.util
test_path=Path(__file__).parent
utils_path = test_path.parent / "core" / "utils.py" 
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
get_server_url = utils_module.get_server_url


# Number of concurrent requests to send
N_REQUESTS = 128  # Change this to the number of requests you want to send
sleep_time = 0.1  # Sleep time between requests to avoid overwhelming the server


test_data_path=test_path/"test_data/C8/graph_data.npz" 
# --- 定义一个包含多个请求任务的列表 --- 
# 这里的每个字典都代表一个独立的请求
job_tickets = [
    {
        "graph_data_path": str(test_data_path),
        "output_path": None,
        "evaluate_loss": True,
    }
]*N_REQUESTS 

# ---  修改异步请求函数以接收单个任务 --- 
async def send_request(client, request_id, single_job_ticket):
    """
    发送单个POST请求到服务器并返回响应。
    现在它接收一个独立的 'single_job_ticket' 作为数据。
    """
    SERVER_URL = get_server_url("hamgnn")
    job_id = single_job_ticket.get("job_id", f"#{request_id}")
    print(f"🚀 Sending request for job '{job_id}'...")
    try:
        # 将单个任务字典作为json数据发送
        print(f"Sending request for job '{job_id}' with data: {json.dumps(single_job_ticket, indent=2)} to {SERVER_URL}/predict")
        response = await client.post(f"{SERVER_URL}/predict", json=single_job_ticket, timeout=300)
        response.raise_for_status()  # 如果状态码不是2xx，则抛出异常
        print(f"✅ Success for job '{job_id}': {response.status_code}")
        return response.json()
    except httpx.RequestError as e:
        print(f"❌ Error for job '{job_id}': {e}")
        return {"error": str(e), "failed_job": single_job_ticket}

# --- 4. 主要执行逻辑 ---
async def main():
    """
    并发发送 'job_tickets' 列表中的所有请求，等待它们完成，并打印结果。
    """
    start_time = time.time()
    num_requests = len(job_tickets) #  请求总数由列表长度决定

    async with httpx.AsyncClient() as client:
        # --- 5. 遍历job_tickets列表来创建任务 --- 
        tasks=[]
        for i, job_ticket in enumerate(job_tickets, 1):
            # 为每个请求创建一个任务
            task = asyncio.create_task(send_request(client, i, job_ticket))
            tasks.append(task)
            # 控制并发请求的速率
            await asyncio.sleep(sleep_time)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / num_requests if num_requests > 0 else 0

    # --- 打印结果 ---
    print("\n--- All Responses Received ---")
    for i, result in enumerate(results, 1):
        job_id = job_tickets[i-1].get("job_id", f"#{i}")
        print(f"\n--- Result for Job '{job_id}' ---")
        print(json.dumps(result, indent=2))
        print("-" * 35)

    print("\n--- Test Summary ---")
    print(f"Total requests sent: {num_requests}")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Average response time: {average_time:.4f} seconds")

# --- 运行主异步函数 ---
if __name__ == "__main__":
    # 确保job_tickets不为空
    if not job_tickets:
        print("Job tickets list is empty. Nothing to send.")
    else:
        asyncio.run(main())