import requests
import time
import json
import threading
from collections import defaultdict
import sys

# vLLM server address
API_URL = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 全局统计变量（线程安全）
lock = threading.Lock()
# 存储每个时间点的统计数据：key=时间戳(秒)，value=该秒内生成的token数
token_stats = defaultdict(int)
# 存储所有请求的TTFT
all_ttft = []
# 记录测试开始时间（作为时间轴的0点）
test_start_time = None
# 控制采样线程的开关（全局变量）
sampling_running = True

def send_request(request_id):
    """发送单个请求，并统计token生成和TTFT"""
    global test_start_time
    
    # 请求数据
    data = {
        "model": "/root/.cache/huggingface/Qwen3-4B-quantized.w4a16",
        "messages": [
            {"role": "user", "content": "Please provide a detailed introduction to the main features of Jetson Orin NX."}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True
    }

    # 记录该请求的开始时间
    req_start_time = time.time()
    first_token_time = None
    response_tokens = 0

    print(f"Request {request_id} started at +{(req_start_time - test_start_time):.2f}s")
    
    try:
        with requests.post(API_URL, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_lines():
                    if chunk:
                        chunk_str = chunk.decode('utf-8')
                        if chunk_str.startswith('data: '):
                            content = chunk_str[6:]
                            if content.strip() == '[DONE]':
                                break
                            try:
                                payload = json.loads(content)
                                # 记录首个token时间
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    with lock:
                                        all_ttft.append((first_token_time - req_start_time) * 1000)
                                
                                # 统计token并记录生成时间
                                if 'choices' in payload and payload['choices'][0]['delta'].get('content'):
                                    response_tokens += 1
                                    # 计算该token生成时相对于测试开始的秒数（取整）
                                    token_time = int(time.time() - test_start_time)
                                    with lock:
                                        token_stats[token_time] += 1
                                        
                            except json.JSONDecodeError:
                                print(f"\nRequest {request_id} - Unable to parse JSON: {content}")
            else:
                print(f"Request {request_id} failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Request {request_id} error: {str(e)}")
    
    req_end_time = time.time()
    print(f"Request {request_id} finished at +{(req_end_time - test_start_time):.2f}s, total tokens: {response_tokens}")

def sampling_thread():
    """每秒采样并打印当前吞吐量"""
    global sampling_running
    print("\n--- Sampling Started (tokens/sec) ---")
    print("Time(s) | Throughput")
    print("---------------------")
    
    while sampling_running:
        current_second = int(time.time() - test_start_time)
        # 等待到下一秒开始
        time.sleep(1.0 - (time.time() % 1.0))
        
        with lock:
            throughput = token_stats.get(current_second, 0)
        print(f"{current_second:6d} | {throughput:10d}")
        
        # 清空已统计的秒数数据（避免重复统计）
        with lock:
            if current_second in token_stats:
                del token_stats[current_second]

def main():
    global test_start_time, sampling_running
    
    # 初始化测试开始时间
    test_start_time = time.time()
    
    # 启动采样线程
    sampler = threading.Thread(target=sampling_thread, daemon=True)
    sampler.start()
    
    # 每隔1秒发起一个请求，共5个
    request_threads = []
    for i in range(5):
        t = threading.Thread(target=send_request, args=(i+1,))
        request_threads.append(t)
        t.start()
        if i < 4:  # 最后一个请求后不等待
            time.sleep(1)
    
    # 等待所有请求完成
    for t in request_threads:
        t.join()
    
    # 停止采样线程
    sampling_running = False
    sampler.join()
    
    # 打印最终统计结果
    print("\n\n--- Final Performance Results ---")
    if all_ttft:
        avg_ttft = sum(all_ttft) / len(all_ttft)
        print(f"Average TTFT across 5 requests: {avg_ttft:.2f} ms")
        print(f"Min TTFT: {min(all_ttft):.2f} ms | Max TTFT: {max(all_ttft):.2f} ms")
    
    # 计算总token和总耗时
    total_tokens = sum(token_stats.values())
    total_time = time.time() - test_start_time
    overall_throughput = total_tokens / total_time if total_time > 0 else 0
    print(f"Total Generated Tokens: {total_tokens}")
    print(f"Total Test Time: {total_time:.2f} s")
    print(f"Overall Throughput: {overall_throughput:.2f} tokens/sec")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 直接修改全局变量，无需重复声明global
        sampling_running = False
        print("\nTest interrupted by user")
        sys.exit(0)
