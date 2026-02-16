from vllm import LLM, SamplingParams
import os

# WSL 环境适配
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # 加载 4bit 量化的 Qwen-7B-Chat，适配 10G 显存
    llm = LLM(
        model="Qwen/Qwen-7B-Chat-Int4",  # 4bit量化，显存占用 ~6GB
        trust_remote_code=True,          # 关键：允许运行Qwen的自定义代码
        gpu_memory_utilization=0.8,      # 占用 8GB 显存，留 2GB 余量
        max_num_batched_tokens=512,      # 减小批处理，降低显存峰值
        tensor_parallel_size=1,          # 单卡运行
        enforce_eager=True,              # WSL 兼容
        disable_log_stats=True,          # 减少日志开销
    )
    
    # Qwen 专用聊天格式（必须按这个格式，否则回答乱码）
    prompt = "<|im_start|>user\n用Python写一个简单的计算器，要求能加减乘除<|im_end|>\n<|im_start|>assistant\n"
    
    # 生成参数
    sampling_params = SamplingParams(
        max_tokens=300,    # 生成最长300个token
        temperature=0.7,   # 随机性
        top_p=0.85         # 采样策略
    )
    
    # 执行推理
    outputs = llm.generate(prompt, sampling_params)
    
    # 打印结果
    print("✅ Qwen-7B-Chat 生成结果：\n")
    print(outputs[0].outputs[0].text.strip())