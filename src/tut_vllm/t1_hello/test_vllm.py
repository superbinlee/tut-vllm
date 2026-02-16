from vllm import LLM, SamplingParams
import os

# 关键：禁用多进程，强制单进程运行
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # 极致精简配置，适配WSL+3080
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.7,    # 进一步降低显存占用（7GB），留足余量
        max_num_batched_tokens=512,    # 最小批处理大小
        tensor_parallel_size=1,        # 强制单卡运行
        enforce_eager=True,            # 禁用CUDA图优化，提升WSL兼容性
        disable_log_stats=True,        # 关闭统计日志，减少进程开销
    )
    # 最简单的生成参数
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    # 执行推理
    outputs = llm.generate("Hello", sampling_params)
    # 打印结果
    print("✅ vLLM 运行成功！")
    print("生成结果：", outputs[0].outputs[0].text.strip())