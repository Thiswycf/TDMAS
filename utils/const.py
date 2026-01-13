import subprocess
import re
# PROMPT_TEMPLATE

# Error feedback prompt templates for multi-round conversation
ERROR_FEEDBACK_HEADER = "=== Previous generation failed, please improve based on the following error information ===\n"
ERROR_FEEDBACK_ERROR_PREFIX = "Error: "
ERROR_FEEDBACK_HISTORY_HEADER = "Previous error history:\n"
ERROR_FEEDBACK_ROUND_PREFIX = "  Round {round_num}: "
ERROR_FEEDBACK_TAIL = """Please carefully analyze the above error and generate a fixed, executable workflow graph. 
Ensure the new code can solve the above problem while still meeting all original requirements.
=== End of error feedback ===
"""

# Error feedback prompt templates for concatenated format
ERROR_FEEDBACK_CONCAT_HEADER = "\n\n=== Previous generation failed, please improve based on the following error information ===\n"
ERROR_FEEDBACK_CONCAT_CODE_HEADER = "Previous generated code (for reference only, please fix the issues):\n"
ERROR_FEEDBACK_CONCAT_GRAPH_START = "<graph>\n"
ERROR_FEEDBACK_CONCAT_GRAPH_END = "</graph>\n\n"


# 最大并发任务数
MAX_CONCURRENT_TASKS = 374 * 8  # GSM8K * 5% * graph_num = 7473 * 5% * 8


def get_max_gpu_count():
    try:
        # 使用 nvidia-smi 获取 GPU 数量
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True
        )
        # 统计 GPU 列表行数
        return len(re.findall(r"^GPU \d+:", result.stdout, re.MULTILINE))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果命令失败或找不到 nvidia-smi，默认返回 0
        return 0


# 最大可用 GPU 数量
MAX_GPU_COUNT = get_max_gpu_count()

# 安全余量比例
SAFETY_MARGIN_RATIO = 0.09 if MAX_GPU_COUNT < 10 else 0.12


# TIME_LIMIT
# 等待 工作流workflow 回复单个请求的时间，由于是并行请求，所以设置为6h
TIME_FOR_GRAPH_TO_RESPONSE = 6 * 60 * 60

# 等待 算子agent 回复单个请求的时间，由于是并行请求，所以设置为60min
# Qwen3-8B 2GPU 30min 374items GSM8K
TIME_FOR_AGENT_TO_RESPONSE = 2 * 60 * 60

# Programmer算子执行代码的延迟
TIME_TO_EXEC_CODE = TIME_FOR_AGENT_TO_RESPONSE // 3

# 为了等待更多的请求来并行，等待10s
TIME_TO_WAIT_MORE_REQUEST = 10

# 初始化 agent llm 实例进程 需要的时间
TIME_FOR_PROCESS_TO_READY = 10 * 60  # 8B 5min; 14B 10min

# OOM错误重试的最大等待时间
TIME_FOR_OOM_RETRY = 5 * 60  # 5分钟

# 写入脚本后读取延迟
TIME_TO_READ_SCRIPT = 0.5


# OS related

# max open files
MAX_OPEN_FILES = 512
