"""
评估模块
在测试集上验证模型性能
"""

import sys
import asyncio
from typing import List, Dict
from metagpt.logs import logger
from .mas import MultiAgentSystem
from .loss import calculate_loss
from utils.progress import run_tasks_with_progress


class Evaluator:
    """评估器，用于在测试集上验证模型性能"""

    def __init__(self, model_name: str, dataset: str, benchmark, temperature: float = 0.7):
        """
        Args:
            model_name: LLM模型名称
            dataset: 数据集名称
            benchmark: 基准测试对象
            temperature: 采样温度
        """
        self.model_name = model_name
        self.dataset = dataset
        self.benchmark = benchmark
        self.temperature = temperature

    async def evaluate_single_problem(
        self,
        problem: Dict,
        max_depth: int = 5,
        max_concurrent_execute_code: int = 128,
        max_loop: int = 5,
        max_debug_attempts: int = 2,
    ) -> Dict:
        """收集单个问题的数据

        Args:
            problem: 问题字典
            ground_truth: 正确答案（用于计算正确性损失）
            max_depth: 最大递归深度
            max_concurrent_execute_code: 最大并发执行代码数
            max_loop: 最大循环次数
            max_debug_attempts: 代码执行失败时的最大调试尝试次数
        Returns:
            包含问题、回答、损失等信息的字典
        """
        # 获取问题文本
        question_text = self.benchmark.get_graph_input_text(problem)

        # 为每个问题创建独立的MAS实例，避免并发时的状态污染
        mas = MultiAgentSystem(self.model_name,
            temperature=self.temperature,
            max_depth=max_depth,
            max_loop=max_loop,
            max_debug_attempts=max_debug_attempts,
            max_concurrent_execute_code=max_concurrent_execute_code,
        )
        result = await mas.solve_problem(question_text, self.benchmark, problem)

        # 获取问题ID
        problem_id = self.benchmark.get_problem_id(problem)

        return {
            'problem_id': problem_id,  # 添加problem_id字段便于后续分组
            'question': question_text,
            'correctness': result["correctness"],
            'total_input_tokens': result["total_input_tokens"],
            'total_output_tokens': result["total_output_tokens"],
        }

    async def evaluate_batch(
        self,
        problems: List[Dict],
        max_depth: int = 5,
        max_concurrent_request: int = 10,
        max_concurrent_execute_code: int = 128,
        max_loop: int = 5,
        test_ask_num: int = 8,
        max_debug_attempts: int = 2,
    ) -> List[Dict]:
        """批量评估问题

        Args:
            problems: 问题列表
            max_depth: 最大递归深度
            max_concurrent_request: 最大并发数
            max_concurrent_execute_code: 最大并发执行代码数
            max_loop: 最大循环次数
            test_ask_num: 每个问题重复询问的次数
            max_debug_attempts: 代码执行失败时的最大调试尝试次数

        Returns:
            评估结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent_request)

        async def evaluate_with_semaphore(problem):
            async with semaphore:
                try:
                    return await self.evaluate_single_problem(problem, max_depth, max_concurrent_execute_code, max_loop, max_debug_attempts)
                except Exception as e:
                    import traceback
                    logger.error(
                        f"评估数据时出错: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")

        tasks = [evaluate_with_semaphore(
            problem) for problem in problems for _ in range(test_ask_num)]

        # 使用通用进度工具，保持返回顺序，并在出现异常时将异常对象写入结果列表
        results = await run_tasks_with_progress(
            tasks,
            desc="Evaluation Progress",
        )

        valid_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                valid_results.append(result)

        return valid_results
