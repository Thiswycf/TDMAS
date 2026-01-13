"""
数据收集模块
收集训练数据并计算损失
"""

import json
import importlib
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from metagpt.logs import logger
import ScoreFlow.params
from .mas import MultiAgentSystem
from .loss import calculate_loss
from .code_executor import extract_and_execute_code


class DataCollector:
    """数据收集器，用于收集训练数据和计算损失"""

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

    async def collect_single_problem(
        self,
        problem: Dict,
        ground_truth: Any = None,
        max_depth: int = 5
    ) -> Dict:
        """收集单个问题的数据

        Args:
            problem: 问题字典
            ground_truth: 正确答案（用于计算正确性损失）
            max_depth: 最大递归深度

        Returns:
            包含问题、回答、损失等信息的字典
        """
        # 获取问题文本
        question_text = self.benchmark.get_graph_input_text(problem)

        # 为每个问题创建独立的MAS实例，避免并发时的状态污染
        mas = MultiAgentSystem(self.model_name, temperature=self.temperature)
        result = await mas.solve_problem_recursive(
            question_text,
            max_depth=max_depth,
            is_first=True
        )

        # 提取答案
        answer = result.get('answer', '')

        # 如果答案是代码，尝试执行并获取结果
        executed_result, is_code = extract_and_execute_code(answer, timeout=30)
        if is_code:
            if executed_result is not None:
                # 执行成功，使用执行结果作为答案
                answer = executed_result.strip()
            else:
                # 执行失败，保留原始代码作为答案
                logger.warning(
                    f"Code execution failed, keeping original answer, problem_id: {self.benchmark.get_problem_id(problem)}")

        # 计算损失
        loss_info = calculate_loss(
            answer=answer,
            ground_truth=ground_truth,
            all_scores=result.get('all_scores', []),
            total_tokens=result.get(
                'total_input_tokens', 0) + result.get('total_output_tokens', 0),
            benchmark=self.benchmark,
            problem=problem
        )

        # 获取问题ID
        problem_id = self.benchmark.get_problem_id(problem)

        return {
            'problem': problem,
            'problem_id': problem_id,  # 添加problem_id字段便于后续分组
            'question': question_text,
            'answer': answer,
            'ground_truth': ground_truth,
            'all_scores': result.get('all_scores', []),
            'total_input_tokens': result.get('total_input_tokens', 0),
            'total_output_tokens': result.get('total_output_tokens', 0),
            'loss': loss_info['total_loss'],
            'correctness_loss': loss_info['correctness_loss'],
            'consistency_loss': loss_info['consistency_loss'],
            'token_loss': loss_info['token_loss'],
            'conversation_history': result.get('conversation_history', []),
            'final': result.get('final', False),
            'reason': result.get('reason', '')
        }

    async def collect_batch(
        self,
        problems: List[Dict],
        ground_truths: Optional[List[Any]] = None,
        max_depth: int = 5,
        max_concurrent: int = 10,
        ask_num: int = 8
    ) -> List[Dict]:
        """批量收集数据

        Args:
            problems: 问题列表
            ground_truths: 正确答案列表（可选）
            max_depth: 最大递归深度
            max_concurrent: 最大并发数
            ask_num: 每个问题重复询问的次数

        Returns:
            收集的数据列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def collect_with_semaphore(problem, gt=None):
            async with semaphore:
                try:
                    return await self.collect_single_problem(problem, gt, max_depth)
                except Exception as e:
                    import traceback
                    logger.error(f"收集数据时出错: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")

        tasks = []
        # 对每个问题重复询问 ask_num 次
        for i, problem in enumerate(problems):
            gt = ground_truths[i] if ground_truths and i < len(
                ground_truths) else None

            # 对每个问题重复询问 ask_num 次
            for _ in range(ask_num):
                tasks.append(collect_with_semaphore(problem, gt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤掉None和异常结果
        valid_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                valid_results.append(result)

        return valid_results
