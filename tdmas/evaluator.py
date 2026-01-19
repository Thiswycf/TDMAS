"""
评估模块
在测试集上验证模型性能
"""

import asyncio
from typing import List, Dict
from metagpt.logs import logger
from .mas import MultiAgentSystem
from .loss import calculate_loss


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
        mas = MultiAgentSystem(self.model_name, temperature=self.temperature)
        result = await mas.solve_problem(
            question_text,
            max_depth=max_depth,
            max_loop=max_loop,
            max_debug_attempts=max_debug_attempts,
        )

        # 提取答案
        answer = result.get('answer', '')
        final = result.get('final', False)

        if not final:
            logger.warning(f"Evaluation failed, problem_id: {self.benchmark.get_problem_id(problem)}, reason: {result.get('reason', '')}")
            return None

        # 计算损失
        loss_info = await calculate_loss(
            answer=answer,
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
            'correct': 1.0 - loss_info['correctness_loss'],
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
                    logger.error(f"评估数据时出错: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        
        tasks = [evaluate_with_semaphore(problem) for problem in problems for _ in range(test_ask_num)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                valid_results.append(result)
        
        return valid_results
    
    def calculate_accuracy(self, results: List[Dict]) -> float:
        """计算准确率
        
        Args:
            results: 评估结果列表
        
        Returns:
            准确率（0-1之间）
        """
        if not results:
            return 0.0
        
        correct = sum(r.get('correct', 0) for r in results)
        total = len(results)
        return correct / total if total > 0 else 0.0
