"""
评估模块
在测试集上验证模型性能
"""

import json
import importlib
import asyncio
from typing import List, Dict, Any, Optional
from metagpt.logs import logger
import ScoreFlow.params
from .mas import MultiAgentSystem
from .data_collector import DataCollector


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
    
    async def evaluate_single_problem(self, problem: Dict, max_depth: int = 5) -> Dict:
        """评估单个问题
        
        Args:
            problem: 问题字典
            max_depth: 最大递归深度
        
        Returns:
            包含问题、预测答案、真实答案、得分等的字典
        """
        mas = MultiAgentSystem(self.model_name, temperature=self.temperature)
        question_text = self.benchmark.get_graph_input_text(problem)
        
        mas.reset()
        result = await mas.solve_problem_recursive(
            question_text,
            max_depth=max_depth,
            is_first=True
        )
        
        answer = result.get('answer', '')
        
        # 使用benchmark的方法评估
        if hasattr(self.benchmark, 'evaluate_problem'):
            try:
                score, predicted, expected, raw_output = await self.benchmark.evaluate_problem(
                    problem, None, None, lambda prob: mas, zcp=None
                )
                # 由于evaluate_problem可能需要graph对象，这里简化处理
                # 实际使用时可能需要调整
                return {
                    'problem': problem,
                    'question': question_text,
                    'predicted': answer,
                    'expected': problem.get('answer', ''),
                    'score': 0.0,  # 需要根据实际评估结果设置
                    'final': result.get('final', False)
                }
            except Exception as e:
                logger.warning(f"评估问题时出错: {e}")
        
        # 简单评估
        return {
            'problem': problem,
            'question': question_text,
            'predicted': answer,
            'expected': problem.get('answer', ''),
            'score': 0.0,
            'final': result.get('final', False)
        }
    
    async def evaluate_batch(
        self,
        problems: List[Dict],
        max_depth: int = 5,
        max_concurrent: int = 10
    ) -> List[Dict]:
        """批量评估问题
        
        Args:
            problems: 问题列表
            max_depth: 最大递归深度
            max_concurrent: 最大并发数
        
        Returns:
            评估结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(problem):
            async with semaphore:
                try:
                    return await self.evaluate_single_problem(problem, max_depth)
                except Exception as e:
                    logger.error(f"评估数据时出错: {e}")
                    return None
        
        tasks = [evaluate_with_semaphore(problem) for problem in problems]
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
        
        correct = sum(1 for r in results if r.get('score', 0) > 0.5)
        return correct / len(results)
