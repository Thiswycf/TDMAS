"""
损失计算模块
计算正确性损失、一致性损失和token损失
"""

from typing import List, Optional
from metagpt.logs import logger
import asyncio

async def calculate_correctness_loss(answer: str, benchmark, problem: dict) -> float:
    """计算正确性损失
    
    Args:
        answer: MAS的回答
        benchmark: 基准测试对象
        problem: 问题字典
    
    Returns:
        正确性损失值（0-1之间，1表示完全错误，0表示完全正确）
    """
    
    try:
        # 使用benchmark的方法计算得分
        # 这里假设benchmark有evaluate_problem方法或类似的方法
        if hasattr(benchmark, 'direct_judge'):
            import inspect
            if inspect.iscoroutinefunction(benchmark.direct_judge):
                score = await benchmark.direct_judge(answer, problem)
            else:
                score = benchmark.direct_judge(answer, problem)
            # 损失 = 1 - 得分（得分越高，损失越小）
            return 1.0 - score
        else:
            # 如果没有direct_judge方法，使用简单的字符串比较
            ground_truth = problem.get('answer', '')
            gt_str = str(ground_truth).strip().lower()
            answer_str = str(answer).strip().lower()
            if answer_str == gt_str:
                return 0.0
            else:
                return 1.0
    except Exception as e:
        import traceback
        logger.warning(f"计算正确性损失时出错: {type(e).__name__}\n{traceback.format_exc()}")
        return 1.0  # 出错时返回最大损失


def calculate_consistency_loss(all_scores: List[int]) -> float:
    """计算一致性损失（所有打分之和的负值）
    
    注意：根据需求描述，损失应该包含"所有打分之和"，但为了使其成为损失（越小越好），
    我们使用负值。或者可以根据具体需求调整。
    
    Args:
        all_scores: 所有打分的列表
    
    Returns:
        一致性损失值
    """
    if not all_scores:
        return 0.0
    
    # 计算所有打分的平均值（标准化到0-1）
    # 由于打分是0-100，我们使用 (100 - 平均分) / 100 作为损失
    # 这样高分对应低损失
    avg_score = sum(all_scores) / len(all_scores)
    # 损失 = 1 - 标准化得分
    normalized_loss = 1.0 - (avg_score / 100.0)
    
    return normalized_loss


def calculate_token_loss(total_tokens: int, token_weight: float = 1e-6) -> float:
    """计算token消耗损失（成本控制）
    
    Args:
        total_tokens: 总token数（输入+输出）
        token_weight: token损失的权重系数
    
    Returns:
        token损失值
    """
    # 使用线性函数，token越多损失越大
    return total_tokens * token_weight


async def calculate_loss(
    answer: str,
    supervision_score: int,
    consistency_score: int,
    input_tokens: int,
    output_tokens: int,
    benchmark,
    problem: dict,
    supervision_weight: float = 1.0,
    consistency_weight: float = 0.1,
    token_weight: float = 1e-6
) -> dict:
    """计算总损失
    
    损失 = supervision_weight * 监督得分损失 + 
          consistency_weight * 一致性损失 + 
          token_weight * token损失
    
    Args:
        answer: MAS的回答
        supervision_score: 监督得分
        consistency_score: 一致性得分
        input_tokens: 输入token数
        output_tokens: 输出token数
        benchmark: 基准测试对象
        problem: 问题字典
        supervision_weight: 监督得分权重
        consistency_weight: 一致性得分权重
        token_weight: token权重
    
    Returns:
        包含各项损失的字典
    """
    correctness_loss = await calculate_correctness_loss(answer, benchmark, problem)
    consistency_loss = calculate_consistency_loss(all_scores)
    token_loss = calculate_token_loss(total_tokens, token_weight)
    
    total_loss = (
        correctness_weight * correctness_loss +
        consistency_weight * consistency_loss +
        token_loss  # token_loss已经包含了权重
    )
    
    return {
        'total_loss': total_loss,
        'correctness_loss': correctness_loss,
        'consistency_loss': consistency_loss,
        'token_loss': token_loss,
        'correctness_weight': correctness_weight,
        'consistency_weight': consistency_weight,
        'token_weight': token_weight
    }
