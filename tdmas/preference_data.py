"""
生成DPO训练所需的preference data
基于收集的数据和损失值生成偏好对
"""

import pickle
import random
from typing import List, Dict, Any, Optional, Callable
from difflib import SequenceMatcher
import numpy as np
from metagpt.logs import logger
from utils.path_utils import (
    ensure_dir,
    preference_data_dir,
    preference_data_file,
)


def similarity_ratio(str1: str, str2: str) -> float:
    """计算两个字符串的相似度

    Args:
        str1: 第一个字符串
        str2: 第二个字符串

    Returns:
        相似度值，范围在0-1之间
    """
    return SequenceMatcher(None, str1, str2).ratio()


def sample_data(p_r_data: List[Dict], N: int, f: Callable[[float, float], float]) -> List[Dict]:
    prob_list = []
    for i in range(len(p_r_data)):
        w = p_r_data[i]["chosen_score"]
        l = p_r_data[i]["rejected_score"]
        prob_list.append(f(w, l))
    sum_prob = sum(prob_list)
    prob_list = [p/sum_prob for p in prob_list]
    sampled_elements = np.random.choice(
        p_r_data, size=N, replace=True, p=prob_list)
    return sampled_elements.tolist()


def generate_preference_pairs(collected_data: List[Dict], similarity_threshold: Optional[float] = None, preference_pairs_limit: Optional[int] = None) -> List[Dict]:
    """从收集的数据中生成偏好对

    对于每个问题，根据损失值生成偏好对：
    - 损失较小的回答作为chosen
    - 损失较大的回答作为rejected

    Args:
        collected_data: 收集的数据列表，每个元素包含问题、回答、损失等信息
        similarity_threshold: 相似度阈值，如果chosen和rejected的相似度>=该阈值，则跳过该样本对。如果为None，则不进行相似度检查

    Returns:
        preference pairs列表，格式为DPO训练所需的格式
    """
    preference_pairs = []

    # 按问题分组
    problem_groups = {}
    for data in collected_data:
        # 优先使用data中的problem_id字段（如果存在）
        problem_id = data.get('problem_id')
        if problem_id is None:
            # 如果没有problem_id，尝试从problem字典中获取id
            problem_id = data.get('problem', {}).get('id')
        if problem_id is None:
            # 如果仍然没有，使用问题文本作为ID
            problem_id = data.get('question', id(data))

        if problem_id not in problem_groups:
            problem_groups[problem_id] = []
        problem_groups[problem_id].append(data)

    # 为每个问题生成偏好对
    for problem_id, problem_data_list in problem_groups.items():
        if len(problem_data_list) < 2:
            continue  # 需要至少两个回答才能生成偏好对

        # 按损失排序
        sorted_data = sorted(
            problem_data_list, key=lambda x: x.get('loss', float('inf')))

        # 生成偏好对
        skipped_pairs = 0
        for chosen_idx in range(len(sorted_data) - 1):
            for rejected_idx in range(chosen_idx + 1, len(sorted_data)):
                chosen = sorted_data[chosen_idx]
                rejected = sorted_data[rejected_idx]

                # 跳过损失相同的pair（避免无效训练）
                if chosen.get('loss', float('inf')) >= rejected.get('loss', float('inf')):
                    skipped_pairs += 1
                    continue

                # 获取原始输入输出（完整的prompt和response）
                chosen_prompt, chosen_response = get_original_prompt_and_response(
                    chosen)
                rejected_prompt, rejected_response = get_original_prompt_and_response(
                    rejected)

                # 使用chosen的prompt作为统一的prompt（因为它们对应同一个问题）
                prompt = chosen_prompt

                # 跳过响应内容相同的pair（这会导致DPO训练时出现NaN）
                if chosen_response.strip() == rejected_response.strip():
                    # logger.warning(
                    #     f"Skipping pair with identical responses for problem_id: {problem_id}")
                    skipped_pairs += 1
                    continue

                # 跳过空响应
                if not chosen_response.strip() or not rejected_response.strip():
                    # logger.warning(
                    #     f"Skipping pair with empty response for problem_id: {problem_id}")
                    skipped_pairs += 1
                    continue

                # 检查相似度阈值（如果设置了）
                if similarity_threshold is not None:
                    similar_score = similarity_ratio(
                        chosen_response, rejected_response)
                    if similar_score >= similarity_threshold:
                        # 相似度过高，跳过该样本对
                        skipped_pairs += 1
                        continue

                preference_pair = {
                    'prompt': prompt,
                    'chosen': chosen_response,
                    'rejected': rejected_response,
                    'chosen_score': 1.0 - chosen.get('loss', 1.0),
                    'rejected_score': 1.0 - rejected.get('loss', 1.0),
                    'chosen_loss': chosen.get('loss', 1.0),
                    'rejected_loss': rejected.get('loss', 1.0),
                    'problem_id': problem_id
                }
                preference_pairs.append(preference_pair)

    logger.info(f"生成了 {len(preference_pairs)} 个偏好对，跳过了 {skipped_pairs} 个偏好对")

    # 对选中的pairs的chosen/rejected_score进行归一化处理（在采样之前进行）
    if preference_pairs:
        all_scores = []
        for pair in preference_pairs:
            chosen_score = pair.get('chosen_score', 0.0)
            rejected_score = pair.get('rejected_score', 0.0)
            all_scores.append(chosen_score)
            all_scores.append(rejected_score)

        min_score = min(all_scores)
        max_score = max(all_scores)
        # avoid division by zero
        score_range = max_score - min_score if max_score != min_score else 1.0

        logger.info(
            f"归一化前：score范围 [{min_score:.6f}, {max_score:.6f}], score_range={score_range:.6f}")

        for pair in preference_pairs:
            original_chosen = pair.get('chosen_score', 0.0)
            original_rejected = pair.get('rejected_score', 0.0)

            normalized_chosen = (original_chosen - min_score) / score_range
            normalized_rejected = (original_rejected - min_score) / score_range

            pair['chosen_score'] = normalized_chosen
            pair['rejected_score'] = normalized_rejected

        # 验证归一化结果
        normalized_scores = []
        for pair in preference_pairs:
            normalized_scores.append(pair.get('chosen_score', 0.0))
            normalized_scores.append(pair.get('rejected_score', 0.0))
        if normalized_scores:
            logger.info(
                f"归一化后：score范围 [{min(normalized_scores):.6f}, {max(normalized_scores):.6f}]")

    # 删除chosen_score和rejected_score相差小于0.01的pairs
    filtered_pairs = []
    threshold = 0.01
    for pair in preference_pairs:
        chosen_score = pair.get('chosen_score', 0.0)
        rejected_score = pair.get('rejected_score', 0.0)
        if abs(chosen_score - rejected_score) >= threshold:
            filtered_pairs.append(pair)
    dropped_count = len(preference_pairs) - len(filtered_pairs)
    if dropped_count > 0:
        logger.info(f"删除了 {dropped_count} 个chosen_score和rejected_score相差小于{threshold}的偏好对")
    preference_pairs = filtered_pairs

    if preference_pairs_limit is not None and len(preference_pairs) > preference_pairs_limit:
        preference_pairs = sample_data(
            preference_pairs, preference_pairs_limit, lambda x, y: (x - y)**3)  # d(x, y) = (x - y)^3
        preference_pairs = preference_pairs[:preference_pairs_limit]
        logger.info(f"由于偏好对数量超过限制，Score-DPO采样了前 {preference_pairs_limit} 个偏好对")

    return preference_pairs


def get_original_prompt_and_response(data: Dict) -> tuple[str, str]:
    """从conversation_history中提取prompt和response

    从conversation_history中提取相应的prompt和response，而不是直接将问题作为prompt，
    拼接对话历史为response。

    Args:
        data: 收集的数据字典

    Returns:
        (prompt, response) 元组，从conversation_history中提取的prompt和response
    """
    conversation_history = data.get('conversation_history', [])

    if conversation_history:
        # 从conversation_history中提取最后一个条目的prompt和response
        # 使用最后一个条目，因为它包含了完整的对话历史和最终的响应
        last_conv = conversation_history[-1]
        prompt = last_conv.get('prompt', '')
        response = last_conv.get('response', '')

        # 如果最后一个条目没有prompt或response，尝试使用其他条目
        if not prompt or not response:
            # 查找第一个有prompt和response的条目
            for conv in reversed(conversation_history):
                conv_prompt = conv.get('prompt', '')
                conv_response = conv.get('response', '')
                if conv_prompt and conv_response:
                    prompt = conv_prompt
                    response = conv_response
                    break

        # 如果仍然没有找到，使用备用方案
        if not prompt:
            prompt = data.get('question', '')
        if not response:
            response = data.get('answer', '')
    else:
        # 如果没有对话历史，使用备用方案
        prompt = data.get('question', '')
        response = data.get('answer', '')

    return prompt, response


def format_response_for_dpo(data: Dict) -> str:
    """将收集的数据格式化为DPO训练所需的响应格式

    Args:
        data: 收集的数据字典

    Returns:
        格式化后的响应文本（使用原始完整response）
    """
    _, response = get_original_prompt_and_response(data)
    return response


def save_preference_data(preference_pairs: List[Dict], dataset: str, zcp: str, epoch: int):
    """保存preference data到文件

    Args:
        preference_pairs: preference pairs列表
        dataset: 数据集名称
        zcp: ZCP名称
        epoch: epoch编号
    """
    ensure_dir(preference_data_dir(dataset, zcp))
    output_file = preference_data_file(dataset, zcp, epoch)

    with open(output_file, 'wb') as f:
        pickle.dump(preference_pairs, f)

    logger.info(f"保存了 {len(preference_pairs)} 个偏好对到 {output_file}")
