"""
解析智能体输出的模块
从格式化回答中提取答案/代码/子问题列表、打分和反馈
"""

import re
from typing import Dict, List, Optional, Tuple


def parse_agent_response(response_text: str) -> Dict:
    """解析智能体的格式化回答

    Returns:
        dict包含以下键：
        - 'type': 'answer' 或 'subquestions'
        - 'answer': 答案或代码（如果type是'answer'）
        - 'subquestions': 子问题列表（如果type是'subquestions'）
        - 'score': 打分（0-100）
        - 'evaluation': 评价文本
        - 'subquestion_scores': 子问题打分字典（如果有）
        - 'subquestion_evaluations': 子问题评价字典（如果有）
    """
    result = {
        'type': None,
        'answer': None,
        'subquestions': [],
        'score': None,
        'evaluation': None,
        'subquestion_scores': {},
        'subquestion_evaluations': {}
    }

    # 提取答案
    answer_match = re.search(r'<answer>(.*?)</answer>',
                             response_text, re.DOTALL)
    if answer_match:
        result['type'] = 'answer'
        result['answer'] = answer_match.group(1).strip()

    # 提取子问题
    subquestions_match = re.search(
        r'<subquestions>(.*?)</subquestions>', response_text, re.DOTALL)
    if subquestions_match:
        result['type'] = 'subquestions'
        subquestions_content = subquestions_match.group(1)
        # 提取每个子问题
        subquestion_pattern = r'<subquestion id="(\d+)">(.*?)</subquestion>'
        subquestion_matches = re.findall(
            subquestion_pattern, subquestions_content, re.DOTALL)
        for sub_id, sub_content in subquestion_matches:
            result['subquestions'].append({
                'id': int(sub_id),
                'question': sub_content.strip()
            })

    # 提取打分
    score_match = re.search(r'<score>(\d+)</score>', response_text)
    if score_match:
        result['score'] = int(score_match.group(1))

    # 提取评价
    evaluation_match = re.search(
        r'<evaluation>(.*?)</evaluation>', response_text, re.DOTALL)
    if evaluation_match:
        result['evaluation'] = evaluation_match.group(1).strip()

    # 提取子问题打分
    subquestion_scores_match = re.search(
        r'<subquestion_scores>(.*?)</subquestion_scores>', response_text, re.DOTALL)
    if subquestion_scores_match:
        scores_content = subquestion_scores_match.group(1)
        score_pattern = r'<subquestion id="(\d+)">(\d+)</subquestion>'
        score_matches = re.findall(score_pattern, scores_content)
        for sub_id, score in score_matches:
            result['subquestion_scores'][int(sub_id)] = int(score)

    # 提取子问题评价
    subquestion_evaluations_match = re.search(
        r'<subquestion_evaluations>(.*?)</subquestion_evaluations>', response_text, re.DOTALL)
    if subquestion_evaluations_match:
        evaluations_content = subquestion_evaluations_match.group(1)
        eval_pattern = r'<subquestion id="(\d+)">(.*?)</subquestion>'
        eval_matches = re.findall(eval_pattern, evaluations_content, re.DOTALL)
        for sub_id, eval_text in eval_matches:
            result['subquestion_evaluations'][int(sub_id)] = eval_text.strip()

    return result


def is_response_truncated(response_text: str, parsed_response: Dict) -> bool:
    """判断响应是否被截断（达到最大输出限制）

    通过检查以下情况来判断：
    1. 响应文本以不完整的标签结尾（如以 '<' 开头但没有闭合）
    2. 检测到开始标签但缺少对应的结束标签
    3. 响应文本以不完整的结构结尾

    Args:
        response_text: 原始响应文本
        parsed_response: 解析后的响应字典

    Returns:
        bool: True 表示响应可能被截断，False 表示响应完整
    """
    if not response_text:
        return False

    # 去除末尾空白字符
    text = response_text.rstrip()
    if not text:
        return False

    # 1. 检查是否以不完整的标签结尾（以 '<' 开头但没有闭合）
    if text.endswith('<'):
        return True

    # 检查是否有未闭合的开始标签
    open_tags = []
    tag_pattern = r'<(/?)(\w+)(?:\s[^>]*)?>'
    for match in re.finditer(tag_pattern, text):
        is_closing = match.group(1) == '/'
        tag_name = match.group(2)

        if is_closing:
            # 找到结束标签，尝试匹配对应的开始标签
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
            # 如果标签不匹配，可能是格式错误，但不一定是截断
        else:
            # 开始标签
            open_tags.append(tag_name)

    # 如果还有未闭合的标签，可能是被截断了
    if open_tags:
        # 检查最后一个未闭合的标签是否在文本末尾附近（可能是截断）
        last_tag_pos = text.rfind(f'<{open_tags[-1]}')
        if last_tag_pos != -1 and len(text) - last_tag_pos < 100:  # 在末尾100字符内
            return True

    # 2. 基于解析结果检查：如果检测到开始标签但解析结果为空，可能是截断
    # 检查是否有 <answer> 开始标签但没有 </answer>
    has_answer_start = re.search(r'<answer>', text, re.DOTALL)
    has_answer_end = re.search(r'</answer>', text, re.DOTALL)
    if has_answer_start and not has_answer_end:
        return True

    # 检查是否有 <subquestions> 开始标签但没有 </subquestions>
    has_subquestions_start = re.search(r'<subquestions>', text, re.DOTALL)
    has_subquestions_end = re.search(r'</subquestions>', text, re.DOTALL)
    if has_subquestions_start and not has_subquestions_end:
        return True

    # 3. 如果解析结果中 type 为 None，但文本中有开始标签，可能是截断
    if parsed_response.get('type') is None:
        if has_answer_start or has_subquestions_start:
            return True

    # 4. 检查子问题结构是否完整
    if parsed_response.get('type') == 'subquestions':
        subquestions = parsed_response.get('subquestions', [])
        if subquestions:
            # 检查最后一个子问题是否有完整的结束标签
            last_subq = subquestions[-1]
            last_subq_id = last_subq.get('id')
            if last_subq_id:
                # 查找最后一个子问题的结束标签
                pattern = rf'<subquestion id="{last_subq_id}">.*?</subquestion>'
                if not re.search(pattern, text, re.DOTALL):
                    # 没有找到完整的结束标签，可能是截断
                    return True

    return False
