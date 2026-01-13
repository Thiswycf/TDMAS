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
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        result['type'] = 'answer'
        result['answer'] = answer_match.group(1).strip()
    
    # 提取子问题
    subquestions_match = re.search(r'<subquestions>(.*?)</subquestions>', response_text, re.DOTALL)
    if subquestions_match:
        result['type'] = 'subquestions'
        subquestions_content = subquestions_match.group(1)
        # 提取每个子问题
        subquestion_pattern = r'<subquestion id="(\d+)">(.*?)</subquestion>'
        subquestion_matches = re.findall(subquestion_pattern, subquestions_content, re.DOTALL)
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
    evaluation_match = re.search(r'<evaluation>(.*?)</evaluation>', response_text, re.DOTALL)
    if evaluation_match:
        result['evaluation'] = evaluation_match.group(1).strip()
    
    # 提取子问题打分
    subquestion_scores_match = re.search(r'<subquestion_scores>(.*?)</subquestion_scores>', response_text, re.DOTALL)
    if subquestion_scores_match:
        scores_content = subquestion_scores_match.group(1)
        score_pattern = r'<subquestion id="(\d+)">(\d+)</subquestion>'
        score_matches = re.findall(score_pattern, scores_content)
        for sub_id, score in score_matches:
            result['subquestion_scores'][int(sub_id)] = int(score)
    
    # 提取子问题评价
    subquestion_evaluations_match = re.search(r'<subquestion_evaluations>(.*?)</subquestion_evaluations>', response_text, re.DOTALL)
    if subquestion_evaluations_match:
        evaluations_content = subquestion_evaluations_match.group(1)
        eval_pattern = r'<subquestion id="(\d+)">(.*?)</subquestion>'
        eval_matches = re.findall(eval_pattern, evaluations_content, re.DOTALL)
        for sub_id, eval_text in eval_matches:
            result['subquestion_evaluations'][int(sub_id)] = eval_text.strip()
    
    return result


def extract_all_scores(parsed_response: Dict, subquestion_responses: List[Dict] = None) -> List[int]:
    """提取所有打分（包括当前问题和子问题的打分）
    
    Args:
        parsed_response: 解析后的响应
        subquestion_responses: 子问题的响应列表（可选）
    
    Returns:
        所有打分的列表
    """
    scores = []
    
    # 添加当前问题的打分
    if parsed_response.get('score') is not None:
        scores.append(parsed_response['score'])
    
    # 添加子问题打分
    if parsed_response.get('subquestion_scores'):
        scores.extend(parsed_response['subquestion_scores'].values())
    
    # 添加子问题响应中的打分
    if subquestion_responses:
        for sub_resp in subquestion_responses:
            if sub_resp.get('score') is not None:
                scores.append(sub_resp['score'])
    
    return scores
