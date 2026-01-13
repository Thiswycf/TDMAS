"""
多智能体系统核心实现
实现递归式的提问、分解、回复机制
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from metagpt.logs import logger
from utils.llm_manager import get_global_llm_manager
from vllm import SamplingParams
from .prompts import format_first_question_prompt, format_reply_prompt, format_non_first_question_prompt
from .parser import parse_agent_response, extract_all_scores


class MultiAgentSystem:
    """多智能体系统类，实现递归式的任务分解和回答机制"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_output_tokens: int = 2048):
        """
        Args:
            model_name: LLM模型名称
            temperature: 采样温度
            max_output_tokens: 最大输出token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_output_tokens,
        )
        
        # 用于跟踪token消耗
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # 用于跟踪所有打分
        self.all_scores = []
        
        # 用于跟踪对话历史
        self.conversation_history = []
    
    async def _call_llm(self, prompt: str) -> Tuple[str, int, int]:
        """调用LLM并返回响应和token消耗
        
        Returns:
            (response_text, input_tokens, output_tokens)
        """
        output = await get_global_llm_manager().generate(
            self.model_name,
            prompt,
            sampling_params=self.sampling_params
        )
        
        response_text = output.outputs[0].text
        
        # 从output中提取token信息（如果可用）
        # vllm的output对象通常有prompt_token_ids属性，但可能不在output对象上
        # 使用简单的估算方法
        try:
            # 尝试从output获取token信息
            if hasattr(output, 'prompt_token_ids') and output.prompt_token_ids:
                input_tokens_count = len(output.prompt_token_ids) if isinstance(output.prompt_token_ids, list) else 0
            else:
                # 估算：中文字符和英文单词的平均token数约为1.3
                input_tokens_count = int(len(prompt.split()) * 1.3)
        except:
            input_tokens_count = int(len(prompt.split()) * 1.3)
        
        try:
            # 尝试从output.outputs获取token信息
            if hasattr(output.outputs[0], 'token_ids') and output.outputs[0].token_ids:
                output_tokens_count = len(output.outputs[0].token_ids) if isinstance(output.outputs[0].token_ids, list) else 0
            else:
                output_tokens_count = int(len(response_text.split()) * 1.3)
        except:
            output_tokens_count = int(len(response_text.split()) * 1.3)
        
        self.total_input_tokens += input_tokens_count
        self.total_output_tokens += output_tokens_count
        
        return response_text, int(input_tokens_count), int(output_tokens_count)
    
    async def solve_subquestion(self, question: str, is_first: bool = False, previous_reply: Optional[Dict] = None) -> Dict:
        """解决一个子问题（员工智能体的角色）
        
        Args:
            question: 子问题
            is_first: 是否是首次提问
            previous_reply: 之前的回复（如果是非首次提问）
        
        Returns:
            包含回答、打分、评价的字典
        """
        if is_first:
            prompt = format_first_question_prompt(question)
        else:
            prompt = format_non_first_question_prompt(question, previous_reply or {})
        
        response_text, input_tokens, output_tokens = await self._call_llm(prompt)
        parsed_response = parse_agent_response(response_text)
        
        # 记录对话历史
        self.conversation_history.append({
            'type': 'subquestion',
            'question': question,
            'prompt': prompt,
            'response': response_text,
            'parsed': parsed_response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        })
        
        result = {
            'question': question,
            'answer': parsed_response.get('answer', ''),
            'score': parsed_response.get('score'),
            'evaluation': parsed_response.get('evaluation', ''),
            'type': parsed_response.get('type'),
            'subquestions': parsed_response.get('subquestions', []),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
        
        # 收集打分
        if parsed_response.get('score') is not None:
            self.all_scores.append(parsed_response['score'])
        
        return result
    
    async def solve_problem_recursive(
        self,
        question: str,
        max_depth: int = 5,
        current_depth: int = 0,
        is_first: bool = True,
        previous_reply: Optional[Dict] = None
    ) -> Dict:
        """递归解决一个问题（领导智能体的角色）
        
        Args:
            question: 问题
            max_depth: 最大递归深度
            current_depth: 当前深度
            is_first: 是否是首次提问
            previous_reply: 之前的回复（如果是非首次提问）
        
        Returns:
            包含最终答案、所有打分、token消耗等的字典
        """
        if current_depth >= max_depth:
            logger.warning(f"达到最大递归深度 {max_depth}，停止递归")
            return {
                'answer': '',
                'final': True,
                'reason': 'max_depth_reached',
                'all_scores': self.all_scores.copy(),
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens
            }
        
        # 领导智能体决定是直接回答还是分解
        if is_first:
            prompt = format_first_question_prompt(question)
        else:
            prompt = format_non_first_question_prompt(question, previous_reply or {})
        
        response_text, input_tokens, output_tokens = await self._call_llm(prompt)
        parsed_response = parse_agent_response(response_text)
        
        # 记录对话历史
        self.conversation_history.append({
            'type': 'leader',
            'question': question,
            'prompt': prompt,
            'response': response_text,
            'parsed': parsed_response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'depth': current_depth
        })
        
        # 收集打分
        if parsed_response.get('score') is not None:
            self.all_scores.append(parsed_response['score'])
        
        # 如果直接回答了，返回答案
        if parsed_response.get('type') == 'answer':
            return {
                'answer': parsed_response.get('answer', ''),
                'final': True,
                'reason': 'direct_answer',
                'all_scores': self.all_scores.copy(),
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'conversation_history': self.conversation_history.copy()
            }
        
        # 如果需要分解为子问题
        if parsed_response.get('type') == 'subquestions' and parsed_response.get('subquestions'):
            subquestions = parsed_response['subquestions']
            logger.info(f"问题被分解为 {len(subquestions)} 个子问题（深度 {current_depth}）")
            
            # 员工智能体回答子问题
            subquestion_tasks = []
            for subq in subquestions:
                task = self.solve_subquestion(subq['question'], is_first=True)
                subquestion_tasks.append(task)
            
            subquestion_responses = await asyncio.gather(*subquestion_tasks)
            
            # 格式化为回复提示词需要的格式
            formatted_subquestion_replies = []
            for i, (subq, resp) in enumerate(zip(subquestions, subquestion_responses)):
                formatted_subquestion_replies.append({
                    'id': subq['id'],
                    'question': subq['question'],
                    'answer': resp.get('answer', ''),
                    'score': resp.get('score'),
                    'evaluation': resp.get('evaluation', '')
                })
            
            # 领导智能体根据子问题的回复进行决策
            reply_prompt = format_reply_prompt(
                question,
                formatted_subquestion_replies,
                len(subquestions)
            )
            
            reply_response_text, reply_input_tokens, reply_output_tokens = await self._call_llm(reply_prompt)
            reply_parsed = parse_agent_response(reply_response_text)
            
            # 记录对话历史
            self.conversation_history.append({
                'type': 'leader_reply',
                'question': question,
                'prompt': reply_prompt,
                'response': reply_response_text,
                'parsed': reply_parsed,
                'input_tokens': reply_input_tokens,
                'output_tokens': reply_output_tokens,
                'depth': current_depth
            })
            
            # 收集子问题打分
            if reply_parsed.get('subquestion_scores'):
                self.all_scores.extend(reply_parsed['subquestion_scores'].values())
            
            # 如果最终回答了，返回答案
            if reply_parsed.get('type') == 'answer':
                return {
                    'answer': reply_parsed.get('answer', ''),
                    'final': True,
                    'reason': 'answered_after_subquestions',
                    'all_scores': self.all_scores.copy(),
                    'total_input_tokens': self.total_input_tokens,
                    'total_output_tokens': self.total_output_tokens,
                    'conversation_history': self.conversation_history.copy()
                }
            
            # 如果需要继续提问，递归处理
            if reply_parsed.get('type') == 'subquestions' and reply_parsed.get('subquestions'):
                # 对于新的子问题，继续递归
                new_subquestions = reply_parsed['subquestions']
                # 这里可以选择继续递归或者返回当前状态
                # 为了简化，我们选择继续递归处理第一个新子问题
                if new_subquestions:
                    next_question = new_subquestions[0]['question']
                    return await self.solve_problem_recursive(
                        next_question,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        is_first=False,
                        previous_reply={
                            'answer': formatted_subquestion_replies[0].get('answer', ''),
                            'score': formatted_subquestion_replies[0].get('score'),
                            'evaluation': formatted_subquestion_replies[0].get('evaluation', '')
                        }
                    )
        
        # 如果无法处理，返回当前状态
        return {
            'answer': '',
            'final': False,
            'reason': 'unknown_state',
            'all_scores': self.all_scores.copy(),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'conversation_history': self.conversation_history.copy()
        }
    
    def reset(self):
        """重置状态（用于处理新问题）"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.all_scores = []
        self.conversation_history = []
