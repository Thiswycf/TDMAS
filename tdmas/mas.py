"""
多智能体系统核心实现
实现递归式的提问、分解、回复机制
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from vllm import SamplingParams
import traceback

from metagpt.logs import logger
from utils.llm_manager import get_global_llm_manager
from utils.VLLMAdapter import VLLMAdapter
from .prompts import format_first_question_prompt, format_reply_prompt, format_non_first_question_prompt, format_debug_prompt
from .parser import parse_agent_response, is_response_truncated
from .code_executor import extract_python_code, extract_and_execute_code


class Agent:
    """单个智能体，维护自身对话历史与上下级关系

    一个 Agent 负责解决一个具体问题（question），在求解过程中可以：
    - 直接回答；
    - 分解为子问题并创建下级 Agent；
    - 在多轮中根据上级反馈调整提问。

    注意：真正的 LLM 调用和全局统计由 MultiAgentSystem 统一维护。
    """

    def __init__(
        self,
        mas: "MultiAgentSystem",
        question: str,
        depth: int = 0,
        parent: Optional["Agent"] = None,
        use_chat_template: bool = True,
    ) -> None:
        self.mas = mas
        self.question = question
        self.depth = depth
        self.parent = parent

        # 自身对话历史（仅与当前 question 相关）
        self.conversation_history: List[Dict[str, Any]] = []
        # 直接用于输入的对话历史（包含所有轮次）
        self.input_conversation: Union[List[Dict[str, str]], List[str]] = []
        # 下级 agent 列表（按 subquestion id 复用）
        self.children: Dict[int, "Agent"] = {}

        # 该 agent 自身的打分记录（不含子树）
        self.scores_to_subordinates: List[int] = []
        self.scores_to_superior: List[int] = []

        # 该 agent 被调用的次数
        self.turn_to_superior: int = 0

        # 是否使用聊天模板
        self.use_chat_template = use_chat_template

    def _record_turn(
        self,
        response_text: str,
        parsed: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """记录本 agent 的一轮对话，同时也追加到 MAS 的全局历史中。"""
        turn_record = {
            "agent_depth": self.depth,
            "agent_id": id(self),
            "turn": len(self.conversation_history) + 1,
            "question": self.question,
            "prompt": self.input_conversation[-1]["content"] if self.use_chat_template else self.input_conversation[-1],
            "response": response_text,
            "parsed": parsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.conversation_history.append(turn_record)
        self.mas.conversation_history.append(turn_record)
        if self.use_chat_template:
            self.input_conversation.append({"role": "assistant", "content": response_text})
        else:
            self.input_conversation.append(response_text)

    def _update_scores(self, parsed: Dict[str, Any]) -> None:
        """根据解析结果更新本 agent 及 MAS 的打分记录。"""
        scores = parsed.get("scores", [])
        if self.turn_to_superior == 1:
            self.scores_to_superior.extend(scores)
        else:
            self.scores_to_subordinates.extend(scores)
        self.mas.all_scores.extend(scores)

    def _clear_children(self) -> None:
        """释放（解散）当前 agent 的所有下级 agent。"""
        self.children.clear()

    async def _call_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int, int]:
        """代理 MAS 统一的 LLM 调用，方便未来在 Agent 级别扩展额外信息。"""
        return await self.mas._call_llm(prompt)

    async def solve(
        self,
        max_depth: int = 5,
        max_loop: int = 5,
        max_debug_attempts: int = 2,
        feedback: Optional[Tuple[float, str]] = None,
    ) -> Dict[str, Any]:
        """递归、循环地解决当前 Agent 的问题，直到返回答案或者达到最大次数。

        当本 Agent 的结果被上级使用（即 solve 调用返回给 parent）时，可认为“自己的问题已提交给上级”，
        此时上级在合适的时机可以选择不再保留该 Agent；而在 solve 内部，当本 Agent 决定将当前问题交给上级
        决策（例如：已经给出答案或子问题已经充分），会解散自己的子 Agent。
        """
        # 首次向上级汇报，之前不应有对话历史；>0: 非首次，需要历史和反馈
        self.turn_to_superior += 1
        if self.turn_to_superior == 1:
            assert not self.input_conversation and feedback is None, "For the first turn to superior, input_conversation must be empty and feedback must be None. " + f"Got input_conversation={self.input_conversation} and feedback={feedback}"
        else:
            assert self.input_conversation and feedback is not None, "For non-first turns to superior, input_conversation must not be empty and feedback must not be None. " + f"Got input_conversation={self.input_conversation} and feedback={feedback}"

        if self.depth >= max_depth:
            logger.warning(f"达到最大递归深度 {max_depth}，停止递归（depth={self.depth}）")
            return {
                "answer": "",
                "score": 0.0,
                "evaluation": "You CANNOT break down the problem any further because the maximum depth of decomposition has been reached. You CANNOT ask this subquestion any further. That means you have to answer the question given directly.",
                "final": True,
                "reason": "max_depth_reached",
            }

        # 1. 构造当前轮的首个 prompt
        # 1.1 第一次接收上级的提问
        if self.turn_to_superior == 1:
            self.input_conversation = format_first_question_prompt(self.question, use_chat_template=self.use_chat_template)
        # 1.2 非第一次接收上级的提问（需要返工）
        else:
            self.input_conversation = format_non_first_question_prompt(
                self.question,
                self.input_conversation,
                feedback,
                use_chat_template=self.use_chat_template,
            )

        turn_to_subordinates = 0
        while turn_to_subordinates < max_loop:
            turn_to_subordinates += 1
            
            response_text, input_tokens, output_tokens = await self._call_llm(self.input_conversation)
            parsed_response = parse_agent_response(response_text)

            # 记录对话与打分（自己给上级问题的打分/下级回复的打分）
            self._record_turn(response_text, parsed_response, input_tokens, output_tokens)
            self._update_scores(parsed_response)

            # 2. 若本轮直接给出答案，则跳出循环返回答案与反馈
            if parsed_response.get("type") == "answer":
                answer = parsed_response.get("answer", "")
                saved_score = parsed_response.get("score")
                saved_evaluation = parsed_response.get("evaluation")
                
                # 检查答案是否为代码
                result, error, is_code = await extract_and_execute_code(answer)
                if is_code and result is not None:
                    # 进行调试（最多 max_debug_attempts 次）
                    debug_attempts = 0
                    while error is not None and debug_attempts < max_debug_attempts:
                        last_debug_result = result  # 保存最后一次调试返回的答案
                        debug_attempts += 1
                        logger.info(f"开始第 {debug_attempts}/{max_debug_attempts} 次代码调试（depth={self.depth}）")
                        
                        # 构造调试 prompt
                        error_message = error[:1000] if error else "Unknown error"  # 限制错误信息长度
                        self.input_conversation = format_debug_prompt(
                            error_message,
                            self.input_conversation,
                            use_chat_template=self.use_chat_template,
                        )
                        
                        # 调用 LLM 获取调试后的代码
                        debug_response_text, debug_input_tokens, debug_output_tokens = await self._call_llm(self.input_conversation)
                        debug_parsed_response = parse_agent_response(debug_response_text)
                        
                        # 记录调试对话
                        self._record_turn(debug_response_text, debug_parsed_response, debug_input_tokens, debug_output_tokens)
                        
                        # 提取调试后的代码
                        answer = debug_parsed_response.get("answer", "")
                        result, error, is_code = await extract_and_execute_code(answer)
                        
                        if not is_code:
                            logger.warning(f"第 {debug_attempts} 次调试未返回有效代码，使用原始代码")
                            break
                        
                        if error is None:
                            # 调试成功，使用执行结果作为答案
                            result = result.strip() if result else answer
                            break
                        else:
                            logger.warning(f"第 {debug_attempts} 次调试后代码仍执行失败：{error[:200] if error else 'Unknown error'}")
                        
                    if error is not None:
                        # 所有调试尝试都失败，保留最后一次尝试的代码作为答案
                        logger.warning(f"经过 {debug_attempts} 次调试后代码仍无法执行（depth={self.depth}），保留最后一次尝试的代码作为答案")
                        result = last_debug_result if debug_attempts > 0 else result
                
                # 当前问题已经"提交给上级"（有了完整答案），不再需要子 agent
                self._clear_children()
                return {
                    "answer": result,
                    "score": saved_score,
                    "evaluation": saved_evaluation,
                    "final": True,
                    "reason": "direct_answer",
                }

            # 3. 若本轮选择分解为子问题，则进入子问题处理流程
            elif parsed_response.get("type") == "subquestions" and parsed_response.get("subquestions"):
                subquestions = parsed_response["subquestions"]
                logger.info(f"问题被分解为 {len(subquestions)} 个子问题（depth={self.depth}, loop={turn_to_subordinates}）")

                # 为每个子问题创建或复用下级 Agent（按 id 复用，实现追问）
                active_children: List[Agent] = []
                for subq in subquestions:
                    sub_id = subq["id"]
                    if sub_id not in self.children:
                        self.children[sub_id] = Agent(
                            self.mas,
                            subq["question"],
                            depth=self.depth + 1,
                            parent=self,
                            use_chat_template=self.use_chat_template,
                        )
                    active_children.append(self.children[sub_id])

                # 3.1 将子问题分发给下级（每个子 agent 自行用自己的循环进行多轮）
                # 为每个子问题提取对应feedback（从parsed_response中提取subquestion_scores和subquestion_evaluations，如果有的话，否则用全体的score/evaluation）
                subquestion_scores = parsed_response.get('subquestion_scores', {})
                subquestion_evaluations = parsed_response.get('subquestion_evaluations', {})

                sub_tasks = []
                for child, subq in zip(active_children, subquestions):
                    feedback = (subquestion_scores.get(subq["id"]), subquestion_evaluations.get(subq["id"]))
                    if feedback[0] is None or feedback[1] is None:
                        feedback = None
                    sub_tasks.append(
                        child.solve(
                            max_depth=max_depth,
                            max_loop=max_loop,
                            feedback=feedback,
                            max_debug_attempts=max_debug_attempts,
                        )
                    )

                sub_results = await asyncio.gather(*sub_tasks)

                # 3.2 读取子问题解答与反馈，汇总为回复提示词需要的格式
                formatted_subquestion_replies = []
                for subq, resp in zip(subquestions, sub_results):
                    formatted_subquestion_replies.append(
                        {
                            "id": subq["id"],
                            "question": subq["question"],
                            "answer": resp.get("answer", ""),
                            "score": resp.get("score"),
                            "evaluation": resp.get("evaluation", ""),
                        }
                    )

                self.input_conversation = format_reply_prompt(
                    self.question,
                    formatted_subquestion_replies,
                    len(subquestions),
                    use_chat_template=self.use_chat_template,
                )

            # 如果本轮既没有直接回答，也没有有效子问题分解，则认为状态未知
            if parsed_response.get("type") not in ["answer", "subquestions"] or not parsed_response.get("subquestions"):
                # 检查响应是否被截断
                if is_response_truncated(response_text, parsed_response):
                    if len(response_text) < 1024:
                        logger.warning(
                            f"问题在第 {turn_to_subordinates} 轮循环中未得到有效分解或回答（depth={self.depth}），因为响应被截断。响应文本长度: {len(response_text)}"
                        )
                    return {
                        "answer": "",
                        "final": False,
                        "reason": "response_truncated",
                    }
                else:
                    logger.warning(f"问题在第 {turn_to_subordinates} 轮循环中未得到有效分解或回答（depth={self.depth}）")
                self._clear_children()
                break


        # 达到最大循环次数或未知状态，返回当前状态
        return {
            "answer": "",
            "final": False,
            "reason": "unknown_state",
        }


class MultiAgentSystem:
    """多智能体系统类，实现递归式的任务分解和回答机制

    现在 MultiAgentSystem 主要负责：
    - 维护 LLM 配置与调用；
    - 统计全局 token 消耗与打分；
    - 维护整个问题级别的全局对话历史；
    - 创建并调度根 Agent。
    """
    
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
        self.all_scores: List[int] = []
        
        # 用于跟踪全局对话历史（聚合所有 Agent）
        self.conversation_history: List[Dict[str, Any]] = []

        # 是否使用聊天模板
        self.use_chat_template = VLLMAdapter.needs_chat_template(self.model_name)
        
        # 从LLMManager获取tokenizer（统一管理，避免重复创建）
        self.tokenizer = get_global_llm_manager().get_tokenizer(self.model_name)
    
    async def _call_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int, int]:
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
        
        # 使用tokenizer准确计算token数
        try:
            if self.tokenizer is not None:
                # 计算输入token数
                if isinstance(prompt, str):
                    # 单轮对话：直接编码
                    input_tokens_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                elif isinstance(prompt, list):
                    # 多轮对话：需要根据是否使用chat_template来处理
                    if self.use_chat_template:
                        # 使用chat_template处理（与VLLMAdapter中的处理方式一致）
                        kwargs = {
                            "tokenize": False,
                            "add_generation_prompt": True,
                        }
                        if "qwen3" in self.model_name.lower():
                            kwargs["enable_thinking"] = False
                        # 应用chat_template得到处理后的文本，然后编码
                        processed_prompt = self.tokenizer.apply_chat_template(
                            prompt,
                            **kwargs
                        )
                        input_tokens_count = len(self.tokenizer.encode(processed_prompt, add_special_tokens=False))
                    else:
                        # 不使用chat_template：拼接所有消息内容
                        joined = ""
                        for msg in prompt:
                            role = msg["role"].capitalize() if isinstance(msg, dict) else ""
                            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                            joined += f"{role}: {content}\n\n"
                        input_tokens_count = len(self.tokenizer.encode(joined.strip(), add_special_tokens=False))
                else:
                    raise ValueError(f"Invalid prompt type: {type(prompt)}")
                
                # 计算输出token数
                output_tokens_count = len(self.tokenizer.encode(response_text, add_special_tokens=False))
            else:
                # tokenizer未加载，使用估算方法作为后备
                logger.warning("tokenizer未加载，使用估算方法作为后备。")
                if isinstance(prompt, str):
                    input_tokens_count = int(len(prompt.split()) * 1.3)
                elif isinstance(prompt, list):
                    joined = " ".join(
                        (m.get("content", "") if isinstance(m, dict) else str(m)) for m in prompt
                    )
                    input_tokens_count = int(len(joined.split()) * 1.3)
                else:
                    raise ValueError(f"Invalid prompt type: {type(prompt)}")
                output_tokens_count = int(len(response_text.split()) * 1.3)
        except Exception as e:
            logger.error(f"计算token数失败: {e}, 使用估算方法")
            # 发生错误时使用估算方法作为后备
            if isinstance(prompt, str):
                input_tokens_count = int(len(prompt.split()) * 1.3)
            elif isinstance(prompt, list):
                joined = " ".join(
                    (m.get("content", "") if isinstance(m, dict) else str(m)) for m in prompt
                )
                input_tokens_count = int(len(joined.split()) * 1.3)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")
            output_tokens_count = int(len(response_text.split()) * 1.3)
        
        self.total_input_tokens += input_tokens_count
        self.total_output_tokens += output_tokens_count
        
        return response_text, int(input_tokens_count), int(output_tokens_count)
    
    async def solve_problem_recursive(
        self,
        question: str,
        max_depth: int = 5,
        max_loop: int = 5,
        max_debug_attempts: int = 2,
    ) -> Dict[str, Any]:
        """递归解决一个问题（对外保持原有接口），由根 Agent 完成实际工作。"""
        # 创建根 Agent
        root_agent = Agent(self, question=question, depth=0, parent=None, use_chat_template=self.use_chat_template)
        result = await root_agent.solve(max_depth=max_depth, max_loop=max_loop, max_debug_attempts=max_debug_attempts)

        # 在返回结果中补充全局统计信息
        result.update(
            {
                "all_scores": self.all_scores.copy(),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "conversation_history": self.conversation_history.copy(),
            }
        )
        return result
    
    def reset(self):
        """重置状态（用于处理新问题）"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.all_scores = []
        self.conversation_history = []
    