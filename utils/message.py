from pydantic import BaseModel
from typing import Dict, List, Tuple
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob

class Message():
    """消息类，用于构建执行图"""
    source_operator: BaseModel = None # 消息来源算子
    content: str = None # 消息内容，可能被 source_operator 规则处理过，不是原输出
    call_index: int = None # 消息来源算子的调用索引（用于区分同一算子的多次调用）
    logprobs_distribution: List[Dict[int, float]] = None # 每个位置的输出分布 token_id: logprob
    chosen_token_ids: List[int] = None # 最后的选择的 token_ids

    tid_logprobs: List[Tuple[str, float]] = None # 最后的选择的 token_id 及其概率对数（区别在于这个token是原输出）

    def __init__(
        self,
        source_operator: BaseModel = None,
        content: str = None,
        raw_output: RequestOutput = None,
        call_index: int = None,
    ):
        self.source_operator = source_operator
        self.content = content
        self.call_index = call_index
        
        if raw_output is not None:
            logprobs_distribution: List[Dict[int, float]] = raw_output.outputs[0].logprobs
            chosen_token_ids: List[Tuple[str, int]] = raw_output.outputs[0].token_ids
            self.logprobs_distribution = logprobs_distribution
            self.chosen_token_ids = chosen_token_ids

        if self.logprobs_distribution is not None and self.chosen_token_ids is not None:
            assert len(self.logprobs_distribution) == len(self.chosen_token_ids), f'logprobs_distribution 长度必须与 chosen_token_tid 长度相同，当前 logprobs_distribution 长度为 {len(self.logprobs_distribution)}，chosen_token_tid 长度为 {len(self.chosen_token_ids)}'
            # 转换为 token_id: logprob 格式
            self.logprobs_distribution = [{k: v.logprob for k, v in step_dist.items()} for step_dist in self.logprobs_distribution]
            self.tid_logprobs = []
            for step_dist, token_id in zip(self.logprobs_distribution, self.chosen_token_ids):
                # 构造self.tid_logprobs
                logprob = step_dist[token_id]
                self.tid_logprobs.append((token_id, logprob))
