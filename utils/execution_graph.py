#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据最终的 Message 反推多智能体系统的执行图。
节点代表产生消息的算子（operator），边代表消息在算子之间的传递。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.message import Message


class ExecutionGraph:
    """从输出 Message 构建执行图的工具类。"""

    def __init__(self, output_message: Message):
        self.output_message = output_message
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._node_index: Dict[str, Dict[str, Any]] = {}
        self._visited_messages = set()
        self._build_graph()

    def _add_node(self, operator: Any, is_external: bool = False, label: Optional[str] = None, call_index: Optional[int] = None) -> str:
        """注册节点并返回节点id。

        Args:
            operator: 算子实例
            is_external: 是否为外部输入节点
            label: 外部节点的标签
            call_index: 调用索引（用于区分同一算子的多次调用）
        """
        if is_external:
            node_id = label or f"external_{len(self._node_index)}"
            name = label or "ExternalInput"
            class_name = "ExternalInput"
        else:
            # 如果提供了调用索引，使用它来区分同一算子的多次调用
            if call_index is not None:
                node_id = f"op_{id(operator)}_call_{call_index}"
            else:
                # 向后兼容：如果没有调用索引，使用旧的逻辑
                node_id = f"op_{id(operator)}"
            name = getattr(operator, "name", operator.__class__.__name__)
            class_name = operator.__class__.__name__
            # 如果有多次调用，在名称中显示调用索引
            if call_index is not None and hasattr(operator, "call_history") and len(operator.call_history) > 1:
                name = f"{name}_call_{call_index}"

        if node_id not in self._node_index:
            node_data = {
                "id": node_id,
                "name": name,
                "class": class_name,
                "problem": getattr(operator, "problem", None) if not is_external else None,
                "fields": getattr(operator, "fields", None) if not is_external else None,
                "call_index": call_index if not is_external else None,
            }
            self._node_index[node_id] = node_data
            self.nodes.append(node_data)
        return node_id

    def _add_edge(self, source_id: str, target_id: str, message: Message):
        """记录边。"""
        preview = None
        if hasattr(message, "content") and isinstance(message.content, str):
            preview = message.content[:160]

        self.edges.append(
            {
                "source": source_id,
                "target": target_id,
                "message": message,
                "message_preview": preview,
                "raw_output": getattr(message, "token_logprobs", None),
            }
        )

    def _traverse(self, message: Message, consumer_operator: Any | None, consumer_call_index: Optional[int] = None):
        """深度优先遍历，构建图。

        Args:
            message: 当前消息
            consumer_operator: 消费该消息的算子
            consumer_call_index: 消费算子的调用索引
        """
        msg_id = id(message)
        if msg_id in self._visited_messages:
            return
        self._visited_messages.add(msg_id)

        producer_operator = getattr(message, "source_operator", None)
        # 获取消息的调用索引（如果消息来自算子调用）
        producer_call_index = getattr(message, "call_index", None)

        if producer_operator:
            producer_id = self._add_node(
                producer_operator, call_index=producer_call_index)
        else:
            producer_id = self._add_node(
                operator=None, is_external=True, label=f"external_msg_{msg_id}")

        if consumer_operator:
            consumer_id = self._add_node(
                consumer_operator, call_index=consumer_call_index)
            self._add_edge(producer_id, consumer_id, message)

        # 递归遍历生产该消息的算子的输入
        # 如果算子有调用历史，使用对应调用的输入消息
        if producer_operator:
            input_messages = None
            if producer_call_index is not None and hasattr(producer_operator, "call_history"):
                # 从调用历史中获取对应调用的输入消息
                call_history = producer_operator.call_history
                if producer_call_index < len(call_history):
                    input_messages = call_history[producer_call_index].get(
                        "input_messages")

            # 如果没有找到调用历史，使用向后兼容的方式
            if input_messages is None:
                input_messages = getattr(
                    producer_operator, "input_messages", None)

            if input_messages:
                for upstream_msg in input_messages:
                    if isinstance(upstream_msg, Message):
                        self._traverse(
                            upstream_msg, producer_operator, producer_call_index)

    def _build_graph(self):
        """从输出消息开始构建执行图。"""
        self._traverse(self.output_message, consumer_operator=None)

    def get_nodes(self) -> List[Dict[str, Any]]:
        """返回节点列表。"""
        return self.nodes

    def get_edges(self) -> List[Dict[str, Any]]:
        """返回边列表。"""
        return self.edges

    def as_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """以字典形式返回图数据，便于序列化。"""
        return {"nodes": self.nodes, "edges": self.edges}
