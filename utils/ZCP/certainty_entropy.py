#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-cost proxy: certainty via entropy over full token distributions.
For each message and each step, compute entropy of the probability distribution
over candidate tokens; sum entropies across the graph.
Lower entropy -> higher certainty; this metric reports entropy (so smaller is better).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Set
from metagpt.logs import logger

from utils.execution_graph import ExecutionGraph
from utils.message import Message


def _unique_messages(edges: Iterable[Dict[str, Any]]) -> Iterable[Message]:
    """Yield unique Message instances from graph edges."""
    seen: Set[int] = set()
    for edge in edges:
        msg = edge.get("message")
        if isinstance(msg, Message):
            mid = id(msg)
            if mid not in seen:
                seen.add(mid)
                yield msg


def _step_entropy(logprob_dict: Dict[int, float]) -> float:
    """Compute entropy from a dict of token_id -> logprob."""
    if not logprob_dict:
        return 0.0
    # Convert logprobs to normalized probabilities
    probs: List[float] = []
    max_logp = max(logprob_dict.values())
    # subtract max to improve numerical stability
    for lp in logprob_dict.values():
        probs.append(math.exp(lp - max_logp))
    Z = sum(probs)
    if Z == 0:
        return 0.0
    probs = [p / Z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _message_entropy(msg: Message) -> float:
    """Sum entropy over all steps for one message."""
    dist = getattr(msg, "logprobs_distribution", None)
    if not dist:
        return 0.0
    return sum(_step_entropy(step) for step in dist)


def evaluate(graph: ExecutionGraph) -> Dict[str, Any]:
    """
    Evaluate certainty (entropy) proxy on an execution graph.

    Return: sum(entropy_values)
    """
    edges = graph.get_edges() if hasattr(graph, "get_edges") else []
    if len(edges) == 0:
        logger.warning("Empty graph, cannot compute certainty entropy.")
        return 0.0
    msgs = list(_unique_messages(edges))

    entropy_values = []
    step_count = 0
    for m in msgs:
        dist = getattr(m, "logprobs_distribution", None)
        if dist:
            step_count += len(dist)
        elif m.source_operator is not None:  # except source message
            logger.warning(
                f"Message {m} has no logprobs distribution, cannot compute certainty entropy. msg info: {m.source_operator}, {m.content}")
            return 0.0
        entropy_values.append(_message_entropy(m))

    return sum(entropy_values)
