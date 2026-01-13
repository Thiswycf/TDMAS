#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-cost proxy: perplexity over an execution graph.
Per-message perplexity is computed from the chosen token logprobs; the proxy
returns the sum of per-message perplexities across the graph.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Set

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


def _message_perplexity(msg: Message) -> float:
    """Compute perplexity for a single message using chosen token logprobs."""
    # Preferred: tid_logprobs already paired with chosen tokens
    if getattr(msg, "tid_logprobs", None):
        logprobs = [lp for _, lp in msg.tid_logprobs]
    else:
        # Fallback: derive from logprobs_distribution and chosen_token_ids
        logprobs = []
        dist = getattr(msg, "logprobs_distribution", None)
        chosen = getattr(msg, "chosen_token_ids", None)
        if dist and chosen and len(dist) == len(chosen):
            for step_dist, token_id in zip(dist, chosen):
                if token_id in step_dist:
                    logprobs.append(step_dist[token_id])

    if not logprobs:
        return 0.0

    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    return math.exp(avg_neg_logprob)


def evaluate(graph: ExecutionGraph) -> Dict[str, Any]:
    """
    Evaluate perplexity proxy on an execution graph.

    Return sum(ppl_values)
    """
    edges = graph.get_edges() if hasattr(graph, "get_edges") else []
    msgs = list(_unique_messages(edges))

    ppl_values = [_message_perplexity(m) for m in msgs]
    return sum(ppl_values)
