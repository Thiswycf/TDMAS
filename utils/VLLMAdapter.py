#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLLMAdapter for local large language models, compatible with all API calls
"""
from __future__ import annotations

import asyncio
from typing import List, Union, Dict, Any
import yaml
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMAdapter():
    """VLLMAdapter for local large language models, compatible with all API calls"""

    def __init__(self, model_path: str, dtype: str = "bfloat16", **kwargs):
        """Initialize VLLMAdapter with local model

        Args:
            model_path: Path to local model
            dtype: Data type for model weights
            **kwargs
        """
        if 'finetuned' in model_path:  # finetune generator model path
            with open("config/generator_config.yaml", "r") as file:
                generator_config = yaml.safe_load(file)
            model_name = generator_config["model"]
        else:
            model_path = model_path
            model_name = os.path.basename(model_path)

        # Initialize VLLM
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            **kwargs
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Other BaseLLM required attributes
        self.auto_max_tokens = False
        self.cost_manager = None
        self.model_name = model_name

    @staticmethod
    def needs_chat_template(model_name: str):
        if 'finetuned' in model_name:  # finetune generator model
            with open("config/generator_config.yaml", "r") as file:
                generator_config = yaml.safe_load(file)
            model_name = generator_config["model"]
        model_name = model_name.lower()
        keywords = ["chat", "instruct", "qwen", "glm", "yi"]
        return any(k in model_name for k in keywords)

    # Implement generate method for compatibility with generate.py
    def generate(self, prompts: List[str] | List[List[Dict[str, str]]], sampling_params: SamplingParams):
        """Generate responses for a list of prompts

        Args:
            prompts: List of prompts. Each element can be:
                - str: Single round prompt (will be treated as user message)
                - List[Dict[str, str]]: Multi-round conversation, each dict has "role" and "content"
        """
        if self.needs_chat_template(self.model_name):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "qwen3" in self.model_name.lower():
                kwargs["enable_thinking"] = False

            processed_prompts = []
            for prompt in prompts:
                if isinstance(prompt, str):
                    # Single round: treat as user message
                    processed_prompts.append(
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            **kwargs
                        )
                    )
                elif isinstance(prompt, list):
                    # Multi-round conversation
                    # Assert: must have odd number of messages (user-assistant-user-...-user)
                    assert len(
                        prompt) % 2 == 1, f"Multi-round conversation must have odd number of messages (user-assistant-user-...-user), got {len(prompt)} messages"
                    # Assert: last message must be from user
                    assert prompt[-1][
                        "role"] == "user", f"Last message in multi-round conversation must be from 'user', got '{prompt[-1]['role']}'"
                    # Assert: messages alternate between user and assistant
                    for i, msg in enumerate(prompt):
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert msg["role"] == expected_role, f"Message {i} should be from '{expected_role}', got '{msg['role']}'"

                    processed_prompts.append(
                        self.tokenizer.apply_chat_template(
                            prompt,
                            **kwargs
                        )
                    )
                else:
                    raise TypeError(
                        f"Prompt must be str or List[Dict[str, str]], got {type(prompt)}")
        else:
            # No chat template: concatenate multi-round conversations into single prompt
            processed_prompts = []
            for prompt in prompts:
                if isinstance(prompt, str):
                    processed_prompts.append(prompt)
                elif isinstance(prompt, list):
                    # Concatenate all messages into a single prompt
                    # Format: "User: ...\nAssistant: ...\nUser: ..."
                    concatenated = ""
                    for msg in prompt:
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        concatenated += f"{role}: {content}\n\n"
                    processed_prompts.append(concatenated.strip())
                else:
                    raise TypeError(
                        f"Prompt must be str or List[Dict[str, str]], got {type(prompt)}")

        return self.llm.generate(processed_prompts, sampling_params)
