#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLLMAdapter for local large language models, compatible with all API calls
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["disable_custom_all_reduce"] = "True"
os.environ["NCCL_IB_DISABLE"] = "True"
os.environ["NCCL_P2P_DISABLE"] = "True"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import logging
import time
from typing import List, Union, Dict
import yaml
import json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.__init__").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


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
            model_name = os.path.basename(model_path)

        # Check if model has quantization configuration
        config_path = os.path.join(model_path, "config.json")
        quantization_config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    quantization_config = config.get("quantization_config")
            except Exception as e:
                logging.warning(f"Failed to read config.json: {e}")

        # Handle quantization
        vllm_kwargs = kwargs.copy()
        if quantization_config:
            logging.info(f"Detected quantization configuration: {quantization_config}")
            
            # Check if it's a 4-bit quantization model
            if quantization_config.get("load_in_4bit") or quantization_config.get("_load_in_4bit"):
                logging.info("Model is 4-bit quantized. Using appropriate VLLM settings.")
                # 处理预量化模型不支持张量并行的问题
                if quantization_config.get("quant_method") == "bitsandbytes":
                    logging.info("BitsAndBytes pre-quantized model detected. Disabling tensor parallelism.")
                    # 强制将张量并行大小设置为1
                    vllm_kwargs['tensor_parallel_size'] = 1
            
            # Check if it's a 8-bit quantization model  
            elif quantization_config.get("load_in_8bit") or quantization_config.get("_load_in_8bit"):
                logging.info("Model is 8-bit quantized. Using appropriate VLLM settings.")
            
            # Extract any useful parameters from quantization_config
            if "bnb_4bit_quant_type" in quantization_config:
                logging.info(f"Quantization type: {quantization_config['bnb_4bit_quant_type']}")
        else:
            logging.info("No quantization configuration found. Using default settings.")

        # Initialize VLLM
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            **vllm_kwargs
        )

        # Initialize tokenizer
        # Fix Mistral regex pattern issue for all models that might need it
        # This is a common issue with many tokenizers based on Mistral architecture
        tokenizer_kwargs = {"fix_mistral_regex": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

        # Other BaseLLM required attributes
        self.auto_max_tokens = False
        self.cost_manager = None
        self.model_name = model_name

    @staticmethod
    def needs_chat_template(model_name: str):
        model_name = model_name.lower()
        keywords = ["chat", "instruct", "qwen", "glm", "yi"]
        return any(k in model_name for k in keywords)

    # Implement generate method for compatibility with generate.py
    def generate(self, prompts: Union[List[str], List[List[Dict[str, str]]]], sampling_params: SamplingParams):
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
                        self.tokenizer.apply_chat_template(prompt, **kwargs)
                    )
                else:
                    raise ValueError(f"Invalid prompt type: {type(prompt)}")

            # Generate responses
            return self.llm.generate(processed_prompts, sampling_params)
        else:
            # For base models, use the prompts as is
            return self.llm.generate(prompts, sampling_params)

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        """Chat method compatible with ChatGPT API format

        Args:
            messages: List of messages with "role" and "content"
            **kwargs: Additional parameters
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Extract sampling parameters
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 50)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )

        # Generate response
        outputs = self.generate([messages], sampling_params)
        return {
            "id": f"chatcmpl-{os.urandom(16).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": outputs[0].outputs[0].text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": outputs[0].prompt_token_ids.shape[0],
                "completion_tokens": outputs[0].outputs[0].token_ids.shape[0],
                "total_tokens": outputs[0].prompt_token_ids.shape[0] + outputs[0].outputs[0].token_ids.shape[0]
            }
        }

    def completion(self, prompt: str, **kwargs):
        """Completion method compatible with OpenAI API format

        Args:
            prompt: Single prompt string
            **kwargs: Additional parameters
        """
        # Extract sampling parameters
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 50)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )

        # Generate response
        outputs = self.generate([prompt], sampling_params)
        return {
            "id": f"cmpl-{os.urandom(16).hex()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "text": outputs[0].outputs[0].text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": outputs[0].prompt_token_ids.shape[0],
                "completion_tokens": outputs[0].outputs[0].token_ids.shape[0],
                "total_tokens": outputs[0].prompt_token_ids.shape[0] + outputs[0].outputs[0].token_ids.shape[0]
            }
        }