#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/11
@Author  : DeepSeek API Implementation
@File    : deepseek_api.py
@Desc    : DeepSeek API implementation compatible with OpenAI API structure
"""
from __future__ import annotations

import json
import re
from typing import Optional, Union

from openai import APIConnectionError, AsyncOpenAI, AsyncStream
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from maas.configs.llm_config import LLMConfig, LLMType
from maas.const import USE_CONFIG_TIMEOUT
from maas.logs import log_llm_stream, logger
from maas.provider.base_llm import BaseLLM
from maas.provider.constant import GENERAL_FUNCTION_SCHEMA
from maas.provider.llm_provider_registry import register_provider
from maas.utils.common import CodeParser, decode_image, log_and_reraise
from maas.utils.cost_manager import CostManager
from maas.utils.exceptions import handle_exception
from maas.utils.token_counter import (
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


@register_provider([LLMType.DEEPSEEK])
class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation that follows OpenAI API structure"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None

    def _init_client(self):
        """Initialize the DeepSeek client using AsyncOpenAI"""
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self) -> dict:
        """Make kwargs for the DeepSeek client"""
        kwargs = {
            "api_key": self.config.api_key, 
            "base_url": self.config.base_url
        }

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> Optional[dict]:
        """Get proxy parameters if available"""
        if not self.config.proxy:
            return None
        return {"proxy": self.config.proxy}

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """_achat_completion implemented by inherited class"""
        kwargs = {}
        if timeout != USE_CONFIG_TIMEOUT:
            kwargs["timeout"] = timeout
        
        # Set default model if not provided
        kwargs["model"] = self.model
        
        # Call the OpenAI client which will work with DeepSeek's API
        rsp = await self.aclient.chat.completions.create(messages=messages, **kwargs)
        
        # Update costs if needed
        if hasattr(rsp, 'usage'):
            usage_dict = {
                "prompt_tokens": rsp.usage.prompt_tokens,
                "completion_tokens": rsp.usage.completion_tokens,
                "total_tokens": rsp.usage.total_tokens
            }
            self._update_costs(usage_dict)
        
        return rsp.model_dump()

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """_achat_completion_stream implemented by inherited class"""
        kwargs = {"stream": True}
        if timeout != USE_CONFIG_TIMEOUT:
            kwargs["timeout"] = timeout
        
        # Set default model if not provided
        kwargs["model"] = self.model
        
        # Call the OpenAI client which will work with DeepSeek's API
        stream = await self.aclient.chat.completions.create(messages=messages, **kwargs)
        
        full_response = []
        async for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    full_response.append(content)
                    log_llm_stream(content)
        
        return ''.join(full_response)

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """Asynchronous version of completion"""
        return await self._achat_completion(messages, timeout)

    async def acall(self, messages: list[dict], **kwargs) -> dict:
        """Call DeepSeek API asynchronously"""
        # DeepSeek uses the same API structure as OpenAI, so we can reuse most of the OpenAI implementation
        # Here we just need to handle any DeepSeek-specific parameters or behaviors
        return await self._achat_completion(messages)

    def get_choice_text(self, rsp):
        """Extract text from response"""
        if isinstance(rsp, dict) and 'choices' in rsp and rsp['choices']:
            return rsp['choices'][0]['message']['content']
        elif hasattr(rsp, 'choices') and rsp.choices:
            return rsp.choices[0].message.content
        return ""

    @handle_exception
    async def amoderation(self, content: Union[str, list[str]]):
        """DeepSeek doesn't provide moderation API, so we'll skip it"""
        logger.warning("DeepSeek doesn't support moderation API")
        return {"results": [{"flagged": False}]}

    async def atext_to_speech(self, **kwargs):
        """DeepSeek doesn't provide TTS API, so we'll skip it"""
        logger.warning("DeepSeek doesn't support text-to-speech API")
        raise NotImplementedError("DeepSeek doesn't support text-to-speech API")

    async def aspeech_to_text(self, **kwargs):
        """DeepSeek doesn't provide speech-to-text API, so we'll skip it"""
        logger.warning("DeepSeek doesn't support speech-to-text API")
        raise NotImplementedError("DeepSeek doesn't support speech-to-text API")

    async def gen_image(self, prompt: str, **kwargs):
        """DeepSeek doesn't provide image generation API, so we'll skip it"""
        logger.warning("DeepSeek doesn't support image generation API")
        raise NotImplementedError("DeepSeek doesn't support image generation API")

    def _calc_usage(self, usage: CompletionUsage) -> dict:
        """Calculate usage and cost"""
        # Implement cost calculation for DeepSeek if needed
        # For now, we'll return basic usage information
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    # Remove _get_timeout as we use the parent class get_timeout method