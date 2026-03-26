import re
import uuid
import time
from typing import Dict, Any, List, Optional
from ...core.base import BaseHttpClient
from ...core.exceptions import ParseError, ModelAPIError
from ...utils.cleaner import OutputCleaner


class MiniMaxAdapter:
    SUPPORTED_MODELS = ["minimax-m2.7", "minimax-m2.5"]
    DEFAULT_MODEL = "minimax-m2.7"
    
    def __init__(
        self,
        api_key: str,
        model: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.client = BaseHttpClient(
            api_key=api_key,
            base_url="https://api.minimaxi.com",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.cleaner = OutputCleaner()
        self.model = model or self.DEFAULT_MODEL
        
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model}\n"
                f"支持的模型: {', '.join(self.SUPPORTED_MODELS)}"
            )

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
        model: str = None
    ) -> Dict[str, Any]:
        use_model = model or self.model
        
        if use_model not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {use_model}")
        
        payload = {
            "model": self._to_minimax_model_name(use_model),
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        try:
            raw_resp = self.client.post("/v1/text/chatcompletion_v2", payload)
        except RuntimeError as e:
            raise ModelAPIError(f"MiniMax API 请求失败: {e}")
        
        return self._to_openai_format(raw_resp, use_model)

    def _to_minimax_model_name(self, model: str) -> str:
        """将统一模型名转换为MiniMax API模型名"""
        mapping = {
            "minimax-m2.7": "MiniMax-M2.7",
            "minimax-m2.5": "MiniMax-M2.5"
        }
        return mapping.get(model, model)

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        MiniMax M2.5/M2.7 原生响应 → 转换为 OpenAI 标准结构
        """
        base_resp = raw.get("base_resp", {})
        if base_resp.get("status_code") and base_resp["status_code"] != 0:
            raise ModelAPIError(
                f"MiniMax API 错误: {base_resp.get('status_msg', '未知错误')}\n"
                f"状态码: {base_resp.get('status_code')}"
            )
        
        try:
            raw_content = raw["choices"][0]["message"]["content"]
            finish_reason = raw["choices"][0].get("finish_reason", "stop")
        except (KeyError, IndexError) as e:
            raise ParseError(f"解析响应失败: 缺少必要字段 {e}\n原始响应: {raw}")
        
        cleaned = self.cleaner.clean(raw_content)
        
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": cleaned
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }