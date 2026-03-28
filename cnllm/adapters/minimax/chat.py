import re
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Iterator
from ...core.base import BaseHttpClient
from ...utils.exceptions import ParseError, ModelAPIError
from ...utils.cleaner import OutputCleaner
from ...core.params import get_provider_name, get_create_params_config

logger = logging.getLogger(__name__)


class MiniMaxAdapter:
    DEFAULT_BASE_URL = "https://api.minimaxi.com"

    def __init__(
        self,
        api_key: str,
        model: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = None,
    ):
        self.client = BaseHttpClient(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.cleaner = OutputCleaner()
        self.model = model

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        use_model = model or self.model

        self._validate_and_filter_params(kwargs, use_model)

        payload = {
            "model": self._to_minimax_model_name(use_model),
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stream:
            payload["stream"] = True

        for key, value in kwargs.items():
            if value is not None and key not in payload:
                payload[key] = value

        try:
            if stream:
                return self._handle_stream_response(payload, use_model)
            else:
                raw_resp = self.client.post("/v1/text/chatcompletion_v2", payload)
                return self._to_openai_format(raw_resp, use_model)
        except RuntimeError as e:
            raise ModelAPIError(f"MiniMax API 请求失败: {e}")

    def _validate_and_filter_params(self, kwargs: Dict[str, Any], model: str):
        provider = get_provider_name(model)
        config = get_create_params_config(provider)
        supported = set(config.get("supported", []))

        for param, value in kwargs.items():
            if value is None:
                continue
            if param not in supported:
                logger.warning(
                    f"⚠️ 参数 '{param}' 在当前模型 ({provider}) 中不支持，已忽略。"
                )

    def _handle_stream_response(self, payload: Dict[str, Any], model: str) -> Iterator[Dict[str, Any]]:
        for raw_chunk in self.client.post_stream("/v1/text/chatcompletion_v2", payload):
            yield self._to_openai_stream_format(raw_chunk, model)

    def _to_minimax_model_name(self, model: str) -> str:
        mapping = {
            "minimax-m2.7": "MiniMax-M2.7",
            "minimax-m2.5": "MiniMax-M2.5"
        }
        return mapping.get(model, model)

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
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

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        if not raw:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None
                    }
                ]
            }
        try:
            choices = raw.get("choices")
            if not choices:
                choices = [{}]
            delta = choices[0].get("delta", {}) if choices[0] else {}
            content = delta.get("content", "") if delta else ""
            finish_reason = choices[0].get("finish_reason") if choices[0] else None
        except (KeyError, IndexError, TypeError):
            content = ""
            finish_reason = None

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content,
                        "role": "assistant"
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
