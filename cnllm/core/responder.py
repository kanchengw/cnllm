import os
import re
import uuid
import time
import logging
from typing import Dict, Any, Optional
from ..utils.vendor_error import VendorErrorRegistry, ErrorTranslator
from ..utils.exceptions import ContentFilteredError

logger = logging.getLogger(__name__)


class OutputCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class Responder:
    CONFIG_DIR = ""

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or self.CONFIG_DIR
        self._config = self._load_config()
        self.cleaner = OutputCleaner()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "configs", self.config_dir, f"response_{self.config_dir}.yaml"
        )
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Response config not found: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load response config: {e}")
            return {}

    def _get_config_value(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value if value is not None else default

    def _get_by_path(self, data: Dict[str, Any], path: str, default=None):
        import re
        keys = re.split(r'\.(?!\d)', path)
        value = data
        for key in keys:
            if value is None:
                return default
            match = re.match(r'(\w+)\[(\d+)\]', key)
            if match:
                dict_key, index = match.groups()
                if isinstance(value, dict):
                    value = value.get(dict_key)
                if isinstance(value, (list, tuple)) and index.isdigit():
                    index = int(index)
                    if index < len(value):
                        value = value[index]
                    else:
                        return default
                else:
                    return default
            elif isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

    def _path_exists(self, data: Dict[str, Any], path: str) -> bool:
        import re
        keys = re.split(r'\.(?!\d)', path)
        value = data
        for key in keys:
            if value is None:
                return False
            match = re.match(r'(\w+)\[(\d+)\]', key)
            if match:
                dict_key, index = match.groups()
                if isinstance(value, dict):
                    value = value.get(dict_key)
                if isinstance(value, (list, tuple)) and index.isdigit():
                    index = int(index)
                    if index < len(value):
                        value = value[index]
                    else:
                        return False
                else:
                    return False
            elif isinstance(value, dict):
                if key not in value:
                    return False
                value = value.get(key)
            else:
                return False
        return True

    def _build_defaults(self) -> Dict[str, Any]:
        return {
            "object": self._get_config_value("defaults", "object", default="chat.completion"),
            "index": self._get_config_value("defaults", "index", default=0),
            "role": self._get_config_value("defaults", "role", default="assistant"),
            "finish_reason": self._get_config_value("defaults", "finish_reason", default="stop"),
        }

    def to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        fields = self._get_config_value("fields", default={})
        defaults = self._build_defaults()

        content = self._get_by_path(raw, fields.get("content", "choices[0].message.content"))
        if content is not None:
            content = self.cleaner.clean(content)

        tool_calls_path = fields.get("tool_calls", "choices[0].message.tool_calls")
        tool_calls = self._get_by_path(raw, tool_calls_path)
        reasoning_content_path = fields.get("reasoning_content", "choices[0].message.reasoning_content")
        reasoning_content = self._get_by_path(raw, reasoning_content_path)

        prompt_tokens = self._get_by_path(raw, fields.get("prompt_tokens", "usage.prompt_tokens"), 0)
        completion_tokens = self._get_by_path(raw, fields.get("completion_tokens", "usage.completion_tokens"), 0)
        total_tokens = self._get_by_path(raw, fields.get("total_tokens", "usage.total_tokens"), 0)

        usage = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0
        }

        reasoning_tokens_path = fields.get("reasoning_tokens", "usage.completion_tokens_details.reasoning_tokens")
        if self._path_exists(raw, reasoning_tokens_path):
            reasoning_tokens = self._get_by_path(raw, reasoning_tokens_path, None)
            usage["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }

        cached_tokens_path = fields.get("cached_tokens", "usage.prompt_tokens_details.cached_tokens")
        if self._path_exists(raw, cached_tokens_path):
            cached_tokens = self._get_by_path(raw, cached_tokens_path, None)
            usage["prompt_tokens_details"] = {
                "cached_tokens": cached_tokens
            }

        message = {
            "role": defaults.get("role", "assistant"),
            "content": content or ""
        }

        if self._path_exists(raw, tool_calls_path):
            message["tool_calls"] = tool_calls
            if message["content"] == "":
                message["content"] = None

        choice = {
            "index": defaults.get("index", 0),
            "message": message,
            "finish_reason": self._get_by_path(raw, fields.get("finish_reason", "choices[0].finish_reason")) or defaults.get("finish_reason")
        }

        logprobs_path = fields.get("logprobs_path", "choices[0].logprobs")
        if self._path_exists(raw, logprobs_path):
            logprobs_value = self._get_by_path(raw, logprobs_path)
            choice["logprobs"] = logprobs_value

        result = {
            "id": self._get_by_path(raw, fields.get("id", "id")) or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": defaults.get("object", "chat.completion"),
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
            "usage": usage
        }

        system_fingerprint = fields.get("system_fingerprint")
        if system_fingerprint and self._path_exists(raw, system_fingerprint):
            result["system_fingerprint"] = self._get_by_path(raw, system_fingerprint)

        return result

    def _extract_extra_fields(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        从原生响应中提取额外字段（still、tools、think）
        返回包含这些字段的字典，供 Adapter 设置到 _cnllm_extra
        """
        if not raw:
            return {}

        cnllm_extra = {}

        reasoning_content_path = self._get_config_value("fields", "reasoning_content")
        if reasoning_content_path and self._path_exists(raw, reasoning_content_path):
            reasoning_content = self._get_by_path(raw, reasoning_content_path)
            if reasoning_content:
                cnllm_extra["_thinking"] = reasoning_content

        content_path = self._get_config_value("fields", "content")
        if content_path and self._path_exists(raw, content_path):
            content = self._get_by_path(raw, content_path)
            if content:
                cnllm_extra["_still"] = self.cleaner.clean(content)

        tool_calls_path = self._get_config_value("fields", "tool_calls")
        if tool_calls_path and self._path_exists(raw, tool_calls_path):
            tool_calls = self._get_by_path(raw, tool_calls_path)
            if tool_calls:
                cnllm_extra["_tools"] = tool_calls

        return cnllm_extra

    def _extract_stream_extra_fields(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        从流式原生 chunk 中提取额外字段
        适用于流式调用的实时累积
        """
        if not raw:
            return {}

        cnllm_extra = {}

        reasoning_content_path = self._get_config_value("stream_fields", "reasoning_content_path")
        if reasoning_content_path and self._path_exists(raw, reasoning_content_path):
            reasoning_content = self._get_by_path(raw, reasoning_content_path)
            if reasoning_content:
                cnllm_extra["_thinking"] = reasoning_content

        content_path = self._get_config_value("stream_fields", "content_path")
        if content_path and self._path_exists(raw, content_path):
            content = self._get_by_path(raw, content_path)
            if content:
                cnllm_extra["_still"] = content

        tool_calls_path = self._get_config_value("stream_fields", "tool_calls_path")
        if tool_calls_path and self._path_exists(raw, tool_calls_path):
            tool_calls = self._get_by_path(raw, tool_calls_path)
            if tool_calls:
                cnllm_extra["_tools"] = tool_calls

        return cnllm_extra

    def to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        if not raw:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": self._get_config_value("stream_fields", "object", default="chat.completion.chunk"),
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": self._get_config_value("stream_fields", "index", default=0),
                        "delta": {},
                        "finish_reason": None
                    }
                ]
            }

        stream_fields = self._get_config_value("stream_fields", default={})
        content_path = stream_fields.get("content_path", "choices[0].delta.content")
        tool_calls_path = stream_fields.get("tool_calls_path", "choices[0].delta.tool_calls")
        reasoning_content_path = stream_fields.get("reasoning_content_path", "choices[0].delta.reasoning_content")

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

        tool_calls = self._get_by_path(raw, tool_calls_path)

        delta_obj = {
            "content": content,
            "role": stream_fields.get("role", "assistant")
        }

        if tool_calls:
            delta_obj["tool_calls"] = tool_calls

        result = {
            "id": self._get_by_path(raw, "id") or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": stream_fields.get("object", "chat.completion.chunk"),
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": stream_fields.get("index", 0),
                    "delta": delta_obj,
                    "finish_reason": finish_reason
                }
            ]
        }

        if "request_id" in raw:
            result["request_id"] = raw["request_id"]

        return result

    def check_error(self, raw_response: Dict[str, Any], adapter_name: str = "") -> None:
        vendor_error = VendorErrorRegistry.create_vendor_error(
            adapter_name.lower(),
            raw_response
        )
        if vendor_error is None:
            return

        success_code = self._get_config_value("error_check", "success_code", default=0)
        auth_code = self._get_config_value("error_check", "auth_code", default=1004)

        translator = ErrorTranslator(self.config_dir)
        translator.translate(vendor_error, success_code=success_code, auth_code=auth_code)
