import os
import re
import uuid
import time
import logging
from typing import Dict, Any, Optional

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

    def _build_defaults(self) -> Dict[str, Any]:
        return {
            "object": self._get_config_value("defaults", "object", default="chat.completion"),
            "index": self._get_config_value("defaults", "index", default=0),
            "role": self._get_config_value("defaults", "role", default="assistant"),
            "logprobs": self._get_config_value("defaults", "logprobs"),
            "finish_reason": self._get_config_value("defaults", "finish_reason", default="stop"),
            "system_fingerprint": self._get_config_value("defaults", "system_fingerprint"),
        }

    def to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        fields = self._get_config_value("fields", default={})
        defaults = self._build_defaults()

        content = self._get_by_path(raw, fields.get("content", "choices[0].message.content"))
        if content is not None:
            content = self.cleaner.clean(content)

        prompt_tokens = self._get_by_path(raw, fields.get("prompt_tokens", "usage.prompt_tokens"), 0)
        completion_tokens = self._get_by_path(raw, fields.get("completion_tokens", "usage.completion_tokens"), 0)
        total_tokens = self._get_by_path(raw, fields.get("total_tokens", "usage.total_tokens"), 0)
        reasoning_tokens = self._get_by_path(raw, fields.get("reasoning_tokens", "usage.completion_tokens_details.reasoning_tokens"), None)

        usage = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0
        }

        if reasoning_tokens is not None and reasoning_tokens > 0:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": defaults.get("object", "chat.completion"),
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": defaults.get("index", 0),
                    "message": {
                        "role": defaults.get("role", "assistant"),
                        "content": content or ""
                    },
                    "finish_reason": self._get_by_path(raw, fields.get("finish_reason", "choices[0].finish_reason")) or defaults.get("finish_reason")
                }
            ],
            "usage": usage
        }

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
            "object": stream_fields.get("object", "chat.completion.chunk"),
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": stream_fields.get("index", 0),
                    "delta": {
                        "content": content,
                        "role": stream_fields.get("role", "assistant")
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
