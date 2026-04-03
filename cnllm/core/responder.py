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

        tool_calls = self._get_by_path(raw, fields.get("tool_calls", "choices[0].message.tool_calls"))
        reasoning_content = self._get_by_path(raw, fields.get("reasoning_content", "choices[0].message.reasoning_content"))

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

        cached_tokens = self._get_by_path(raw, fields.get("cached_tokens", "usage.prompt_tokens_details.cached_tokens"), None)
        if cached_tokens is not None:
            usage["prompt_tokens_details"] = {
                "cached_tokens": cached_tokens
            }

        message = {
            "role": defaults.get("role", "assistant"),
            "content": content or ""
        }

        if tool_calls:
            message["tool_calls"] = tool_calls
            if message["content"] == "":
                message["content"] = None

        choice = {
            "index": defaults.get("index", 0),
            "message": message,
            "finish_reason": self._get_by_path(raw, fields.get("finish_reason", "choices[0].finish_reason")) or defaults.get("finish_reason")
        }

        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": defaults.get("object", "chat.completion"),
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
            "usage": usage
        }

        if reasoning_content:
            result["_thinking"] = reasoning_content

        return result

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
        reasoning_content = self._get_by_path(raw, reasoning_content_path)

        delta_obj = {
            "content": content,
            "role": stream_fields.get("role", "assistant")
        }

        if tool_calls:
            delta_obj["tool_calls"] = tool_calls

        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
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

        if reasoning_content:
            result["_reasoning_content"] = reasoning_content

        return result

    def check_error(self, raw_response: Dict[str, Any], adapter_name: str = "") -> None:
        self._check_sensitive(raw_response, adapter_name)

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

    def _check_sensitive(self, raw_response: Dict[str, Any], adapter_name: str = "") -> None:
        sensitive_check = self._get_config_value("error_check", "sensitive_check")
        if not sensitive_check:
            return

        input_path = sensitive_check.get("input_sensitive_type_path", "").split(".")
        output_path = sensitive_check.get("output_sensitive_type_path", "").split(".")

        input_type = raw_response
        for key in input_path:
            if isinstance(input_type, dict):
                input_type = input_type.get(key)
            else:
                input_type = None
                break

        if input_type is not None and input_type != "null" and input_type != "" and input_type != 0:
            raise ContentFilteredError(
                message=f"{adapter_name} 输入内容敏感: {input_type}",
                provider=adapter_name
            )

        output_type = raw_response
        for key in output_path:
            if isinstance(output_type, dict):
                output_type = output_type.get(key)
            else:
                output_type = None
                break

        if output_type is not None and output_type != "null" and output_type != "" and output_type != 0:
            raise ContentFilteredError(
                message=f"{adapter_name} 输出内容敏感: {output_type}",
                provider=adapter_name
            )

    def collect_stream_result(self, raw_response: Dict[str, Any], result: Dict[str, Any]) -> None:
        if raw_response is None:
            return
        if "chunks" not in raw_response:
            raw_response["chunks"] = []
        raw_response["chunks"].append(result)

        reasoning_content = result.pop("_reasoning_content", None)
        if reasoning_content:
            if "_thinking" not in raw_response:
                raw_response["_thinking"] = ""
            raw_response["_thinking"] += reasoning_content

        delta = result.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content") or ""
        if content:
            if "_still" not in raw_response:
                raw_response["_still"] = ""
            raw_response["_still"] += content

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            if "_tools" not in raw_response:
                raw_response["_tools"] = []
            raw_response["_tools"].extend(tool_calls)

        return result
