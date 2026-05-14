import os
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Iterator
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry

logger = logging.getLogger(__name__)


class MiniMaxVendorError(VendorError):
    VENDOR_NAME = "minimax"
    SENSITIVE_CONTENT_CODE = 99999

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["MiniMaxVendorError"]:
        if not raw_response:
            return None

        input_sensitive = raw_response.get("input_sensitive_type")
        output_sensitive = raw_response.get("output_sensitive_type")

        if input_sensitive and input_sensitive not in ("null", "", 0):
            return cls(
                code=cls.SENSITIVE_CONTENT_CODE,
                message=f"输入内容敏感: {input_sensitive}",
                vendor=cls.VENDOR_NAME,
                raw_response=raw_response
            )

        if output_sensitive and output_sensitive not in ("null", "", 0):
            return cls(
                code=cls.SENSITIVE_CONTENT_CODE,
                message=f"输出内容敏感: {output_sensitive}",
                vendor=cls.VENDOR_NAME,
                raw_response=raw_response
            )

        base_resp = raw_response.get("base_resp", {})
        code = base_resp.get("status_code")
        if code is None:
            return None
        message = base_resp.get("status_msg", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register(MiniMaxVendorError.VENDOR_NAME, MiniMaxVendorError)
VendorErrorRegistry.register("minimax-native", MiniMaxVendorError)


class MiniMaxNativeResponder(Responder):
    CONFIG_DIR = "minimax"

    def __init__(self):
        super().__init__("minimax")
        self._config = self._config.get("native", {})
        self._stream_prev_had_finish = False

    def _reset_stream_state(self):
        self._stream_prev_had_finish = False

    def to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
        choices = raw.get("choices", [])
        is_new_stream = (
            self._stream_prev_had_finish
            and any(
                c.get("delta", {}).get("role") for c in choices
            )
        )
        if is_new_stream:
            self._reset_stream_state()

        if self._stream_prev_had_finish and not is_new_stream:
            return None

        if self._stream_prev_had_finish and self._is_raw_effectively_empty(raw):
            return None

        result = super().to_openai_stream_format(raw, model)
        for choice in result.get("choices", []):
            if "message" in choice:
                del choice["message"]

        for choice in result.get("choices", []):
            if choice.get("finish_reason"):
                self._stream_prev_had_finish = True

        return result

    def _is_raw_effectively_empty(self, raw: Dict[str, Any]) -> bool:
        for choice in raw.get("choices", []):
            delta = choice.get("delta", {})
            has_content = bool(delta.get("content"))
            has_tool_calls = bool(delta.get("tool_calls"))
            has_role = bool(delta.get("role"))
            if has_content or has_tool_calls or has_role:
                return False
        return True


class MiniMaxNativeAdapter(BaseAdapter):
    ADAPTER_NAME = "minimax-native"
    CONFIG_DIR = "minimax"

    @classmethod
    def _load_class_config(cls) -> Dict[str, Any]:
        if cls._class_config is not None:
            return cls._class_config
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "configs", cls.CONFIG_DIR, "request_minimax.yaml"
        )
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._class_config = yaml.safe_load(f) or {}
                mapping = cls._class_config.get("model_mapping", {})
                if isinstance(mapping, dict) and "chat" in mapping:
                    mapping = mapping["chat"]
                cls._supported_models = list(mapping.keys()) if mapping else []
                return cls._class_config
        except FileNotFoundError:
            cls._class_config = {}
            cls._supported_models = []
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            cls._class_config = {}
            cls._supported_models = []
            return {}

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        base_url: str = None,
        fallback_models: Optional[Dict[str, Optional[str]]] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            fallback_models=fallback_models,
            **kwargs
        )
        self.responder = MiniMaxNativeResponder()
        self._stream_prev_had_finish = False
        self._last_content = ""

    def _get_responder(self):
        return self.responder

    def _reset_stream_state(self):
        self._stream_prev_had_finish = False
        self.responder._reset_stream_state()

    def _accumulate_extra_fields(self, result: Dict[str, Any]) -> None:
        choices = result.get("choices", [])
        super()._accumulate_extra_fields(result)
        for choice in choices:
            choice.pop("message", None)

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
        choices = raw.get("choices", [])
        is_new_stream = (
            self._stream_prev_had_finish
            and any(
                c.get("delta", {}).get("role") for c in choices
            )
        )
        if is_new_stream:
            self._reset_stream_state()

        result = self.responder.to_openai_stream_format(raw, model)

        if result is not None:
            for choice in result.get("choices", []):
                if choice.get("finish_reason"):
                    self._stream_prev_had_finish = True

        return result


class MiniMaxResponder(Responder):
    """MiniMax OpenAI 兼容接口 Responder"""
    CONFIG_DIR = "minimax"
    def __init__(self):
        super().__init__("minimax")
        self._config = self._config.get("openai", {})
        self._think_buffer = ""
        self._in_think = False

    def _reset_think_state(self):
        self._think_buffer = ""
        self._in_think = False

    def _process_think_content(self, content: str) -> str:
        """处理流式 <think> 标签，返回清理后的 content，将 thinking 部分添加到 _think_buffer"""
        if not content:
            return ""

        if self._in_think:
            # 已在 <think> 块内
            close_idx = content.find("</think>")
            if close_idx >= 0:
                self._think_buffer += content[:close_idx]
                self._in_think = False
                return content[close_idx + len("</think>"):].strip()
            else:
                self._think_buffer += content
                return ""
        else:
            # 不在 <think> 块内，检查是否有新的 <think>
            open_idx = content.find("<think>")
            if open_idx >= 0:
                # 提取打开标签前的内容（非 thinking 部分）
                before = content[:open_idx]
                after_open = content[open_idx + len("<think>"):]
                close_idx = after_open.find("</think>")
                if close_idx >= 0:
                    # 同一 chunk 内闭合
                    self._think_buffer = after_open[:close_idx]
                    return (before + after_open[close_idx + len("</think>"):]).strip()
                else:
                    # 跨 chunk
                    self._in_think = True
                    self._think_buffer = after_open
                    return before.strip()
            else:
                return content

    def to_openai_stream_format(self, raw, model):
        result = super().to_openai_stream_format(raw, model)
        if result is None:
            return None
        for choice in result.get("choices", []):
            delta = choice.get("delta", {})
            # 处理 reason_details → reasoning_content
            rds = delta.get("reasoning_details")
            if rds and isinstance(rds, list) and len(rds) > 0:
                texts = []
                for rd in rds:
                    if isinstance(rd, dict) and rd.get("text"):
                        texts.append(rd["text"])
                if texts:
                    delta["reasoning_content"] = "".join(texts)

            # 处理 <think> 标签流式分块
            content = delta.get("content", "") or ""
            if content:
                cleaned = self._process_think_content(content)
                delta["content"] = cleaned
                if self._think_buffer:
                    delta["reasoning_content"] = delta.get("reasoning_content", "") + self._think_buffer
                    self._think_buffer = ""

            # finish_reason 出现时重置 think 状态（新请求）
            if choice.get("finish_reason"):
                self._reset_think_state()

        return result

class MiniMaxAdapter(BaseAdapter):
    """MiniMax OpenAI 兼容接口 Adapter"""
    ADAPTER_NAME = "minimax"
    CONFIG_DIR = "minimax"
    def __init__(self, api_key, model, timeout=None, max_retries=None, retry_delay=None, base_url=None, fallback_models=None, protocol=None, **kwargs):
        super().__init__(api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, retry_delay=retry_delay, base_url=base_url, fallback_models=fallback_models, protocol=protocol, **kwargs)
        self.responder = MiniMaxResponder()
    def _get_responder(self):
        return self.responder
    def _to_openai_format(self, raw, model):
        return self.responder.to_openai_format(raw, model)
    def _do_to_openai_stream_format(self, raw, model):
        return self.responder.to_openai_stream_format(raw, model)

MiniMaxAdapter._register()
MiniMaxNativeAdapter._register()