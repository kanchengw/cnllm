from typing import Optional, Dict, Any, Iterator, List
import logging
import os

from ..utils.exceptions import ModelNotSupportedError, MissingParameterError
from ..utils.fallback import FallbackManager

logger = logging.getLogger(__name__)


class CNLLM:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        fallback_models: Optional[Dict[str, Optional[str]]] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        import os
        if "prompt" in kwargs or "messages" in kwargs:
            raise TypeError(
                "客户端初始化不接受 prompt 或 messages 参数，"
                "请使用 client().create(prompt=...) 或 client().create(messages=[...]) 方法"
            )
        self.model = model.lower() if model else None
        self.api_key = api_key if api_key is not None else os.getenv("MINIMAX_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_models = fallback_models or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self._kwargs = kwargs

        self._adapters = {}
        self.adapter = None
        self.chat = self.ChatNamespace(self)

    def _on_fallback(self, from_model: str, to_model: str, error: Exception):
        logger.warning(
            f"[CNLLM Fallback] 模型 {from_model} 失败: {error}\n"
            f"正在切换到备用模型: {to_model}"
        )

    def _get_adapter(
            self,
            model: str,
            api_key: str,
            timeout: int = None,
            max_retries: int = None,
            retry_delay: float = None,
            base_url: str = None
        ):
            from ..core.adapter import BaseAdapter

            if os.getenv("CNLLM_SKIP_MODEL_VALIDATION") == "true":
                logger.warning(f"[测试模式] 跳过模型验证: {model}")
                adapter_name = os.getenv("CNLLM_DEFAULT_ADAPTER", "minimax")
                adapter_class = BaseAdapter.get_adapter_class(adapter_name)
                if not adapter_class:
                    adapter_name = BaseAdapter.get_all_adapter_names()[0]
                    adapter_class = BaseAdapter.get_adapter_class(adapter_name)
                return adapter_class(
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    base_url=base_url
                )

            adapter_name = BaseAdapter.get_adapter_name_for_model(model)
            if not adapter_name:
                raise ModelNotSupportedError(
                    message=f"暂不支持模型: {model}",
                    provider="minimax"
                )

            adapter_class = BaseAdapter.get_adapter_class(adapter_name)
            if not adapter_class:
                raise ModelNotSupportedError(
                    message=f"模型 {model} 的 Adapter {adapter_name} 不可用",
                    provider="minimax"
                )

            adapter = adapter_class(
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                base_url=base_url
            )
            adapter._validator.validate_model(model)
            return adapter

    def _prompt_to_messages(self, prompt: str) -> list[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def __call__(self, prompt: str) -> Dict[str, Any]:
        return self.chat.create(prompt=prompt)

    class ChatNamespace:
        def __init__(self, parent):
            self.parent = parent
            self._last_response = None

        @property
        def still(self) -> str:
            raw = self.raw
            if raw and "chunks" in raw:
                return raw.get("_still")
            if self._last_response is None:
                return None
            return self._last_response["choices"][0]["message"]["content"]

        @property
        def raw(self) -> Dict[str, Any]:
            adapter = getattr(self.parent, "_last_adapter", None)
            if adapter is None:
                return {}
            return getattr(adapter, "_raw_response", {})

        @property
        def think(self) -> str:
            raw = self.raw
            if not raw:
                return None
            return raw.get("_thinking")

        @property
        def tools(self) -> Optional[List[Dict[str, Any]]]:
            raw = self.raw
            if raw and "chunks" in raw:
                if "_tools" not in raw:
                    return None
                return raw.get("_tools")
            if self._last_response is None:
                return None
            return self._last_response["choices"][0]["message"].get("tool_calls")

        def create(
            self,
            prompt: str = None,
            messages: list[Dict[str, str]] = None,
            model: str = None,
            api_key: Optional[str] = None,
            temperature: float = None,
            max_tokens: Optional[int] = None,
            stream: bool = None,
            timeout: int = None,
            max_retries: int = None,
            retry_delay: float = None,
            base_url: str = None,
            **kwargs
        ):
            if "fallback_models" in kwargs:
                raise TypeError(
                    "chat.create 不接受 fallback_models 参数，"
                    "请在客户端初始化时使用 fallback_models 参数"
                )

            if messages is None and prompt is not None:
                messages = self.parent._prompt_to_messages(prompt)

            actual_api_key = api_key if api_key is not None else self.parent.api_key
            actual_timeout = timeout if timeout is not None else self.parent.timeout
            actual_max_retries = max_retries if max_retries is not None else self.parent.max_retries
            actual_retry_delay = retry_delay if retry_delay is not None else self.parent.retry_delay
            actual_base_url = base_url if base_url is not None else self.parent.base_url
            actual_temperature = temperature if temperature is not None else self.parent.temperature
            actual_max_tokens = max_tokens if max_tokens is not None else self.parent.max_tokens
            actual_stream = stream if stream is not None else self.parent.stream

            merged_kwargs = {**self.parent._kwargs, **kwargs}

            if model is not None and model != "":
                adapter = self.parent._get_adapter(model, actual_api_key, actual_timeout, actual_max_retries, actual_retry_delay, actual_base_url)
                self.parent._last_adapter = adapter
                resp = adapter.create_completion(
                    messages=messages,
                    temperature=actual_temperature,
                    max_tokens=actual_max_tokens,
                    stream=actual_stream,
                    model=model,
                    timeout=actual_timeout,
                    max_retries=actual_max_retries,
                    retry_delay=actual_retry_delay,
                    base_url=actual_base_url,
                    **merged_kwargs
                )
                if actual_stream and hasattr(resp, '__iter__') and not isinstance(resp, (list, dict, str)):
                    from cnllm.utils.stream import StreamResultAccumulator
                    return StreamResultAccumulator(resp, adapter)
                self._last_response = resp
                return resp

            fb_manager = FallbackManager(
                fallback_config=self.parent.fallback_models,
                primary_api_key=actual_api_key,
                get_adapter_func=self.parent._get_adapter,
                on_fallback=self.parent._on_fallback,
                timeout=actual_timeout,
                max_retries=actual_max_retries,
                retry_delay=actual_retry_delay,
                base_url=actual_base_url
            )
            resp = fb_manager.execute_with_fallback(
                primary_model=self.parent.model,
                primary_api_key=actual_api_key,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
                stream=actual_stream,
                **merged_kwargs
            )
            self.parent._last_adapter = fb_manager._last_adapter
            if actual_stream and hasattr(resp, '__iter__') and not isinstance(resp, (list, dict, str)):
                from cnllm.utils.stream import StreamResultAccumulator
                return StreamResultAccumulator(resp, fb_manager._last_adapter)
            self._last_response = resp
            return resp
