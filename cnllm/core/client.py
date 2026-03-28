from typing import Optional, Dict, Any, Iterator, List
import logging

from .models import SUPPORTED_MODELS, ADAPTER_MAP, validate_model
from ..utils.exceptions import ModelNotSupportedError, MissingParameterError

logger = logging.getLogger(__name__)


class CNLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_models: Optional[Dict[str, Optional[str]]] = None
    ):
        self.model = self._normalize_model(model) if model else None
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_models = fallback_models or {}

        self._adapters = {}
        self.adapter = None
        self.chat = self.ChatNamespace(self)

    def _normalize_model(self, model: str) -> str:
        model = model.lower()
        if model == "minimax":
            model = "minimax-m2.7"
        return model

    def _on_fallback(self, from_model: str, to_model: str, error: Exception):
        logger.warning(
            f"[CNLLM Fallback] 模型 {from_model} 失败: {error}\n"
            f"正在切换到备用模型: {to_model}"
        )

    def _get_adapter(self, model: str, api_key: str):
        validate_model(model)
        adapter_name = SUPPORTED_MODELS.get(model)
        if not adapter_name:
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}\n支持的模型: {', '.join(SUPPORTED_MODELS.keys())}",
                provider="unknown"
            )

        adapter_class = ADAPTER_MAP.get(adapter_name)
        if not adapter_class:
            raise ModelNotSupportedError(
                message=f"模型 {model} 的 Adapter {adapter_name} 不可用",
                provider="unknown"
            )

        return adapter_class(
            api_key=api_key,
            model=model,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            base_url=self.base_url
        )

    def _prompt_to_messages(self, prompt: str) -> list[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def __call__(self, prompt: str) -> Dict[str, Any]:
        return self.chat.create(prompt=prompt)

    class ChatNamespace:
        def __init__(self, parent):
            self.parent = parent

        def create(
            self,
            messages: list[Dict[str, str]] = None,
            prompt: str = None,
            model: str = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ):
            if messages is None and prompt is None:
                raise MissingParameterError(parameter="messages 或 prompt")

            if messages is None:
                messages = self.parent._prompt_to_messages(prompt)

            if model is not None:
                validate_model(model)
                adapter = self.parent._get_adapter(model, self.parent.api_key)
                return adapter.create_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    model=model,
                    **kwargs
                )

            if self.parent.model is None and not self.parent.fallback_models:
                raise MissingParameterError(parameter="model")

            from ..utils.fallback import FallbackManager
            fb_manager = FallbackManager(
                fallback_config=self.parent.fallback_models,
                primary_api_key=self.parent.api_key,
                supported_models=SUPPORTED_MODELS,
                adapter_map=ADAPTER_MAP,
                get_adapter_func=self.parent._get_adapter,
                on_fallback=self.parent._on_fallback,
                timeout=self.parent.timeout,
                max_retries=self.parent.max_retries,
                retry_delay=self.parent.retry_delay,
                base_url=self.parent.base_url
            )
            return fb_manager.execute_with_fallback(
                primary_model=self.parent.model,
                primary_api_key=self.parent.api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
