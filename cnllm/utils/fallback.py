import logging
import warnings
from typing import Dict, Callable, Any, Optional, List
from .exceptions import ModelNotSupportedError, FallbackError

logger = logging.getLogger(__name__)


class FallbackManager:
    def __init__(
        self,
        fallback_config: Dict[str, Optional[str]],
        primary_api_key: str,
        supported_models: Dict[str, str],
        adapter_map: Dict[str, Any],
        get_adapter_func: Callable[[str, str], Any],
        on_fallback: Callable[[str, str, Exception], None] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = None
    ):
        self.fallback_config = fallback_config
        self.primary_api_key = primary_api_key
        self.supported_models = supported_models
        self.adapter_map = adapter_map
        self.get_adapter_func = get_adapter_func
        self.on_fallback = on_fallback or self._default_fallback_handler
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self._last_adapter = None

    def _default_fallback_handler(self, from_model: str, to_model: str, error: Exception):
        msg = f"模型 {from_model} 失败: {error} -> 切换到 {to_model}"
        logger.warning(f"[CNLLM Fallback] {msg}")
        warnings.warn(msg, UserWarning)

    def _get_adapter_for_model(self, model: str, api_key: str, timeout: int, max_retries: int, retry_delay: float, base_url: str):
        adapter_name = self.supported_models.get(model)
        if not adapter_name:
            raise ModelNotSupportedError(
                message=f"暂不支持模型: {model}\n支持的模型: {', '.join(self.supported_models.keys())}",
                provider="unknown"
            )

        adapter_class = self.adapter_map.get(adapter_name)
        if not adapter_class:
            raise ModelNotSupportedError(
                message=f"模型 {model} 的 Adapter {adapter_name} 不可用",
                provider="unknown"
            )

        return adapter_class(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url
        )

    def execute_with_fallback(
        self,
        primary_model: str,
        primary_api_key: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        **kwargs
    ) -> Any:
        tried = []
        last_error = None

        models_to_try = [(primary_model, primary_api_key)]
        for fb_model, fb_key in self.fallback_config.items():
            models_to_try.append((fb_model, fb_key or primary_api_key))

        for model, api_key in models_to_try:
            tried.append(model)
            if model not in self.supported_models:
                continue
            try:
                adapter = self._get_adapter_for_model(
                    model, api_key,
                    self.timeout, self.max_retries, self.retry_delay,
                    self.base_url
                )
                self._last_adapter = adapter
                return adapter.create_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    model=model,
                    **kwargs
                )
            except ModelNotSupportedError:
                raise
            except Exception as e:
                last_error = e
                if model != tried[0]:
                    self.on_fallback(tried[0], model, e)
                continue

        raise FallbackError(
            f"所有模型均失败。已尝试: {', '.join(tried)}\n最后错误: {last_error}"
        ) from last_error