import logging
import warnings
from typing import Dict, Callable, Any, Optional, List
from .exceptions import ModelNotSupportedError, FallbackError, MissingParameterError, ContentFilteredError

logger = logging.getLogger(__name__)


class FallbackManager:
    def __init__(
        self,
        fallback_config: Dict[str, Optional[str]],
        primary_api_key: str,
        get_adapter_func: Callable[[str, str], Any],
        on_fallback: Callable[[str, str, Exception], None] = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        base_url: str = None
    ):
        self.fallback_config = fallback_config
        self.primary_api_key = primary_api_key
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

        last_error = None
        only_primary_failed = len(models_to_try) == 1

        for model, api_key in models_to_try:
            tried.append(model)
            try:
                adapter = self.get_adapter_func(
                    model, api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    base_url=self.base_url
                )
                self._last_adapter = adapter
            except Exception as e:
                last_error = e
                if only_primary_failed:
                    raise
                if model != tried[0]:
                    self.on_fallback(tried[0], model, e)
                continue

            try:
                return adapter.create_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    model=model,
                    **kwargs
                )
            except (ModelNotSupportedError, MissingParameterError, ContentFilteredError):
                raise
            except Exception as e:
                last_error = e
                if model != tried[0]:
                    self.on_fallback(tried[0], model, e)
                continue

        raise FallbackError(
            f"所有模型均失败。已尝试: {', '.join(tried)}\n最后错误: {last_error}"
        ) from last_error
