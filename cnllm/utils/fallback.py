import logging
import warnings
import asyncio
from typing import Dict, Callable, Any, Optional, List, AsyncIterator
from .exceptions import ModelNotSupportedError, FallbackError, MissingParameterError, ContentFilteredError, ModelAPIError

logger = logging.getLogger(__name__)


class FallbackManager:
    def __init__(
        self,
        fallback_config: Dict[str, Any],
        primary_api_key: str,
        get_adapter_func: Callable[[str, str], Any],
        on_fallback: Callable[[str, str, Exception], None] = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        base_url: str = None,
        drop_params: str = None
    ):
        self.fallback_config = fallback_config
        self.primary_api_key = primary_api_key
        self.get_adapter_func = get_adapter_func
        self.on_fallback = on_fallback or self._default_fallback_handler
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.drop_params = drop_params
        self._last_adapter = None

    def _default_fallback_handler(self, from_model: str, to_model: str, error: Exception):
        msg = f"模型 {from_model} 失败: {error} -> 切换到 {to_model}"
        logger.warning(f"[CNLLM Fallback] {msg}")
        warnings.warn(msg, UserWarning)

    def _try_models(self, models_to_try, execute_fn, is_async=False):
        """通用 fallback 循环：按顺序尝试模型，执行 execute_fn(adapter, model, api_key)"""
        tried = []
        all_errors = []
        original_exceptions = []

        for idx, (model, api_key, base_url) in enumerate(models_to_try):
            tried.append(model)
            next_model = models_to_try[idx + 1][0] if idx + 1 < len(models_to_try) else None
            try:
                adapter = self.get_adapter_func(
                    model, api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    base_url=base_url or self.base_url,
                    drop_params=self.drop_params
                )
                self._last_adapter = adapter
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

            try:
                return execute_fn(adapter, model, api_key)
            except (ModelNotSupportedError, MissingParameterError, ContentFilteredError) as e:
                original_exceptions.append(e)
                raise
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

        if len(models_to_try) == 1 and original_exceptions:
            raise original_exceptions[0] from None

        raise FallbackError(
            message=f"所有模型均失败。已尝试: {', '.join(str(m) for m in tried)}",
            errors=all_errors
        ) from None

    def _build_models_to_try(self, primary_model: str, primary_api_key: str) -> list:
        models_to_try = [(primary_model, primary_api_key, None)]  # (model, api_key, base_url)
        for fb_model, fb_config in self.fallback_config.items():
            fb_api_key = fb_config["api_key"]
            fb_base_url = fb_config.get("base_url")
            models_to_try.append((fb_model, fb_api_key, fb_base_url))
        return models_to_try

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
        models = self._build_models_to_try(primary_model, primary_api_key)
        return self._try_models(models, lambda a, m, k: a.create_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
            stream=stream, model=m, **kwargs
        ))

    def execute_embedding_fallback(
        self,
        primary_model: str,
        primary_api_key: str,
        input_data: str,
        **kwargs
    ) -> Any:
        models = self._build_models_to_try(primary_model, primary_api_key)
        return self._try_models(models, lambda a, m, k: a.create(
            input=input_data, model=m, **kwargs
        ))

    async def aexecute_embedding_fallback(
        self,
        primary_model: str,
        primary_api_key: str,
        input_data: str,
        **kwargs
    ) -> Any:
        tried = []
        all_errors = []
        original_exceptions = []

        models_to_try = self._build_models_to_try(primary_model, primary_api_key)

        for idx, (model, api_key, base_url) in enumerate(models_to_try):
            tried.append(model)
            next_model = models_to_try[idx + 1][0] if idx + 1 < len(models_to_try) else None
            try:
                adapter = self.get_adapter_func(
                    model, api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    base_url=base_url or self.base_url,
                    drop_params=self.drop_params
                )
                self._last_adapter = adapter
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

            try:
                return await adapter.acreate(
                    input=input_data, model=model, **kwargs
                )
            except (ModelNotSupportedError, MissingParameterError, ContentFilteredError) as e:
                original_exceptions.append(e)
                raise
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

        if len(models_to_try) == 1 and original_exceptions:
            raise original_exceptions[0] from None

        raise FallbackError(
            message=f"所有模型均失败。已尝试: {', '.join(str(m) for m in tried)}",
            errors=all_errors
        ) from None

    async def aexecute_with_fallback(
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
        all_errors = []
        original_exceptions = []

        models_to_try = self._build_models_to_try(primary_model, primary_api_key)

        for idx, (model, api_key, base_url) in enumerate(models_to_try):
            tried.append(model)
            next_model = models_to_try[idx + 1][0] if idx + 1 < len(models_to_try) else None
            try:
                adapter = self.get_adapter_func(
                    model, api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    base_url=base_url or self.base_url,
                    drop_params=self.drop_params
                )
                self._last_adapter = adapter
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

            try:
                return await adapter.acreate_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    model=model,
                    **kwargs
                )
            except (ModelNotSupportedError, MissingParameterError, ContentFilteredError) as e:
                original_exceptions.append(e)
                raise
            except Exception as e:
                all_errors.append(f"{model}: {type(e).__name__}: {e}")
                original_exceptions.append(e)
                if len(models_to_try) == 1:
                    raise
                self.on_fallback(model, next_model, e)
                continue

        if len(models_to_try) == 1 and original_exceptions:
            raise original_exceptions[0] from None

        raise FallbackError(
            message=f"所有模型均失败。已尝试: {', '.join(str(m) for m in tried)}",
            errors=all_errors
        ) from None
