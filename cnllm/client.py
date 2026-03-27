from typing import Optional, Dict, Any, Iterator
import logging

from .adapters.minimax.chat import MiniMaxAdapter
from .core.exceptions import ModelNotSupportedError

logger = logging.getLogger(__name__)


class CNLLM:
    SUPPORTED_MODELS = ["minimax", "minimax-m2.5", "minimax-m2.7"]

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = None,
        organization: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        extra_config: Dict[str, Any] = None
    ):
        self.model = self._normalize_model(model)
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = extra_config or {}
        self.adapter = self._get_adapter()
        self.chat = self.ChatNamespace(self)

    def _normalize_model(self, model: str) -> str:
        model = model.lower()
        if model == "minimax":
            model = "minimax-m2.7"
        return model

    def _get_adapter(self):
        if self.model.startswith("minimax"):
            return MiniMaxAdapter(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                base_url=self.base_url,
                extra_config=self.extra_config
            )
        raise ModelNotSupportedError(
            message=f"暂不支持模型: {self.model}\n支持的模型: {', '.join(self.SUPPORTED_MODELS)}",
            provider="unknown"
        )

    def _prompt_to_messages(self, prompt: str) -> list[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return self.chat.create(prompt=prompt, **kwargs)

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
            extra_config: Dict[str, Any] = None,
            **kwargs
        ):
            if messages is None and prompt is None:
                from .core.exceptions import MissingParameterError
                raise MissingParameterError(parameter="messages 或 prompt")

            if messages is None:
                messages = self.parent._prompt_to_messages(prompt)

            return self.parent.adapter.create_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                model=model,
                extra_config=extra_config,
                **kwargs
            )
