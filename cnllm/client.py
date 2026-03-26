from typing import Optional, Dict, Any
from dataclasses import dataclass

from .adapters.minimax.chat import MiniMaxAdapter


class CNLLM:
    SUPPORTED_MODELS = ["minimax", "minimax-m2.5", "minimax-m2.7"]
    
    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model = self._normalize_model(model)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
                retry_delay=self.retry_delay
            )
        raise ValueError(f"暂不支持模型: {self.model}\n支持的模型: {', '.join(self.SUPPORTED_MODELS)}")

    def create_chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
        model: str = None
    ) -> Dict[str, Any]:
        return self.adapter.create_completion(
            messages=messages,
            temperature=temperature,
            stream=stream,
            model=model
        )

    class ChatNamespace:
        def __init__(self, parent):
            self.parent = parent

        def create(
            self,
            messages: list[Dict[str, str]],
            temperature: float = 0.1,
            stream: bool = False,
            model: str = None
        ):
            return self.parent.create_chat_completion(
                messages=messages,
                temperature=temperature,
                stream=stream,
                model=model
            )