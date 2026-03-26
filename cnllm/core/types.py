from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatResponse:
    content: str
    role: str = "assistant"
    finish_reason: Optional[str] = "stop"