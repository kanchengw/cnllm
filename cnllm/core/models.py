from typing import Dict, Type, Set
from ..utils.exceptions import ModelNotSupportedError
from ..adapters.minimax.chat import MiniMaxAdapter

SUPPORTED_MODELS: Dict[str, str] = {
    "minimax-m2.7": "minimax",
    "minimax-m2.5": "minimax",
    "minimax-m2.1": "minimax",
    "minimax-m2": "minimax",
}

ADAPTER_MAP: Dict[str, Type] = {
    "minimax": MiniMaxAdapter,
}

_validation_ref_count: int = 0


def validate_model(model: str) -> bool:
    if _validation_ref_count > 0:
        return True
    if model not in SUPPORTED_MODELS:
        raise ModelNotSupportedError(
            message=f"暂不支持模型: {model}\n支持的模型: {', '.join(SUPPORTED_MODELS.keys())}",
            provider="unknown"
        )
    return True


def enable_validation() -> None:
    global _validation_ref_count
    if _validation_ref_count > 0:
        _validation_ref_count -= 1


def disable_validation() -> None:
    global _validation_ref_count
    _validation_ref_count += 1
