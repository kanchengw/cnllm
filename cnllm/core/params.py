"""
CNLLM 参数注册表
统一管理各厂商的参数分类
"""
from typing import Dict, List

PROVIDER_PARAMS = {
    "minimax": {
        "init": {
            "required": ["api_key", "model"],
            "supported": ["base_url", "timeout", "max_retries", "retry_delay"],
        },
        "create": {
            "required": [],
            "supported": ["messages", "temperature", "max_tokens", "stream", "tools", "tool_choice", "group_id"],
        }
    },
    "doubao": {
        "init": {
            "required": [],
            "supported": [],
        },
        "create": {
            "supported": [],
        }
    },
    "kimi": {
        "init": {
            "required": [],
            "supported": [],
        },
        "create": {
            "supported": [],
        }
    }
}


def get_provider_name(model: str) -> str:
    model_lower = model.lower()
    if "minimax" in model_lower:
        return "minimax"
    elif "doubao" in model_lower or "ark" in model_lower:
        return "doubao"
    elif "kimi" in model_lower or "moonshot" in model_lower:
        return "kimi"
    else:
        return "minimax"


def get_init_params_config(provider: str) -> Dict[str, List[str]]:
    if provider in PROVIDER_PARAMS:
        return PROVIDER_PARAMS[provider]["init"]
    return PROVIDER_PARAMS["minimax"]["init"]


def get_create_params_config(provider: str) -> Dict[str, List[str]]:
    if provider in PROVIDER_PARAMS:
        return PROVIDER_PARAMS[provider]["create"]
    return PROVIDER_PARAMS["minimax"]["create"]
