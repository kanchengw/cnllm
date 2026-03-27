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
            "ignored": ["default_headers", "default_query"],
            "provider_specific": []
        },
        "create": {
            "required": [],
            "supported": ["messages", "temperature", "max_tokens", "stream", "tools", "tool_choice"],
            "ignored": ["top_p", "n", "stop", "presence_penalty", "frequency_penalty", "user"],
            "provider_specific": ["group_id"]
        }
    },
    "doubao": {
        "init": {
            "required": [],
            "supported": [],
            "ignored": [],
            "provider_specific": []
        },
        "create": {
            "supported": [],
            "ignored": [],
            "provider_specific": []
        }
    },
    "kimi": {
        "init": {
            "required": [],
            "supported": [],
            "ignored": [],
            "provider_specific": []
        },
        "create": {
            "supported": [],
            "ignored": [],
            "provider_specific": []
        }
    }
}


def get_provider_name(model: str) -> str:
    """根据模型名推断提供商"""
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
    """获取指定提供商的初始化参数配置"""
    if provider in PROVIDER_PARAMS:
        return PROVIDER_PARAMS[provider]["init"]
    return PROVIDER_PARAMS["minimax"]["init"]


def get_create_params_config(provider: str) -> Dict[str, List[str]]:
    """获取指定提供商的 create 参数配置"""
    if provider in PROVIDER_PARAMS:
        return PROVIDER_PARAMS[provider]["create"]
    return PROVIDER_PARAMS["minimax"]["create"]
