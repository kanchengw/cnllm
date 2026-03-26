import os
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()

def get_env(key: str, default: str = None) -> str:
    value = os.getenv(key, default)
    if not value:
        raise ValueError(f"环境变量 {key} 未设置，请检查 .env 文件")
    return value.strip()

# MiniMax
MINIMAX_API_KEY = get_env("MINIMAX_API_KEY")