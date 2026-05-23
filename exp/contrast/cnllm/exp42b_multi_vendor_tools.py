
"""
实验一 场景 d: 多厂商混合批量 + tools 对比

实现: CNLLM
对比 baseline: 三次独立 SDK 调用 + 手动聚合
CNLLM 一次 batch() + per-request api_key + .tools 自动聚合
"""

import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "your-glm-api-key")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "your-qwen-api-key")

TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如北京、上海"},
            },
            "required": ["city"],
        },
    },
}]


def cnllm_impl():
    """CNLLM 实现：一次 batch() + per-request api_key + .tools"""
    from cnllm import CNLLM

    client = CNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)

    results = client.chat.batch(
        requests=[
            {"prompt": "北京今天天气怎么样？", "tools": TOOL_DEF, "model": "deepseek-chat"},
            {"prompt": "上海今天天气怎么样？", "tools": TOOL_DEF, "model": "glm-4.7", "api_key": GLM_API_KEY},
            {"prompt": "广州今天天气怎么样？", "tools": TOOL_DEF, "model": "qwen3.5-flash", "api_key": QWEN_API_KEY},
        ],
    )

    print(f"[CNLLM] 批量完成: {results.status['success_count']}/{results.status['total']}")
    print(f"[CNLLM] .tools 结果: {results.tools}")
    return results.tools


if __name__ == "__main__":
    print("=" * 50)
    print("场景 d: 多厂商批量 + tools 对比 (CNLLM)")
    print("=" * 50)
    result = cnllm_impl()
    print("=" * 50)
