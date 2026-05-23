"""
E2E 测试：字段访问（.think / .still / .tools / .raw）

安排：
- ernie-4.5-turbo + thinking → .think / .raw
- hy3-preview + thinking → .think / .raw
- kimi-k2.5 + thinking → .think / .raw
- glm-4.7 + tools → .tools / .raw
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()
import pytest
from cnllm import CNLLM

BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
HUNYUAN_API_KEY = os.getenv("HUNYUAN_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
GLM_API_KEY = os.getenv("GLM_API_KEY")

requires_baidu = pytest.mark.skipif(not BAIDU_API_KEY, reason="需要 BAIDU_API_KEY")
requires_hunyuan = pytest.mark.skipif(not HUNYUAN_API_KEY, reason="需要 HUNYUAN_API_KEY")
requires_kimi = pytest.mark.skipif(not KIMI_API_KEY, reason="需要 KIMI_API_KEY")
requires_glm = pytest.mark.skipif(not GLM_API_KEY, reason="需要 GLM_API_KEY")


def _get_weather_tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名"}
                },
                "required": ["location"]
            }
        }
    }]


# ========== thinking 字段测试 ==========

@requires_baidu
def test_ernie_thinking():
    """ernie-5.1 + thinking"""
    client = CNLLM(model="ernie-5.1", api_key=BAIDU_API_KEY)
    resp = client.chat.create(
        messages=[{"role": "user", "content": "用一句话介绍北京"}],
        thinking=True,
    )
    print(f"[ernie] raw={resp.raw}")
    print(f"[ernie] think={repr(resp.think)}")
    print(f"[ernie] still={repr(resp.still)}")
    assert resp.think is not None, ".think 不应为空"
    assert isinstance(resp.think, str), f".think 应为 str, 实际 {type(resp.think)}"
    assert len(resp.think) > 0, ".think 不应为空字符串"
    assert resp.raw is not None, ".raw 不应为空"
    print(f"[ernie-think] think len={len(resp.think)}, still len={len(resp.still)}")


@requires_hunyuan
def test_hunyuan_thinking():
    """hy3-preview + thinking → .think / .raw"""
    client = CNLLM(model="hy3-preview", api_key=HUNYUAN_API_KEY)
    resp = client.chat.create(
        messages=[{"role": "user", "content": "用一句话介绍北京"}],
        thinking=True,
    )
    assert resp.think is not None, ".think 不应为空"
    assert isinstance(resp.think, str), f".think 应为 str, 实际 {type(resp.think)}"
    assert len(resp.think) > 0, ".think 不应为空字符串"
    assert resp.raw is not None, ".raw 不应为空"
    print(f"[hunyuan-think] think len={len(resp.think)}, still len={len(resp.still)}")


@requires_kimi
def test_kimi_thinking():
    """kimi-k2.5 + thinking → .think / .raw"""
    client = CNLLM(model="kimi-k2.5", api_key=KIMI_API_KEY)
    resp = client.chat.create(
        messages=[{"role": "user", "content": "用一句话介绍北京"}],
        thinking=True,
    )
    assert resp.think is not None, ".think 不应为空"
    assert isinstance(resp.think, str), f".think 应为 str, 实际 {type(resp.think)}"
    assert len(resp.think) > 0, ".think 不应为空字符串"
    assert resp.raw is not None, ".raw 不应为空"
    print(f"[kimi-think] think len={len(resp.think)}, still len={len(resp.still)}")


# ========== tools 字段测试 ==========

@requires_glm
def test_glm_tools():
    """glm-4.7 + tools → .tools / .raw"""
    client = CNLLM(model="glm-4.7", api_key=GLM_API_KEY)
    tools = _get_weather_tools()
    resp = client.chat.create(
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools,
    )
    assert resp.tools is not None, ".tools 不应为空"
    assert len(resp.tools) > 0, ".tools 不应为空字典"
    # 验证 tool 调用结构
    first_tool = resp.tools[0] if isinstance(resp.tools, dict) else resp.tools[0]
    tool_id = first_tool.get("id", first_tool.get("index", None))
    assert tool_id is not None, f"tool 调用缺少 id/index: {first_tool}"
    assert resp.raw is not None, ".raw 不应为空"
    print(f"[glm-tools] tools={resp.tools}, raw keys={list(resp.raw.keys()) if isinstance(resp.raw, dict) else 'ok'}")
