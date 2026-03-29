"""
CNLLM v0.3.1 系统测试
使用真实 API 调用
测试完整数据链传递
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from cnllm import CNLLM

API_KEY = os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_API_KEY")
if not API_KEY:
    if "__pytest__" in sys.modules or "pytest" in sys.modules:
        import pytest
        pytest.skip("MINIMAX_API_KEY 环境变量未设置", allow_module_level=True)
    else:
        print("请设置 MINIMAX_API_KEY 环境变量")
        sys.exit(1)


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode())


def test_full_chain_prompt_to_response():
    """测试完整链：prompt -> messages -> API -> OpenAI格式"""
    print("\n" + "=" * 60)
    print("test_full_chain_prompt_to_response")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    prompt = "用一句话介绍自己"
    print(f"输入: {prompt}")

    resp = client(prompt)

    assert "id" in resp
    assert "object" in resp
    assert "created" in resp
    assert "model" in resp
    assert "choices" in resp
    assert "usage" in resp

    choice = resp["choices"][0]
    assert "message" in choice
    assert "role" in choice["message"]
    assert "content" in choice["message"]
    assert "finish_reason" in choice

    content = choice["message"]["content"]
    assert len(content) > 0

    print(f"模型: {resp['model']}")
    safe_print(f"回复: {content}")
    print(f"Usage: {resp['usage']}")
    print("[PASS]")


def test_chain_with_messages():
    """测试 messages 格式完整链"""
    print("\n" + "=" * 60)
    print("test_chain_with_messages")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好"}
    ]
    print(f"输入: {messages}")

    resp = client.chat.create(messages=messages)

    assert "choices" in resp
    content = resp["choices"][0]["message"]["content"]
    assert len(content) > 0

    safe_print(f"回复: {content}")
    print("[PASS]")


def test_chain_model_override():
    """测试模型覆盖机制"""
    print("\n" + "=" * 60)
    print("test_chain_model_override")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(
        prompt="介绍自己",
        model="minimax-m2.5"
    )
    assert resp["model"] == "minimax-m2.5"

    content = resp["choices"][0]["message"]["content"]
    print(f"覆盖模型: minimax-m2.5 -> {resp['model']}")
    safe_print(f"回复: {content}")
    print("[PASS]")


def test_chain_temperature():
    """测试温度参数"""
    print("\n" + "=" * 60)
    print("test_chain_temperature")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(
        prompt="说一个随机数",
        temperature=0.9
    )
    assert "choices" in resp

    content = resp["choices"][0]["message"]["content"]
    print(f"Temperature: 0.9")
    safe_print(f"回复: {content}")
    print("[PASS]")


def test_chain_max_tokens():
    """测试max_tokens参数"""
    print("\n" + "=" * 60)
    print("test_chain_max_tokens")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(
        prompt="写一个很长的自我介绍",
        max_tokens=50
    )
    assert "choices" in resp

    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"MaxTokens: 50")
    safe_print(f"回复: {content}")
    print(f"Usage: {usage}")
    print("[PASS]")


def test_usage_stats():
    """测试usage统计"""
    print("\n" + "=" * 60)
    print("test_usage_stats")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client("hi")
    usage = resp["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] > 0

    print(f"Usage: {usage}")
    print("[PASS]")


def test_stream_output():
    """测试 chat.create 流式输出"""
    print("\n" + "=" * 60)
    print("test_stream_output")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    chunks = []
    print("流式输出: ", end="", flush=True)
    for chunk in client.chat.create(prompt="数到3", stream=True):
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if content:
            chunks.append(content)
            print(content, end="", flush=True)

    full_content = "".join(chunks)
    print()
    assert len(chunks) > 0
    assert len(full_content) > 0

    safe_print(f"总chunks: {len(chunks)}, 完整内容: {full_content}")
    print("[PASS]")


def test_runnable_full_chain():
    """测试 Runnable: 输入 -> LC转换 -> API -> AIMessage"""
    print("\n" + "=" * 60)
    print("test_runnable_full_chain")
    print("=" * 60)

    from cnllm.adapters.framework import LangChainRunnable

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    runnable = LangChainRunnable(client)
    result = runnable.invoke("你好")

    assert hasattr(result, 'content')
    assert hasattr(result, 'type')
    assert result.type == "ai"
    assert len(result.content) > 0

    print(f"输入: 你好")
    print(f"输出类型: {result.type}")
    print(f"输出内容: {result.content}")
    print("[PASS]")


def test_chat_still():
    """测试 client.chat.still 纯净文本获取"""
    print("\n" + "=" * 60)
    print("test_chat_still")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])

    still_text = client.chat.still
    assert isinstance(still_text, str)
    assert len(still_text) > 0
    assert still_text == resp["choices"][0]["message"]["content"]

    safe_print(f"still: {still_text}")
    print("[PASS]")


def test_chat_raw():
    """测试 client.chat.raw 完整原始响应"""
    print("\n" + "=" * 60)
    print("test_chat_raw")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])

    raw = client.chat.raw
    assert isinstance(raw, dict)
    assert len(raw) > 0
    assert "base_resp" in raw
    assert "status_code" in raw["base_resp"]

    print(f"raw keys: {list(raw.keys())}")
    print(f"base_resp: {raw['base_resp']}")
    print("[PASS]")


def test_api_key_override():
    """测试调用入口 api_key 参数覆盖"""
    print("\n" + "=" * 60)
    print("test_api_key_override")
    print("=" * 60)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    resp = client.chat.create(
        messages=[{"role": "user", "content": "你是哪个模型？"}],
        model="minimax-m2.5",
        api_key=API_KEY
    )
    assert resp["model"] == "minimax-m2.5"

    content = resp["choices"][0]["message"]["content"]
    safe_print(f"回复: {content}")
    print("[PASS]")


def test_model_required():
    """测试客户端入口 model 必填"""
    print("\n" + "=" * 60)
    print("test_model_required")
    print("=" * 60)

    try:
        client = CNLLM(api_key=API_KEY)
        print("[FAIL] 应该抛出 TypeError")
        assert False
    except TypeError as e:
        print(f"[PASS] 正确抛出 TypeError: {e}")


def test_fallback_success():
    """测试 Fallback 机制成功切换"""
    print("\n" + "=" * 60)
    print("test_fallback_success")
    print("=" * 60)

    client = CNLLM(
        model="minimax-m2.7",
        api_key=API_KEY,
        fallback_models={
            "minimax-m2.5": None
        }
    )
    resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])
    assert "choices" in resp
    content = resp["choices"][0]["message"]["content"]
    assert len(content) > 0

    safe_print(f"回复: {content}")
    print("[PASS]")


def test_raw_with_fallback():
    """测试 fallback 后 raw 仍可访问"""
    print("\n" + "=" * 60)
    print("test_raw_with_fallback")
    print("=" * 60)

    client = CNLLM(
        model="nonexistent-model",
        api_key=API_KEY,
        fallback_models={
            "minimax-m2.7": None
        }
    )
    resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])

    raw = client.chat.raw
    assert isinstance(raw, dict)
    assert "base_resp" in raw

    print(f"raw base_resp: {raw['base_resp']}")
    print("[PASS]")


if __name__ == "__main__":
    test_full_chain_prompt_to_response()
    test_chain_with_messages()
    test_chain_model_override()
    test_chain_temperature()
    test_chain_max_tokens()
    test_usage_stats()
    test_stream_output()
    test_runnable_full_chain()
    test_chat_still()
    test_chat_raw()
    test_api_key_override()
    test_model_required()
    test_fallback_success()
    test_raw_with_fallback()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)