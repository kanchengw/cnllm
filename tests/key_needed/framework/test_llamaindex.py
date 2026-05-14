"""
LlamaIndex 兼容性 E2E 测试。
测试 CNLLM 的输出能否被 LlamaIndex 消费。
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

pytest.importorskip("llama_index.core")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"


def test_chatmessage_construction():
    """CNLLM 响应 → LlamaIndex ChatMessage 构造"""
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from llama_index.core.llms import ChatMessage, MessageRole
    from cnllm import CNLLM

    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.create(prompt="1+1=?", stream=False)
    content = resp["choices"][0]["message"]["content"]
    msg = ChatMessage(role=MessageRole.ASSISTANT, content=content)

    assert msg.role == MessageRole.ASSISTANT
    assert len(msg.content) > 0
    client.close()


def test_multi_round():
    """CNLLM 输出 → LlamaIndex 多轮对话"""
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from llama_index.core.llms import ChatMessage, MessageRole
    from cnllm import CNLLM

    client = CNLLM(model=MODEL, api_key=API_KEY)

    resp1 = client.chat.create(messages=[{"role": "user", "content": "我的名字是小红"}], stream=False)
    assistant_content = resp1["choices"][0]["message"]["content"]

    history = [
        ChatMessage(role=MessageRole.USER, content="我的名字是小红"),
        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_content),
        ChatMessage(role=MessageRole.USER, content="我叫什么名字？"),
    ]
    assert len(history) == 3
    assert history[1].role == MessageRole.ASSISTANT

    resp2 = client.chat.create(messages=[
        {"role": "user", "content": "我的名字是小红"},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "我叫什么名字？"},
    ], stream=False)
    content2 = resp2["choices"][0]["message"]["content"]
    assert "小红" in content2, f"多轮对话丢失上下文: {content2}"
    client.close()


def test_streaming_chatmessage():
    """CNLLM 流式累积 → LlamaIndex ChatMessage"""
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from llama_index.core.llms import ChatMessage, MessageRole
    from cnllm import CNLLM

    client = CNLLM(model=MODEL, api_key=API_KEY)
    acc = ""
    for chunk in client.chat.create(prompt="数到3", stream=True):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta:
            acc += delta

    msg = ChatMessage(role=MessageRole.ASSISTANT, content=acc)
    assert len(msg.content) > 0
    assert "1" in msg.content or "2" in msg.content or "3" in msg.content
    client.close()
