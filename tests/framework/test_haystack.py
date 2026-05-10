"""
Haystack 兼容性 E2E 测试。
CNLLM 的 embedding 输出作为 Haystack 文档向量。
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

pytest.importorskip("haystack")


def test_haystack_embedding_document():
    """CNLLM embedding → Haystack Document"""
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        pytest.skip("GLM_API_KEY not set")
    from haystack import Document
    from cnllm import CNLLM

    text = "CNLLM 是一个中文大模型适配器"
    client = CNLLM(model="embedding-2", api_key=api_key)
    resp = client.embeddings.create(input=text)
    embedding = resp["data"][0]["embedding"]

    doc = Document(content=text, embedding=embedding)
    assert doc.embedding is not None
    assert len(doc.embedding) > 0
    assert isinstance(doc.embedding[0], float)
    client.close()


def test_haystack_chat_message():
    """CNLLM chat → Haystack ChatMessage"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from haystack.dataclasses import ChatMessage
    from cnllm import CNLLM

    client = CNLLM(model="deepseek-v4-flash", api_key=api_key)
    resp = client.chat.create(prompt="1+1=?", stream=False)
    content = resp["choices"][0]["message"]["content"]

    msg = ChatMessage.from_assistant(content)
    assert msg.role == "assistant"
    assert len(msg.text) > 0
    client.close()
