"""
Qwen (通义千问) E2E 测试 - 验证客户端完整调用链

测试目标：验证 Qwen Adapter 的核心能力
1. 非流式对话
2. 流式对话
3. .still / .think / .raw 属性访问
4. Batch 调用
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from cnllm import CNLLM

API_KEY = os.getenv("QWEN_API_KEY")
MODEL = "qwen3.6-flash"


def test_nonstream_chat():
    """非流式对话"""
    if not API_KEY:
        print("SKIP: 未配置 QWEN_API_KEY")
        return

    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        resp = client.chat.create(prompt="用一句话介绍自己", thinking=False)

        assert resp is not None, "响应不应为空"
        assert resp.still is not None, "still 不应为空"
        assert len(resp.still) > 0, "回复内容不应为空"
        assert resp.think == "", "非思考模型 think 应空字符串"
        assert resp.raw is not None, "raw 不应为空"
        assert "choices" in resp.raw, "raw 应包含 choices"
        assert "usage" in resp.raw, "raw 应包含 usage"

        print(f"[PASS] 非流式对话")
        print(f"  response: {resp.still[:60]}...")
        print(f"  usage: {resp.raw.get('usage', {})}")
    finally:
        client.close()


def test_stream_chat():
    """流式对话"""
    if not API_KEY:
        print("SKIP: 未配置 QWEN_API_KEY")
        return

    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        collected = []
        resp = client.chat.create(prompt="从1数到5", stream=True)
        for chunk in resp:
            collected.append(chunk)

        assert len(collected) > 0, "应收到至少一个 chunk"
        assert resp.still is not None, "still 不应为空"
        assert len(resp.still) > 0, "累积内容不应为空"

        print(f"[PASS] 流式对话")
        print(f"  chunks: {len(collected)}")
        print(f"  accumulated: {resp.still[:60]}...")
    finally:
        client.close()


def test_batch_chat():
    """批量对话"""
    if not API_KEY:
        print("SKIP: 未配置 QWEN_API_KEY")
        return

    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        resp = client.chat.batch(
            prompt=["你好", "1+1等于多少"],
        )

        assert resp.status["total"] == 2, "应有 2 个请求"
        assert resp.status["success_count"] == 2, "应有 2 个成功"
        assert resp.status["fail_count"] == 0, "应无失败"

        assert "request_0" in resp.still, "应包含 request_0"
        assert "request_1" in resp.still, "应包含 request_1"
        assert len(resp.still["request_0"]) > 0, "request_0 回复不应为空"

        print(f"[PASS] 批量对话")
        print(f"  status: {resp.status}")
        for rid, content in resp.still.items():
            print(f"  {rid}: {content[:40]}...")
    finally:
        client.close()


def test_messages_input():
    """标准 messages 输入"""
    if not API_KEY:
        print("SKIP: 未配置 QWEN_API_KEY")
        return

    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        resp = client.chat.create(
            messages=[{"role": "user", "content": "请输出 hello"}]
        )

        assert resp.still is not None, "still 不应为空"
        assert "hello" in resp.still.lower() or "Hello" in resp.still, "回复应包含 hello"

        print(f"[PASS] messages 输入")
        print(f"  response: {resp.still[:60]}...")
    finally:
        client.close()


def test_stream_batch():
    """流式批量对话"""
    if not API_KEY:
        print("SKIP: 未配置 QWEN_API_KEY")
        return

    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        chunk_count = 0
        for chunk in client.chat.batch(
            prompt=["你好", "介绍一下自己"],
            stream=True
        ):
            chunk_count += 1

        assert chunk_count > 0, "应收到至少一个 chunk"
        batch_resp = client.chat.batch_result
        assert batch_resp.status["total"] == 2, "应有 2 个请求"

        print(f"[PASS] 流式批量对话")
        print(f"  chunks: {chunk_count}")
        print(f"  status: {batch_resp.status}")
    finally:
        client.close()


if __name__ == "__main__":
    test_nonstream_chat()
    test_stream_chat()
    test_batch_chat()
    test_messages_input()
    test_stream_batch()
    print("\n所有 Qwen E2E 测试完成!")
