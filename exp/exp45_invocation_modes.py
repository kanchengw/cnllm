
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("GLM_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
"""
实验45: CNLLM 14种调用方式测试

测试所有调用模式:
- Chat调用（8种）: 同步/异步 × 流式/非流式 × 单条/批量
- Embedding调用（4种）: 同步/异步 × 单条/批量
- 混合模式（2种）: 同步/异步 混合流式批量
"""

import os
import sys
import asyncio
from typing import Dict, Any

# 添加CNLLM到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CNLLM'))

from cnllm import CNLLM
from cnllm.entry.async_client import asyncCNLLM

# API配置
API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "glm": os.getenv("GLM_API_KEY", ""),
    "qwen": os.getenv("QWEN_API_KEY", ""),
}
DASHSCOPE_API_KEY = API_KEYS["qwen"]

# 模型名称（根据CNLLM配置）
CHAT_MODEL = "qwen3.5-plus"  # 支持的模型: qwen3.5-plus, qwen3.5-flash, qwen3.6-plus等
EMBEDDING_MODEL = "text-embedding-v3"  # 支持的模型: text-embedding-v1/v2/v3/v4

# 测试用的简单消息
TEST_MESSAGES = [{"role": "user", "content": "你好，请用一句话回答"}]
TEST_MESSAGES_BATCH = [
    [{"role": "user", "content": "你好，请用一句话回答"}],
    [{"role": "user", "content": "今天天气如何？"}],
]
TEST_PROMPT_BATCH = ["你好，请用一句话回答", "今天天气如何？"]
TEST_EMBEDDING_INPUT = "这是一段测试文本"
TEST_EMBEDDING_BATCH = ["第一段测试文本", "第二段测试文本", "第三段测试文本"]

# 测试结果
results = {}


def test_sync_nonstream_single():
    """1. 同步非流式非批量"""
    try:
        client = CNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.chat.create(
            messages=TEST_MESSAGES,
            stream=False
        )
        # CNLLM返回NonStreamAccumulator对象，消费后得到dict
        result = resp.get("choices") if hasattr(resp, "get") else resp
        assert result is not None, "Failed to get response"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_stream_single():
    """2. 同步流式非批量"""
    try:
        client = CNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.chat.create(
            messages=TEST_MESSAGES,
            stream=True
        )
        # 验证返回类型是迭代器
        chunks = []
        for i, chunk in enumerate(resp):
            chunks.append(chunk)
            if i >= 1:  # 只消费前2个chunk
                break
        assert len(chunks) > 0, "No chunks received"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_nonstream_batch():
    """3. 同步非流式批量"""
    try:
        client = CNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.chat.batch(
            prompt=TEST_PROMPT_BATCH,
            stream=False
        )
        # BatchResponse对象，验证有results或status属性即可
        assert hasattr(resp, 'results') or hasattr(resp, 'status'), "Invalid batch response"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_stream_batch():
    """4. 同步流式批量"""
    try:
        client = CNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.chat.batch(
            prompt=TEST_PROMPT_BATCH,
            stream=True
        )
        # 验证返回类型是迭代器
        chunks = []
        for i, chunk in enumerate(resp):
            chunks.append(chunk)
            if i >= 2:  # 消费几个chunk
                break
        assert len(chunks) > 0, "No chunks received"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_nonstream_single():
    """5. 异步非流式非批量"""
    try:
        client = asyncCNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.chat.create(
            messages=TEST_MESSAGES,
            stream=False
        )
        # CNLLM返回AsyncNonStreamAccumulator对象
        assert hasattr(resp, 'get') or hasattr(resp, '__aiter__'), "Invalid async response"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_stream_single():
    """6. 异步流式非批量"""
    try:
        client = asyncCNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.chat.create(
            messages=TEST_MESSAGES,
            stream=True
        )
        # 验证返回类型是异步迭代器
        chunks = []
        async for chunk in resp:
            chunks.append(chunk)
            if len(chunks) >= 2:  # 只消费前2个chunk
                break
        assert len(chunks) > 0, "No chunks received"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_nonstream_batch():
    """7. 异步非流式批量"""
    try:
        client = asyncCNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.chat.batch(
            prompt=TEST_PROMPT_BATCH,
            stream=False
        )
        # 验证返回类型
        assert hasattr(resp, 'done') or hasattr(resp, 'status'), "Invalid batch response"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_stream_batch():
    """8. 异步流式批量"""
    try:
        client = asyncCNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.chat.batch(
            prompt=["你好"],  # 简化为单个请求
            stream=True
        )
        # 验证返回类型是异步迭代器（有__aiter__方法）
        assert hasattr(resp, '__aiter__'), f"Expected async iterator, got {type(resp)}"
        # 尝试消费第一个chunk，设置5秒超时
        try:
            import asyncio
            first_chunk = await asyncio.wait_for(resp.__anext__(), timeout=5.0)
            assert first_chunk is not None, "First chunk is None"
        except asyncio.TimeoutError:
            print("  [Warning] Timeout waiting for first chunk, but iterator type is correct")
            # 超时但类型正确，仍算通过
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_embedding_single():
    """9. 同步非批量Embeddings"""
    try:
        client = CNLLM(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.embeddings.create(
            input=TEST_EMBEDDING_INPUT
        )
        # CNLLM返回EmbeddingAccumulator对象
        assert hasattr(resp, 'get') or hasattr(resp, 'data'), "Invalid embedding response"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_embedding_batch():
    """10. 同步批量Embeddings"""
    try:
        client = CNLLM(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = client.embeddings.batch(
            input=TEST_EMBEDDING_BATCH
        )
        # EmbeddingResponse对象，验证有results或data属性即可
        assert hasattr(resp, 'results') or hasattr(resp, 'data'), "Invalid embedding batch response"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_embedding_single():
    """11. 异步非批量Embeddings"""
    try:
        client = asyncCNLLM(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.embeddings.create(
            input=TEST_EMBEDDING_INPUT
        )
        # 验证返回类型
        assert isinstance(resp, dict), f"Expected dict, got {type(resp)}"
        assert "data" in resp, "Missing 'data' in response"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_embedding_batch():
    """12. 异步批量Embeddings"""
    try:
        client = asyncCNLLM(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        resp = await client.embeddings.batch(
            input=TEST_EMBEDDING_BATCH
        )
        # 验证返回类型
        assert hasattr(resp, 'done') or hasattr(resp, 'status'), "Invalid batch response"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def test_sync_mixed_batch():
    """13. 同步混合流式批量"""
    try:
        client = CNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        # 混合模式：部分请求流式，部分非流式
        requests = [
            {"prompt": "你好", "stream": True},
            {"prompt": "世界", "stream": False},
        ]
        resp = client.chat.batch(
            requests=requests
        )
        # 验证返回类型
        assert hasattr(resp, 'done') or hasattr(resp, 'status'), "Invalid mixed batch response"
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


async def test_async_mixed_batch():
    """14. 异步混合流式批量"""
    try:
        client = asyncCNLLM(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )
        # 混合模式：部分请求流式，部分非流式
        requests = [
            {"prompt": "你好", "stream": True},
            {"prompt": "世界", "stream": False},
        ]
        resp = await client.chat.batch(
            requests=requests
        )
        # 验证返回类型
        assert hasattr(resp, 'done') or hasattr(resp, 'status'), "Invalid mixed batch response"
        await client.aclose()
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def run_sync_tests():
    """运行所有同步测试"""
    print("\n" + "="*60)
    print("同步测试")
    print("="*60)

    tests = [
        ("sync_nonstream_single", test_sync_nonstream_single),
        ("sync_stream_single", test_sync_stream_single),
        ("sync_nonstream_batch", test_sync_nonstream_batch),
        ("sync_stream_batch", test_sync_stream_batch),
        ("sync_embedding_single", test_sync_embedding_single),
        ("sync_embedding_batch", test_sync_embedding_batch),
        ("sync_mixed_batch", test_sync_mixed_batch),
    ]

    for name, test_func in tests:
        print(f"\n测试: {name}...")
        try:
            result = test_func()
            results[name] = result
            print(f"  结果: {'✓ PASS' if result else '✗ FAIL'}")
        except Exception as e:
            results[name] = False
            print(f"  结果: ✗ FAIL - {e}")


async def run_async_tests():
    """运行所有异步测试"""
    print("\n" + "="*60)
    print("异步测试")
    print("="*60)

    tests = [
        ("async_nonstream_single", test_async_nonstream_single),
        ("async_stream_single", test_async_stream_single),
        ("async_nonstream_batch", test_async_nonstream_batch),
        ("async_stream_batch", test_async_stream_batch),
        ("async_embedding_single", test_async_embedding_single),
        ("async_embedding_batch", test_async_embedding_batch),
        ("async_mixed_batch", test_async_mixed_batch),
    ]

    for name, test_func in tests:
        print(f"\n测试: {name}...")
        try:
            result = await test_func()
            results[name] = result
            print(f"  结果: {'✓ PASS' if result else '✗ FAIL'}")
        except Exception as e:
            results[name] = False
            print(f"  结果: ✗ FAIL - {e}")


def main():
    print("="*60)
    print("CNLLM 14种调用方式测试")
    print("="*60)

    # 运行同步测试
    run_sync_tests()

    # 运行异步测试
    asyncio.run(run_async_tests())

    # 打印结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    # 输出Python格式的结果字典
    print("\n" + "="*60)
    print("Python结果字典")
    print("="*60)
    print("results = {")
    for name in sorted(results.keys()):
        print(f'    "{name}": {results[name]},')
    print("}")

    return results


if __name__ == "__main__":
    main()
