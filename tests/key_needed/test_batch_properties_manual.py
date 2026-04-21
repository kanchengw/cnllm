"""
批量调用属性访问测试 - 人工审核验证

测试目标：
1. 验证同步流式批量调用的 .raw/.think/.still/.tools 属性访问
2. 验证异步流式批量调用的属性访问
3. 验证同步非流式批量调用的属性访问
4. 验证异步非流式批量调用的属性访问
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM, asyncCNLLM as AsyncCNLLM

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
GLM_API_KEY = os.getenv("GLM_API_KEY")


def test_sync_stream_batch_properties():
    """同步流式批量 - 属性访问测试"""
    print("\n" + "=" * 70)
    print("【同步流式批量 - 属性访问测试】")
    print("=" * 70)

    if not MINIMAX_API_KEY:
        print("⚠️  需要 MINIMAX_API_KEY，跳过测试")
        return

    client = CNLLM(model="minimax-m2", api_key=MINIMAX_API_KEY, thinking=True)

    requests = [
        {"messages": [{"role": "user", "content": "你好"}]},
        {"messages": [{"role": "user", "content": "1+1等于几？"}]},
        {"messages": [{"role": "user", "content": "写一首诗"}]},
    ]

    print("\n📌 批量请求数量:", len(requests))
    print("-" * 70)

    response = client.chat.batch(requests, stream=True)

    print("\n🔄 开始迭代 (实时输出):")
    print("-" * 70)

    chunk_count = 0
    for chunk in response:
        chunk_count += 1
        request_id = chunk.get("request_id")
        index = int(request_id.split("_")[-1])

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")

        if chunk_count <= 10:
            content_preview = content[:20] if content else "(空)"
            print(f"[Chunk {chunk_count:2d}] request_id={request_id}, content='{content_preview}...' " if content else f"[Chunk {chunk_count:2d}] request_id={request_id}, content='{content}'")

        if chunk_count == 10:
            print(f"\n📊 实时查看 batch_result 属性 (前10个chunks后):")
            print("-" * 70)
            result = client.chat.batch_result

            print(f"\n1. result.think (所有请求的推理内容):")
            for cid, think in result.think.items():
                preview = think[:30] if think else "(空)"
                print(f"   {cid}: {preview}...")

            print(f"\n2. result.still (所有请求的回复内容):")
            for cid, still in result.still.items():
                preview = still[:30] if still else "(空)"
                print(f"   {cid}: {preview}...")

            print(f"\n3. result.tools (所有请求的工具调用):")
            for cid, tools in result.tools.items():
                print(f"   {cid}: {tools}")

            print(f"\n4. result.raw (每个请求的原始数据):")
            for cid, raw in result.raw.items():
                chunk_count_in_raw = len(raw.get("chunks", []))
                print(f"   {cid}: {chunk_count_in_raw} chunks")

            print(f"\n5. 按索引访问单个请求:")
            for i in range(len(requests)):
                item = result[i]
                print(f"   result[{i}].think: {item.think[:30] if item.think else '(空)'}...")
                print(f"   result[{i}].still: {item.still[:30] if item.still else '(空)'}...")

    print("\n" + "-" * 70)
    print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
    print("-" * 70)

    print("\n📊 迭代完成后各属性值:")
    print("-" * 70)
    result = client.chat.batch_result

    print(f"\n1. result.think:")
    for cid, think in result.think.items():
        print(f"   {cid}: {think}")

    print(f"\n2. result.still:")
    for cid, still in result.still.items():
        print(f"   {cid}: {still}")

    print(f"\n3. result.tools:")
    for cid, tools in result.tools.items():
        print(f"   {cid}: {tools}")

    print(f"\n4. result.raw:")
    for cid, raw in result.raw.items():
        chunk_count_in_raw = len(raw.get("chunks", []))
        print(f"   {cid}: {chunk_count_in_raw} chunks")

    print(f"\n5. 按 request_id 访问:")
    print(f"   result['request_0'].think: {result['request_0'].think}")
    print(f"   result['request_1'].still: {result['request_1'].still}")
    print(f"   result['request_2'].raw chunks: {len(result['request_2'].raw.get('chunks', []))}")

    print("\n" + "=" * 70)
    print("【同步流式批量测试结束】")
    print("=" * 70)


async def test_async_stream_batch_properties():
    """异步流式批量 - 属性访问测试"""
    print("\n" + "=" * 70)
    print("【异步流式批量 - 属性访问测试】")
    print("=" * 70)

    if not MINIMAX_API_KEY:
        print("⚠️  需要 MINIMAX_API_KEY，跳过测试")
        return

    client = AsyncCNLLM(model="minimax-m2", api_key=MINIMAX_API_KEY, thinking=True)

    requests = [
        {"messages": [{"role": "user", "content": "你好"}]},
        {"messages": [{"role": "user", "content": "1+1等于几？"}]},
    ]

    print("\n📌 批量请求数量:", len(requests))
    print("-" * 70)

    chunk_count = 0
    batch_iterator = client.chat.batch(requests, stream=True)
    async for chunk in batch_iterator:
        chunk_count += 1
        request_id = chunk.get("request_id")
        index = int(request_id.split("_")[-1])

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")

        if chunk_count <= 5:
            content_preview = content[:20] if content else "(空)"
            print(f"[Chunk {chunk_count:2d}] request_id={request_id}, content='{content_preview}...' " if content else f"[Chunk {chunk_count:2d}] request_id={request_id}, content='{content}'")

        if chunk_count == 5:
            print(f"\n📊 实时查看 batch_result 属性 (前5个chunks后):")
            print("-" * 70)
            result = client.chat.batch_result

            print(f"\n1. result.think:")
            for cid, think in result.think.items():
                preview = think[:30] if think else "(空)"
                print(f"   {cid}: {preview}...")

            print(f"\n2. result.still:")
            for cid, still in result.still.items():
                preview = still[:30] if still else "(空)"
                print(f"   {cid}: {preview}...")

    print("\n" + "-" * 70)
    print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
    print("-" * 70)

    print("\n📊 迭代完成后各属性值:")
    print("-" * 70)
    result = client.chat.batch_result

    print(f"\n1. result.think:")
    for cid, think in result.think.items():
        print(f"   {cid}: {think}")

    print(f"\n2. result.still:")
    for cid, still in result.still.items():
        print(f"   {cid}: {still}")

    print("\n" + "=" * 70)
    print("【异步流式批量测试结束】")
    print("=" * 70)


def test_sync_nonstream_batch_properties():
    """同步非流式批量 - 属性访问测试"""
    print("\n" + "=" * 70)
    print("【同步非流式批量 - 属性访问测试】")
    print("=" * 70)

    if not GLM_API_KEY:
        print("⚠️  需要 GLM_API_KEY，跳过测试")
        return

    client = CNLLM(model="GLM-5.1", api_key=GLM_API_KEY, thinking=True)

    requests = [
        {"messages": [{"role": "user", "content": "你好"}]},
        {"messages": [{"role": "user", "content": "1+1等于几？"}]},
        {"messages": [{"role": "user", "content": "写一首诗"}]},
    ]

    print("\n📌 批量请求数量:", len(requests))
    print("-" * 70)

    result = client.chat.batch(requests, stream=False)

    print("\n📊 非流式批量结果:")
    print("-" * 70)

    print(f"\n📈 批量统计:")
    print(f"   总计: {result.total}")
    print(f"   成功: {result.success_count}")
    print(f"   失败: {result.fail_count}")
    print(f"   耗时: {result.elapsed:.2f}s")

    if result.errors:
        print(f"\n❌ 失败详情:")
        for err in result.errors:
            print(f"   {err}")

    print(f"\n1. result.think:")
    for cid, think in result.think.items():
        print(f"   {cid}: {think}")

    print(f"\n2. result.still:")
    for cid, still in result.still.items():
        print(f"   {cid}: {still}")

    print(f"\n3. result.tools:")
    for cid, tools in result.tools.items():
        print(f"   {cid}: {tools}")

    print(f"\n4. result.raw:")
    for cid, raw in result.raw.items():
        keys = list(raw.keys())[:5]
        print(f"   {cid}: {keys}...")

    print(f"\n5. 按索引访问单个请求:")
    for i in range(len(requests)):
        item = result.get_item(i)
        if item is None:
            print(f"   result[{i}]: (请求失败或无响应)")
            continue
        print(f"   result[{i}]:")
        print(f"     .think: {item.think}")
        print(f"     .still: {item.still[:50] if item.still else '(空)'}...")
        print(f"     .tools: {item.tools}")

    print("\n" + "=" * 70)
    print("【同步非流式批量测试结束】")
    print("=" * 70)


async def test_async_nonstream_batch_properties():
    """异步非流式批量 - 属性访问测试"""
    print("\n" + "=" * 70)
    print("【异步非流式批量 - 属性访问测试】")
    print("=" * 70)

    if not GLM_API_KEY:
        print("⚠️  需要 GLM_API_KEY，跳过测试")
        return

    client = AsyncCNLLM(model="GLM-5.1", api_key=GLM_API_KEY, thinking=True)

    requests = [
        {"messages": [{"role": "user", "content": "你好"}]},
        {"messages": [{"role": "user", "content": "1+1等于几？"}]},
    ]

    print("\n📌 批量请求数量:", len(requests))
    print("-" * 70)

    result = None
    async for r in client.chat.batch(requests, stream=False):
        result = r

    print("\n📊 异步非流式批量结果:")
    print("-" * 70)

    print(f"\n1. result.think:")
    for cid, think in result.think.items():
        print(f"   {cid}: {think}")

    print(f"\n2. result.still:")
    for cid, still in result.still.items():
        print(f"   {cid}: {still}")

    print(f"\n3. result.tools:")
    for cid, tools in result.tools.items():
        print(f"   {cid}: {tools}")

    print(f"\n4. 按索引访问单个请求:")
    for i in range(len(requests)):
        item = result.get_item(i)
        if item is None:
            print(f"   result[{i}]: (请求不存在)")
        elif not item.is_success:
            print(f"   result[{i}]: (请求失败: {item.error})")
        else:
            print(f"   result[{i}]: ✅ 成功")
            print(f"     .think: {item.think[:30] if item.think else '(空)'}...")
            print(f"     .still: {item.still[:30] if item.still else '(空)'}...")

    print(f"\n5. 演示访问不存在的索引:")
    print(f"   result[999]: {result.get_item(999)}")
    print(f"   result['request_999']: {result.get_item('request_999')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("批量调用属性访问测试")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1. 同步流式批量")
    print("=" * 70)
    test_sync_stream_batch_properties()

    print("\n" + "=" * 70)
    print("2. 同步非流式批量")
    print("=" * 70)
    test_sync_nonstream_batch_properties()

    print("\n" + "=" * 70)
    print("3. 异步流式批量")
    print("=" * 70)
    asyncio.run(test_async_stream_batch_properties())

    print("\n" + "=" * 70)
    print("4. 异步非流式批量")
    print("=" * 70)
    asyncio.run(test_async_nonstream_batch_properties())

    print("\n" + "=" * 70)
    print("✅ 所有批量测试完成")
    print("=" * 70)
