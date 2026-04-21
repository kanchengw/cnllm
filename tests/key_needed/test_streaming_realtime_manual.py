"""
流式实时消费测试 - 人工审核验证
测试目标：
1. 实时打印每个 chunk 的 content
2. 验证 client.chat.think/still/tools/raw 在迭代过程中的实时累积
3. 验证迭代完成后各属性的最终值
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

from cnllm import CNLLM, asyncCNLLM as AsyncCNLLM

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XIAOMI_API_KEY = os.getenv("XIAOMI_API_KEY")

def test_sync_streaming_realtime():
    """同步流式实时消费测试"""
    print("\n" + "=" * 70)
    print("【同步流式实时消费测试】")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print("⚠️  需要 DEEPSEEK_API_KEY，跳过测试")
        return

    client = CNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY,thinking=True)

    print("\n📌 发送请求: 说 '你好'")
    print("-" * 70)

    response = client.chat.create(
        messages=[{"role": "user", "content": "你好"}],
        stream=True
    )

    print("\n🔄 开始迭代 (实时输出):")
    print("-" * 70)

    chunk_count = 0
    full_content = ""

    for chunk in response:
        chunk_count += 1

        # 获取 delta 中的 content
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            full_content += content
            print(f"[Chunk {chunk_count:2d}] content: '{content}'", end="", flush=True)
        else:
            print(f"[Chunk {chunk_count:2d}] (empty delta)", end="", flush=True)

        # 实时显示各属性的当前累积状态
        think_preview = client.chat.think[:30] if client.chat.think is not None else "(空)"
        still_preview = client.chat.still[:30] if client.chat.still is not None else "(空)"

        print(f"  | .think: {think_preview}... | .still: {still_preview}...")

    print("\n" + "-" * 70)
    print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
    print("-" * 70)

    print("\n📊 迭代完成后各属性值:")
    print("-" * 70)
    print(f"client.chat.think: {client.chat.think}")
    print(f"client.chat.still: {client.chat.still}")
    print(f"client.chat.still 长度: {len(client.chat.still) if client.chat.still else 0}")
    print(f"client.chat.tools: {client.chat.tools}")
    print(f"client.chat.raw 类型: {type(client.chat.raw)}")
    print(f"client.chat.raw keys: {list(client.chat.raw.keys()) if isinstance(client.chat.raw, dict) else len(client.chat.raw)}")

    print("\n📝 client.chat.raw keys:")
    print("-" * 70)
    if isinstance(client.chat.raw, dict):
        for k in list(client.chat.raw.keys())[:5]:
            print(f"  {k}: {client.chat.raw[k]}")

    print("\n" + "=" * 70)
    print("【同步流式测试结束】")
    print("=" * 70)


async def test_async_streaming_realtime():
    """异步流式实时消费测试"""
    print("\n" + "=" * 70)
    print("【异步流式实时消费测试】")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print("⚠️  需要 DEEPSEEK_API_KEY，跳过测试")
        return

    async with AsyncCNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY) as client:
        print("\n📌 发送请求: 说 '你好'")
        print("-" * 70)

        response = await client.chat.create(
            messages=[{"role": "user", "content": "你好"}],
            stream=True
        )

        print("\n🔄 开始迭代 (实时输出):")
        print("-" * 70)

        chunk_count = 0
        full_content = ""

        async for chunk in response:
            chunk_count += 1

            # 获取 delta 中的 content
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                full_content += content
                print(f"[Chunk {chunk_count:2d}] content: '{content}'", end="", flush=True)
            else:
                print(f"[Chunk {chunk_count:2d}] (empty delta)", end="", flush=True)

            # 实时显示各属性的当前累积状态
            think_preview = client.chat.think[:30] if client.chat.think is not None else "(空)"
            still_preview = client.chat.still[:30] if client.chat.still is not None else "(空)"

            print(f"  | .think: {think_preview}... | .still: {still_preview}...")

        print("\n" + "-" * 70)
        print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
        print("-" * 70)

        print("\n📊 迭代完成后各属性值:")
        print("-" * 70)
        print(f"client.chat.think: {client.chat.think}")
        print(f"client.chat.still: {client.chat.still}")
        print(f"client.chat.still 长度: {len(client.chat.still) if client.chat.still else 0}")
        print(f"client.chat.tools: {client.chat.tools}")
        print(f"client.chat.raw 类型: {type(client.chat.raw)}")
        print(f"client.chat.raw keys: {list(client.chat.raw.keys()) if isinstance(client.chat.raw, dict) else len(client.chat.raw)}")

        print("\n📝 client.chat.raw keys:")
        print("-" * 70)
        if isinstance(client.chat.raw, dict):
            for k in list(client.chat.raw.keys())[:5]:
                print(f"  {k}: {client.chat.raw[k]}")

    print("\n" + "=" * 70)
    print("【异步流式测试结束】")
    print("=" * 70)


def test_sync_streaming_with_thinking():
    """同步流式 + thinking=true 测试"""
    print("\n" + "=" * 70)
    print("【同步流式 + thinking=true 测试】")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print("⚠️  需要 DEEPSEEK_API_KEY，跳过测试")
        return

    client = CNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)

    print("\n📌 发送请求: '1+1等于几' (thinking=true)")
    print("-" * 70)

    response = client.chat.create(
        messages=[{"role": "user", "content": "1+1等于几"}],
        thinking=True,
        stream=True
    )

    print("\n🔄 开始迭代 (实时输出):")
    print("-" * 70)

    chunk_count = 0
    think_content = ""
    still_content = ""

    for chunk in response:
        chunk_count += 1

        # 获取 delta 中的 content 和 _thinking
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        thinking = chunk.get("_thinking", "")

        if content:
            still_content += content
            print(f"[Chunk {chunk_count:2d}] content: '{content}'", end="", flush=True)
        else:
            print(f"[Chunk {chunk_count:2d}] (empty)", end="", flush=True)

        if thinking:
            think_content += thinking
            print(f" ⚡thinking: '{thinking[:20]}...' ", end="", flush=True)

        # 实时显示各属性的当前累积状态
        think_preview = client.chat.think[:50] if client.chat.think is not None else "(空)"
        still_preview = client.chat.still[:30] if client.chat.still is not None else "(空)"
        raw_preview = list(client.chat.raw.keys())[:3] if isinstance(client.chat.raw, dict) and client.chat.raw is not None else "(空)"

        print(f" | .think: {think_preview}... | .still: {still_preview}... | .raw: {raw_preview}...")

        # 限制输出数量
        if chunk_count >= 50:
            print("\n... (限制50个chunks，不再继续输出) ...")
            break

    print("\n" + "-" * 70)
    print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
    print("-" * 70)

    print("\n📊 迭代完成后各属性值:")
    print("-" * 70)
    print(f"client.chat.think: {client.chat.think[:200] if client.chat.think is not None else None}...")
    print(f"client.chat.think 长度: {len(client.chat.think) if client.chat.think is not None else 0}")
    print(f"client.chat.still: {client.chat.still}")
    print(f"client.chat.still 长度: {len(client.chat.still) if client.chat.still else 0}")
    print(f"client.chat.tools: {client.chat.tools}")

    print("\n" + "=" * 70)
    print("【thinking测试结束】")
    print("=" * 70)


def test_sync_streaming_with_tools():
    """同步流式 + tools 测试"""
    print("\n" + "=" * 70)
    print("【同步流式 + tools 测试】")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print("⚠️  需要 DEEPSEEK_API_KEY，跳过测试")
        return

    client = CNLLM(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    print("\n📌 发送请求: '北京天气怎么样' (tools=[get_weather])")
    print("-" * 70)

    response = client.chat.create(
        messages=[{"role": "user", "content": "北京天气怎么样"}],
        tools=tools,
        stream=True
    )

    print("\n🔄 开始迭代 (实时输出):")
    print("-" * 70)

    chunk_count = 0
    tool_calls_so_far = []

    for chunk in response:
        chunk_count += 1

        # 获取 delta 中的 content 和 tool_calls
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        tool_calls = delta.get("tool_calls", [])

        if content:
            print(f"[Chunk {chunk_count:2d}] content: '{content}'", end="", flush=True)
        elif tool_calls:
            print(f"[Chunk {chunk_count:2d}] tool_calls: {tool_calls}", end="", flush=True)
        else:
            print(f"[Chunk {chunk_count:2d}] (empty)", end="", flush=True)

        # 实时显示各属性的当前累积状态
        tools_preview = list(client.chat.tools.values())[:100] if isinstance(client.chat.tools, dict) and client.chat.tools is not None else "(空)"
        still_preview = client.chat.still[:30] if client.chat.still is not None else "(空)"

        print(f" | .tools: {tools_preview}... | .still: {still_preview}...")

        # 限制输出数量
        if chunk_count >= 50:
            print("\n... (限制50个chunks，不再继续输出) ...")
            break

    print("\n" + "-" * 70)
    print(f"✅ 迭代完成，共 {chunk_count} 个 chunks")
    print("-" * 70)

    print("\n📊 迭代完成后各属性值:")
    print("-" * 70)
    print(f"client.chat.think: {client.chat.think[:200] if client.chat.think is not None else None}...")
    print(f"client.chat.still: {client.chat.still}")
    print(f"client.chat.still 长度: {len(client.chat.still) if client.chat.still is not None else 0}")
    print(f"client.chat.tools: {client.chat.tools}")
    print(f"client.chat.tools 数量: {len(client.chat.tools) if client.chat.tools is not None else 0}")

    print("\n" + "=" * 70)
    print("【tools测试结束】")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio

    print("\n" + "#" * 70)
    print("# 流式实时消费测试 - 人工审核验证")
    print("#" * 70)

    # 同步测试
    test_sync_streaming_realtime()

    # 异步测试
    asyncio.run(test_async_streaming_realtime())

    # thinking 测试
    test_sync_streaming_with_thinking()

    # tools 测试
    test_sync_streaming_with_tools()

    print("\n" + "#" * 70)
    print("# 所有测试完成")
    print("#" * 70)
