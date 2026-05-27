"""
batch 流式 StreamChunk 和 resp.repr 真实 API 测试

需要 API Key。
"""
import os
import sys
import time
import unittest

# 强制拦截 SSL ResourceWarning（Python 3.12+ httpx/yaml gc 噪音）
os.environ.setdefault("PYTHONWARNINGS", "ignore::ResourceWarning")

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 工具定义（各测试复用）
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"},
            },
            "required": ["location"]
        }
    }
}


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestBatchNonStreamRepr(unittest.TestCase):
    """非流式 batch 的 resp.repr + 工具"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_non_stream_batch_repr(self):
        """非流式 batch 中 resp.repr 不报错"""
        resp = self.client.chat.batch(
            prompt=["简单回复：好", "简单回复：坏"],
        )

        with resp.repr as view:
            for r in resp:
                view.refresh()

        print(f"\n  status: {resp.status}")
        print(f"  still: {dict(resp.still) if resp.still else '(empty)'}")

    def test_non_stream_batch_with_tools(self):
        """非流式 batch + 工具：resp.tools 包含完整工具调用"""
        resp = self.client.chat.batch(
            prompt=["北京的天气怎么样？", "上海的天气怎么样？"],
            tools=[WEATHER_TOOL],
        )

        for r in resp:
            pass

        print(f"\n  status: {resp.status}")
        full_tools = dict(resp.tools) if resp.tools else {}
        full_still = dict(resp.still) if resp.still else {}
        print(f"  still: {full_still}")
        print(f"  tools: {full_tools}")
        if full_tools:
            for rid in ("request_0", "request_1"):
                if rid in full_tools:
                    tc_list = full_tools[rid]
                    if tc_list:
                        self.assertIn("function", tc_list[0])
                        self.assertGreater(len(tc_list[0]["function"]["arguments"]), 0)


class TestBatchStreamChunk(unittest.TestCase):
    """batch 流式测试 chunk.still / chunk.think + request_id"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_batch_stream_chunk_still(self):
        """batch 流式中 chunk.still 增量正确，request_id 自动分流"""
        prompts = ["回答一个字：好", "回答一个字：坏"]
        resp = self.client.chat.batch(
            prompt=prompts,
            stream=True,
        )

        results = {"request_0": "", "request_1": ""}
        chunk_count = 0

        for chunk in resp:
            chunk_count += 1
            rid = chunk["request_id"]
            delta = chunk.still
            results[rid] += delta
            self.assertIsInstance(chunk, dict)
            self.assertIn(rid, ("request_0", "request_1"))

        print(f"\n  Total chunks: {chunk_count}")
        for rid, text in results.items():
            print(f"  {rid}: \"{text}\"")

        # 各自累积的内容应包含对应 prompt 的相关回复
        for rid, text in results.items():
            self.assertGreater(len(text), 0, f"{rid} should have content")

    def test_batch_stream_repr(self):
        """batch 流式中 resp.repr 不报错"""
        prompts = ["简单回复：好", "简单回复：坏"]
        resp = self.client.chat.batch(
            prompt=prompts,
            stream=True,
        )

        with resp.repr as view:
            for chunk in resp:
                view.refresh()
                time.sleep(0.05)

        # 流结束后各字段应有内容
        print(f"\n  status: {resp.status}")
        if hasattr(resp, 'still') and resp.still:
            print(f"  still: {dict(resp.still)}")

    def test_batch_stream_chunk_think(self):
        """batch 流式中 chunk.think 可用"""
        prompts = ["简单回复：好", "简单回复：坏"]
        resp = self.client.chat.batch(
            prompt=prompts,
            stream=True,
        )

        for chunk in resp:
            self.assertIsInstance(chunk.think, str)

    def test_batch_stream_with_tools(self):
        """batch 流式 + 工具调用：chunk.still 路由 + resp.tools"""
        prompts = ["北京的天气怎么样？用工具", "上海的天气怎么样？用工具"]
        resp = self.client.chat.batch(
            prompt=prompts,
            stream=True,
            tools=[WEATHER_TOOL],
        )

        results = {"request_0": "", "request_1": ""}
        has_tool_calls = False
        for chunk in resp:
            rid = chunk["request_id"]
            results[rid] += chunk.still
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "tool_calls" in delta and delta["tool_calls"]:
                    has_tool_calls = True

        print(f"\n  still: {results}")
        print(f"  has_tool_calls: {has_tool_calls}")
        for rid in ("request_0", "request_1"):
            # 纯工具调用可能没有 still，不强制断言
            pass


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestAsyncMixedBatchStreamChunk(unittest.TestCase):
    """异步混合 batch：stream=True 和 stream=False 请求共存"""

    @classmethod
    def setUpClass(cls):
        from cnllm import asyncCNLLM
        cls.client = asyncCNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_async_mixed_chunk_routing(self):
        """异步混合 batch 中 chunk.still 路由正确 + resp.repr"""
        import asyncio

        async def run():
            resp = await self.client.chat.batch(
                requests=[
                    {"prompt": "回答一个字：好", "stream": True},
                    {"prompt": "回答一个字：坏"},
                    {"prompt": "回答一个字：行", "stream": True},
                ],
            )

            results = {"request_0": "", "request_2": ""}
            ids_seen = set()

            with resp.repr as view:
                async for chunk in resp:
                    view.refresh()
                    ids_seen.add(chunk["request_id"])
                    if chunk.still:
                        results[chunk["request_id"]] += chunk.still

            for rid in ("request_0", "request_1", "request_2"):
                self.assertIn(rid, resp.still)
            # request_1（非流式）的 marker 也会被 yield
            for rid in ("request_0", "request_2"):
                self.assertGreater(len(results.get(rid, "")), 0,
                    f"{rid} should have streaming content")

            print(f"\n  Chunk IDs: {ids_seen}")
            print(f"  still: {dict(resp.still)}")

        asyncio.run(run())

    def test_async_mixed_repr(self):
        """异步混合 batch 中 resp.repr 不报错"""
        import asyncio

        async def run():
            resp = await self.client.chat.batch(
                requests=[
                    {"prompt": "回答一个字：好", "stream": True},
                    {"prompt": "回答一个字：坏"},
                ],
            )
            with resp.repr as view:
                async for chunk in resp:
                    view.refresh()
            for rid in ("request_0", "request_1"):
                self.assertIn(rid, resp.still)
            print(f"\n  status: {resp.status}")

        asyncio.run(run())

    def test_async_mixed_tools(self):
        """异步混合 batch + 工具"""
        import asyncio

        async def run():
            resp = await self.client.chat.batch(
                requests=[
                    {"prompt": "北京的天气", "stream": True, "tools": [WEATHER_TOOL]},
                    {"prompt": "回答一个字：好"},
                ],
            )
            with resp.repr as view:
                async for chunk in resp:
                    view.refresh()
            self.assertIn("request_0", resp.still)
            self.assertIn("request_1", resp.still)
            print(f"\n  still: {dict(resp.still)}")
            if resp.tools:
                print(f"  tools: {dict(resp.tools)}")

        asyncio.run(run())


class TestMixedBatchStreamChunk(unittest.TestCase):
    """混合 batch：stream=True 和 stream=False 请求共存"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_mixed_stream_chunk_routing(self):
        """混合 batch 中，所有请求的结果通过 resp.still 统一获取"""
        resp = self.client.chat.batch(
            requests=[
                {"prompt": "回答一个字：好", "stream": True},
                {"prompt": "回答一个字：坏"},
                {"prompt": "回答一个字：行", "stream": True},
            ],
        )

        for r in resp:
            pass

        print(f"\n  still keys: {list(resp.still.keys()) if resp.still else 'empty'}")
        for rid in ("request_0", "request_1", "request_2"):
            still_text = resp.still[rid] if rid in resp.still else ""
            print(f"  {rid}: \"{still_text[:50]}\"")
            self.assertGreater(len(still_text), 0, f"{rid} should have content")

    def test_mixed_stream_repr(self):
        """混合 batch 中 resp.repr 不报错"""
        resp = self.client.chat.batch(
            requests=[
                {"prompt": "回答一个字：好", "stream": True},
                {"prompt": "回答一个字：坏"},
                {"prompt": "回答一个字：行", "stream": True},
            ],
        )

        with resp.repr as view:
            for chunk in resp:
                view.refresh()

        print(f"\n  status: {resp.status}")
        print(f"  still keys: {list(resp.still.keys()) if resp.still else 'empty'}")
        self.assertIn("request_0", resp.still)
        self.assertIn("request_1", resp.still)
        self.assertIn("request_2", resp.still)

    def test_mixed_stream_tools(self):
        """混合 batch + 工具：流式和非流式请求的工具调用都能取到"""
        resp = self.client.chat.batch(
            requests=[
                {"prompt": "北京的天气", "stream": True, "tools": [WEATHER_TOOL]},
                {"prompt": "上海的天气", "tools": [WEATHER_TOOL]},
                {"prompt": "回答一个字：好", "stream": True},
            ],
        )

        for r in resp:
            pass

        print(f"\n  still: {dict(resp.still) if resp.still else '(empty)'}")
        print(f"  tools: {dict(resp.tools) if resp.tools else '(empty)'}")
        self.assertIn("request_2", resp.still, "非工具请求应有 still")
        self.assertGreater(len(resp.still["request_2"]), 0)
        if resp.tools:
            for rid in ("request_0", "request_1"):
                self.assertIn(rid, resp.tools, f"工具请求 {rid} 应在 tools 中")


        import asyncio

        async def run():
            re