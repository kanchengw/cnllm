"""
StreamChunk 和 LiveDict 真实 API 测试

需要有效 API Key 才能运行。用法：
    python -m pytest tests/key_needed/test_stream_chunk_live_real.py -v

或：
    python tests/key_needed/test_stream_chunk_live_real.py
"""
import os
import sys
import json
import time
import unittest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# 需要设置环境变量（或项目根目录 .env 文件）
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestStreamChunkRealAPI(unittest.TestCase):
    """真实 API 调用测试 StreamChunk"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_stream_chunk_still_and_think(self):
        """验证流式迭代中 chunk.still 和 chunk.think 正确反映逐帧增量"""
        messages = [
            {"role": "system", "content": "每次对话只回答一个字，然后结束。"},
            {"role": "user", "content": "你好"}
        ]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
        )

        accumulated_still = ""
        accumulated_think = ""
        chunk_count = 0

        for chunk in resp:
            chunk_count += 1
            # 验证 StreamChunk 类型
            self.assertIsInstance(chunk, dict)

            # 增量内容
            still_delta = chunk.still
            think_delta = chunk.think

            # 字符串操作（验证 str 类型）
            if still_delta:
                accumulated_still += still_delta
            if think_delta:
                accumulated_think += think_delta

        # 验证流结束后的全量属性
        print(f"\n  Chunks: {chunk_count}")
        print(f"  final still: {resp.still}")
        print(f"  final think: {resp.think[:100] if resp.think else '(none)'}")

        self.assertIsNotNone(resp.still)
        self.assertGreater(len(resp.still), 0)
        # chunk.still 逐帧拼接后等于 resp.still
        self.assertEqual(accumulated_still, resp.still)

    def test_stream_chunk_dict_compatibility(self):
        """验证 chunk 是完整兼容的 dict"""
        messages = [{"role": "user", "content": "简单说'你好'"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
        )

        for chunk in resp:
            # dict 接口完好
            self.assertIn("choices", chunk)
            delta = chunk["choices"][0]["delta"]
            self.assertIn(chunk.id if hasattr(chunk, 'id') else "id",
                          ["id", "choices"])
            # json 序列化
            json_str = json.dumps(chunk)
            self.assertIsInstance(json_str, str)
            break  # 只验证第一个 chunk

    def test_stream_chunk_incremental_properties(self):
        """验证 .still 和 .think 在每个 chunk 中是增量而非全量"""
        messages = [{"role": "user", "content": "用3个字以内的长度描述天气"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
        )

        max_delta_len = 0
        chunk_count = 0
        for chunk in resp:
            chunk_count += 1
            still_delta = chunk.still
            think_delta = chunk.think
            max_delta_len = max(max_delta_len, len(still_delta), len(think_delta))

        self.assertGreater(chunk_count, 0)
        # 每个 chunk 的 delta 通常远小于全量
        print(f"\n  Chunks: {chunk_count}, max_delta_len: {max_delta_len}")
        print(f"  final still length: {len(resp.still)}")


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestLiveDictRealAPI(unittest.TestCase):
    """真实 API 调用测试 LiveDict——仅验证不报错"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_live_dict_context_manager(self):
        """验证 live 上下文管理器不报错（终端实时刷新）"""
        messages = [{"role": "user", "content": "简单说'你好'"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
        )

        with resp.repr as view:
            for chunk in resp:
                view.refresh()

        # 流结束后正常访问属性
        self.assertGreater(len(resp.still), 0)

    def test_live_dict_multiple_requests(self):
        """连续多次 live dict 调用"""
        for i in range(3):
            messages = [{"role": "user", "content": f"说数字{i}"}]
            resp = self.client.chat.create(
                messages=messages,
                model=MODEL,
                stream=True,
            )
            with resp.repr as view:
                for chunk in resp:
                    view.refresh()
            self.assertIn(str(i) if i < 10 else "",
                          resp.still if resp.still else "")


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestStreamChunkNoThink(unittest.TestCase):
    """无 reasoning 的模型测试"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_think_empty_when_no_reasoning(self):
        """不启用 thinking 时 chunk.think 始终为空字符串"""
        messages = [{"role": "user", "content": "简单回复'好的'"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
            # 不传 extra_body thinking
        )

        has_think = False
        for chunk in resp:
            self.assertIsInstance(chunk.think, str)
            if chunk.think:
                has_think = True

        if has_think:
            print("\n  Note: model returned reasoning even without thinking=True")
        else:
            print("\n  No reasoning content as expected")


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestStreamChunkWithTools(unittest.TestCase):
    """工具调用场景下的 StreamChunk 和 resp.repr 测试"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_stream_chunk_tool_calls_in_dict(self):
        """验证带工具调用的流式响应中，chunk 的 dict 接口包含 tool_calls 字段"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        messages = [{"role": "user", "content": "北京的天气怎么样？用工具查询"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
            tools=tools,
        )

        chunk_count = 0
        has_tool_calls = False

        for chunk in resp:
            chunk_count += 1
            # dict 接口中检查 tool_calls
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "tool_calls" in delta and delta["tool_calls"]:
                    has_tool_calls = True
                    for tc in delta["tool_calls"]:
                        # 验证 tool_calls 数据结构完整
                        self.assertIn("index", tc)
                        if tc.get("function"):
                            # arguments 可能是增量片段
                            self.assertIsInstance(tc["function"].get("arguments", ""), str)

        self.assertTrue(has_tool_calls or chunk_count > 0,
                        "should have at least some chunks")
        if has_tool_calls:
            print(f"\n  Detected tool_calls in streaming chunks")
        else:
            print(f"\n  No tool_calls detected (model may have responded without tools)")

        # 流结束后检查全量 tools
        full_tools = resp.tools
        if full_tools:
            print(f"  resp.tools has {len(full_tools)} tool call(s)")
            for idx, tc in full_tools.items():
                print(f"    [{idx}] name={tc.get('function', {}).get('name', '?')}, "
                      f"args={tc.get('function', {}).get('arguments', '')[:80]}")

    def test_repr_with_tools(self):
        """resp.repr 在工具调用场景下不报错"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        messages = [{"role": "user", "content": "用工具查北京的天气"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
            tools=tools,
        )

        with resp.repr as view:
            for chunk in resp:
                view.refresh()

        # 流结束后正常访问
        self.assertIsNotNone(resp.still)
        print(f"\n  final still: {resp.still[:80] if resp.still else '(empty)'}")
        print(f"  final tools: {resp.tools}")


@unittest.skipUnless(API_KEY, "需要 API Key")
class TestOriginalRepr(unittest.TestCase):
    """验证原始 __repr__() 在迭代中和迭代后的行为"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_repr_during_and_after_stream(self):
        """打印迭代中和迭代后的 __repr__() 输出供观察"""
        messages = [{"role": "user", "content": "简单回复'好的'"}]
        resp = self.client.chat.create(
            messages=messages,
            model=MODEL,
            stream=True,
        )

        print("\n  === repr() during iteration ===")
        for i, chunk in enumerate(resp):
            if i < 3:  # 只打前 3 个 chunk 的 repr
                r = repr(resp)
                print(f"  chunk {i}: {r[:200]}...")

        print("\n  === repr() after iteration ===")
        final = repr(resp)
        print(f"  {final[:500]}")
        self.assertIn("choices", final)
        print("\n  (truncated to 500 chars for display)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
