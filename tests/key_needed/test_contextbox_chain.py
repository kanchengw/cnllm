"""
ContextBox 多轮调用链测试

测试覆盖：
  1. 流式单链 — 工具调用 + ContextBox 构建上下文
  2. 非流式单链 — 同上，非流式模式
  3. 混合四链 — 非流式 → 流式 → 非流式 → 流式

需要有效 API Key。
"""
import os
import sys
import json
import unittest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"]
        }
    }
}


@unittest.skipUnless(API_KEY, "need API Key")
class TestContextBoxChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def _simulate_weather(self, tc):
        args = json.loads(tc["function"]["arguments"])
        city = args.get("location", "")
        data = {
            "Beijing": "Beijing: 22C, sunny, humidity 40%",
            "Shanghai": "Shanghai: 28C, cloudy, humidity 70%",
            "Moscow": "Moscow: -5C, snow, humidity 85%",
        }
        result = data.get(city, city + ": 15C")
        print(f"      [execute_tool] {city} -> {result}")
        return result

    # ----------------------------------------------------------------
    # Test 1: 流式单链 — 工具调用 + 上下文构建 + 模型读取上下文
    # ----------------------------------------------------------------
    def test_stream_chain(self):
        from cnllm import ContextBox

        messages = [
            {"role": "user",
             "content": "What is the weather in Beijing and Moscow?"}
        ]

        print("=" * 60)
        print("STREAM CHAIN: Turn 1 - Beijing + Moscow weather")
        print("=" * 60)

        r1 = self.client.chat.create(
            messages=messages, stream=True, tools=[WEATHER_TOOL])
        for _ in r1:
            pass
        print(f"  still='{r1.still[:80] if r1.still else '(empty)'}'")
        print(f"  tools={r1.tools}")

        messages += ContextBox(
            r1.still, r1.think,
            r1.tools if r1.tools else None,
            executor=self._simulate_weather,
        )
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        all_tc = " ".join(m["content"] for m in tool_msgs)
        self.assertIn("Beijing", all_tc)
        self.assertIn("Moscow", all_tc)

        print(f"  ContextBox: {len(tool_msgs)} tool result(s) in context")

        print("\n" + "=" * 60)
        print("STREAM CHAIN: Turn 2 - Ask difference")
        print("(model MUST read tool results to answer)")
        print("=" * 60)

        messages.append({"role": "user",
                         "content": "What is the temperature "
                                    "difference between them?"})
        r2 = self.client.chat.create(
            messages=messages, stream=True)
        for _ in r2:
            pass
        print(f"  still='{r2.still[:120] if r2.still else '(empty)'}'")
        self.assertGreater(len(r2.still) + len(r2.tools), 0)
        if r2.still:
            self.assertIn("Beijing", r2.still)
            self.assertIn("Moscow", r2.still)
        print("  PASS: Stream chain works\n")

    # ----------------------------------------------------------------
    # Test 2: 非流式单链
    # ----------------------------------------------------------------
    def test_nonstream_chain(self):
        from cnllm import ContextBox

        messages = [{"role": "user",
                     "content": "What is the weather in Shanghai?"}]

        print("=" * 60)
        print("NON-STREAM CHAIN: Turn 1 - Shanghai weather")
        print("=" * 60)

        r1 = self.client.chat.create(
            messages=messages, tools=[WEATHER_TOOL])
        print(f"  still='{r1.still[:80] if r1.still else '(empty)'}'")
        print(f"  tools={r1.tools}")

        messages += ContextBox(
            r1.still, r1.think,
            r1.tools if r1.tools else None,
            executor=self._simulate_weather,
        )
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1) if r1.tools else None
        print(f"  ContextBox: {len(tool_msgs)} tool result(s) in context")

        print("\n" + "=" * 60)
        print("NON-STREAM CHAIN: Turn 2 - Follow up")
        print("=" * 60)

        messages.append({"role": "user",
                         "content": "Should I bring an umbrella today?"})
        r2 = self.client.chat.create(
            messages=messages, tools=[WEATHER_TOOL])
        print(f"  still='{r2.still[:120] if r2.still else '(empty)'}'")
        self.assertGreater(len(r2.still) + len(r2.tools), 0)
        print("  PASS: Non-stream chain works\n")

    # ----------------------------------------------------------------
    # Test 3: 混合四链 — 非流式 → 流式 → 非流式 → 流式
    # ----------------------------------------------------------------
    def test_mixed_chain(self):
        from cnllm import ContextBox

        messages = [{"role": "user",
                     "content": "What is the weather in Beijing?"}]

        def turn(n, mode, label, stream, append_q=None):
            nonlocal messages
            print(f"\n{'='*60}")
            print(f"TURN {n} [{mode}]: {label}")
            print(f"{'='*60}")
            if append_q:
                messages.append({"role": "user", "content": append_q})
            print(f"  messages in context: {len(messages)}")

            resp = self.client.chat.create(
                messages=messages, stream=stream, tools=[WEATHER_TOOL])
            if stream:
                for _ in resp:
                    pass
     