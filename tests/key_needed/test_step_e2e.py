"""
Step (阶跃星辰) E2E 测试

需要环境变量 STEP_API_KEY
"""
import os, sys, time, unittest
from dotenv import load_dotenv

load_dotenv()
STEP_API_KEY = os.environ.get("STEP_API_KEY")

requires_step_key = unittest.skipUnless(STEP_API_KEY, "需要 STEP_API_KEY")


@requires_step_key
class TestStepChatCreate(unittest.TestCase):
    """Step 单请求 create 测试"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=STEP_API_KEY, model="step-3-5-flash")

    def test_non_streaming(self):
        resp = self.client.chat.create(
            messages=[{"role": "user", "content": "1+1等于几？"}],
        )
        self.assertIsNotNone(resp)
        choices = resp.get("choices", [])
        self.assertGreater(len(choices), 0)
        content = choices[0].get("message", {}).get("content", "")
        self.assertIn("2", content)
        print(f"\n[Step] 非流式响应: {content[:50]}...")

    def test_streaming(self):
        chunks = []
        resp = self.client.chat.create(
            messages=[{"role": "user", "content": "用一句话介绍北京"}],
            stream=True,
        )
        for chunk in resp:
            chunks.append(chunk)
        self.assertGreater(len(chunks), 0)
        full = "".join(
            c.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for c in chunks
        )
        self.assertGreater(len(full), 0)
        print(f"\n[Step] 流式响应: {full[:50]}...")

    def test_with_reasoning(self):
        """测试 reasoning_effort 参数（step-3.7-flash 支持）"""
        from cnllm import CNLLM
        client = CNLLM(api_key=STEP_API_KEY, model="step-3-7-flash")
        resp = client.chat.create(
            messages=[{"role": "user", "content": "请一步一步思考：17×23=？"}],
            reasoning_effort="medium",
        )
        choices = resp.get("choices", [])
        self.assertGreater(len(choices), 0)
        msg = choices[0].get("message", {})
        self.assertIn("content", msg)
        print(f"\n[Step+推理] content: {msg.get('content', '')[:80]}...")

    def test_tools(self):
        """tool_call 功能"""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"]
                }
            }
        }]
        resp = self.client.chat.create(
            messages=[{"role": "user", "content": "北京的天气怎么样？"}],
            tools=tools,
            tool_choice="auto",
        )
        choices = resp.get("choices", [])
        self.assertGreater(len(choices), 0)
        print(f"\n[Step+Tools] response: {str(choices[0].get('message', {}))[:200]}...")


@requires_step_key
class TestStepChatBatch(unittest.TestCase):
    """Step 批量请求测试"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=STEP_API_KEY, model="step-3-5-flash")

    def test_non_streaming_batch(self):
        resp = self.client.chat.batch(
            requests=[
                {"prompt": "你好"},
                {"prompt": "1+1等于几？"},
            ],
        )
        for _ in resp:
            pass
        s = resp.status
        self.assertGreater(s.get("success_count", 0), 0)
        print(f"\n[Step Batch] status: {s}")

    def test_streaming_batch(self):
        acc = self.client.chat.batch(
            requests=[
                {"prompt": "用一句话介绍北京"},
                {"prompt": "用一句话介绍上海"},
            ],
            stream=True,
        )
        chunks = []
        for chunk in acc:
            chunks.append(chunk)
        self.assertGreater(len(chunks), 0)
        print(f"\n[Step Stream Batch] chunks: {len(chunks)}")


@requires_step_key
class TestStepAsync(unittest.TestCase):
    """Step 异步测试"""

    @classmethod
    def setUpClass(cls):
        from cnllm.entry.async_client import asyncCNLLM
        cls.client = asyncCNLLM(api_key=STEP_API_KEY, model="step-3-5-flash")

    def test_async_non_streaming(self):
        import asyncio
        async def run():
            resp = await self.client.chat.create(
                messages=[{"role": "user", "content": "1+1等于几？"}],
            )
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.assertIn("2", content)
            print(f"\n[Step Async] {content[:50]}...")
        asyncio.run(run())

    def test_async_streaming_batch(self):
        import asyncio
        async def run():
            acc = await self.client.chat.batch(
                requests=[
                    {"prompt": "用一句话介绍北京"},
                    {"prompt": "用一句话介绍上海"},
                ],
                stream=True,
            )
            chunks = []
            async for chunk in acc:
                chunks.append(chunk)
            print(f"\n[Step Async Stream] chunks: {len(chunks)}")
            self.assertGreater(len(chunks), 0)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
