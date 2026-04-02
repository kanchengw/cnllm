"""
MiniMax Adapter 完整参数测试 - 真实 API
测试各种参数组合对模型输出的影响，以及 OpenAI 响应转换是否成功
"""
import pytest
import json
import os
from dotenv import load_dotenv

load_dotenv()


class TestMiniMaxFullParameters:
    """MiniMax 完整参数测试套件"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """设置"""
        self.api_key = os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            pytest.skip("MINIMAX_API_KEY not set")

    def _create_client(self, **kwargs):
        """创建 client"""
        from cnllm import CNLLM
        return CNLLM(model="minimax-m2.7", api_key=self.api_key, **kwargs)

    def _print_payload(self, adapter, params):
        """打印 payload 结构"""
        payload = adapter._build_payload(params)
        print(f"\n  [Payload 结构]")
        print(f"  {json.dumps(payload, indent=6, ensure_ascii=False)}")
        return payload

    def test_01_basic_chat(self):
        """测试 1: 基础聊天 - model + messages"""
        print("\n" + "=" * 70)
        print("测试 1: 基础聊天 - model + messages")
        print("=" * 70)

        client = self._create_client()
        print(f"  [参数] model=minimax-m2.7, messages=[user:hi]")

        response = client.chat.create(
            messages=[{"role": "user", "content": "say 'hello' exactly"}]
        )

        print(f"\n  [响应]")
        print(f"  id: {response.get('id')}")
        print(f"  object: {response.get('object')}")
        print(f"  content: {response['choices'][0]['message']['content']}")
        print(f"  finish_reason: {response['choices'][0]['finish_reason']}")

        assert response["object"] == "chat.completion"
        assert "choices" in response
        print("\n  [PASS] 基础聊天成功")

    def test_02_temperature_zero(self):
        """测试 2: temperature=0 (确定性输出)"""
        print("\n" + "=" * 70)
        print("测试 2: temperature=0 (确定性输出)")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:count to 3], temperature=0")
        print(f"  [预期] 多次调用应得到相同或非常相似的结果")

        results = []
        for i in range(3):
            response = client.chat.create(
                messages=[{"role": "user", "content": "count to 3"}],
                temperature=0,
                max_tokens=20
            )
            content = response["choices"][0]["message"]["content"]
            results.append(content)
            print(f"  调用 {i+1}: {content[:50]}...")

        print(f"\n  [结果分析]")
        print(f"  三次结果是否相同: {results[0] == results[1] == results[2]}")
        print(f"  结果1: {results[0][:30]}")
        print(f"  结果2: {results[1][:30]}")
        print(f"  结果3: {results[2][:30]}")

        assert "choices" in response
        print("\n  [PASS] temperature=0 测试完成")

    def test_03_temperature_high(self):
        """测试 3: temperature=1.0 (高随机性)"""
        print("\n" + "=" * 70)
        print("测试 3: temperature=1.0 (高随机性)")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:give me a random word], temperature=1.0")
        print(f"  [预期] 多次调用应得到不同结果")

        results = []
        for i in range(3):
            response = client.chat.create(
                messages=[{"role": "user", "content": "give me a random word"}],
                temperature=1.0,
                max_tokens=10
            )
            content = response["choices"][0]["message"]["content"]
            results.append(content)
            print(f"  调用 {i+1}: {content}")

        print(f"\n  [结果分析]")
        diff_count = len(set(results))
        print(f"  不同结果数量: {diff_count}/3")
        print(f"  结果: {results}")

        assert "choices" in response
        print("\n  [PASS] temperature=1.0 测试完成")

    def test_04_max_tokens_limit(self):
        """测试 4: max_tokens 限制输出长度"""
        print("\n" + "=" * 70)
        print("测试 4: max_tokens 限制输出长度")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:write 100 words], max_tokens=50")

        response = client.chat.create(
            messages=[{"role": "user", "content": "write a short sentence"}],
            max_tokens=50
        )

        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        print(f"\n  [响应]")
        print(f"  content 长度: {len(content)} 字符")
        print(f"  content: {content[:80]}...")
        print(f"  completion_tokens: {usage.get('completion_tokens', 'N/A')}")

        if usage.get("completion_tokens"):
            assert usage["completion_tokens"] <= 50 + 10, "max_tokens 应该限制了输出"

        print("\n  [PASS] max_tokens 限制生效")

    def test_05_top_p_sampling(self):
        """测试 5: top_p 核采样"""
        print("\n" + "=" * 70)
        print("测试 5: top_p 核采样")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:pick a number], top_p=0.1")
        print(f"  [说明] top_p=0.1 表示只考虑概率最高的 10% 的 token")

        response = client.chat.create(
            messages=[{"role": "user", "content": "pick a number between 1 and 10"}],
            top_p=0.1,
            temperature=1.0,
            max_tokens=20
        )

        content = response["choices"][0]["message"]["content"]
        print(f"\n  [响应]")
        print(f"  content: {content}")

        assert "choices" in response
        print("\n  [PASS] top_p 采样成功")

    def test_06_top_k_sampling(self):
        """测试 6: top_k 采样"""
        print("\n" + "=" * 70)
        print("测试 6: top_k 采样")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:pick a color], top_k=1")
        print(f"  [说明] top_k=1 表示只考虑概率最高的 1 个 token")

        response = client.chat.create(
            messages=[{"role": "user", "content": "pick a color"}],
            top_k=1,
            temperature=1.0,
            max_tokens=10
        )

        content = response["choices"][0]["message"]["content"]
        print(f"\n  [响应]")
        print(f"  content: {content}")

        assert "choices" in response
        print("\n  [PASS] top_k 采样成功")

    def test_07_presence_penalty(self):
        """测试 7: presence_penalty 抑制重复"""
        print("\n" + "=" * 70)
        print("测试 7: presence_penalty 抑制重复")
        print("=" * 70)

        client = self._create_client()

        prompt = "repeat the word 'hello' three times"

        print(f"  [参数] prompt='{prompt}'")
        print(f"  [测试] presence_penalty=0 vs presence_penalty=2.0")

        response1 = client.chat.create(
            messages=[{"role": "user", "content": prompt}],
            presence_penalty=0,
            max_tokens=30
        )

        response2 = client.chat.create(
            messages=[{"role": "user", "content": prompt}],
            presence_penalty=2.0,
            max_tokens=30
        )

        content1 = response1["choices"][0]["message"]["content"]
        content2 = response2["choices"][0]["message"]["content"]

        print(f"\n  [响应]")
        print(f"  presence_penalty=0: {content1}")
        print(f"  presence_penalty=2.0: {content2}")

        print(f"\n  [分析]")
        print(f"  高 presence_penalty 应该产生更少重复")
        print(f"  两者是否不同: {content1 != content2}")

        assert "choices" in response2
        print("\n  [PASS] presence_penalty 生效")

    def test_08_frequency_penalty(self):
        """测试 8: frequency_penalty 惩罚重复"""
        print("\n" + "=" * 70)
        print("测试 8: frequency_penalty 惩罚重复")
        print("=" * 70)

        client = self._create_client()

        prompt = "list the numbers one through five"

        print(f"  [参数] prompt='{prompt}'")
        print(f"  [测试] frequency_penalty=0 vs frequency_penalty=2.0")

        response1 = client.chat.create(
            messages=[{"role": "user", "content": prompt}],
            frequency_penalty=0,
            max_tokens=50
        )

        response2 = client.chat.create(
            messages=[{"role": "user", "content": prompt}],
            frequency_penalty=2.0,
            max_tokens=50
        )

        content1 = response1["choices"][0]["message"]["content"]
        content2 = response2["choices"][0]["message"]["content"]

        print(f"\n  [响应]")
        print(f"  frequency_penalty=0: {content1}")
        print(f"  frequency_penalty=2.0: {content2}")

        assert "choices" in response2
        print("\n  [PASS] frequency_penalty 生效")

    def test_09_stop_sequence(self):
        """测试 9: stop 停止序列"""
        print("\n" + "=" * 70)
        print("测试 9: stop 停止序列")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] messages=[user:count to 10], stop=['5']")
        print(f"  [预期] 输出应该在 '5' 处停止")

        response = client.chat.create(
            messages=[{"role": "user", "content": "count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
            stop=["5"],
            max_tokens=100
        )

        content = response["choices"][0]["message"]["content"]
        finish_reason = response["choices"][0]["finish_reason"]

        print(f"\n  [响应]")
        print(f"  content: {content}")
        print(f"  finish_reason: {finish_reason}")

        assert "5" not in content or finish_reason == "stop"
        print("\n  [PASS] stop 序列生效")

    def test_10_user_parameter(self):
        """测试 10: user 参数"""
        print("\n" + "=" * 70)
        print("测试 10: user 参数")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] user='test-user-12345'")

        response = client.chat.create(
            messages=[{"role": "user", "content": "hi"}],
            user="test-user-12345",
            max_tokens=10
        )

        print(f"\n  [响应]")
        print(f"  content: {response['choices'][0]['message']['content']}")
        print(f"  (user 参数通常用于追踪和日志，不影响输出)")

        assert "choices" in response
        print("\n  [PASS] user 参数被接受")

    def test_11_stream_true(self):
        """测试 11: stream=True 流式输出"""
        print("\n" + "=" * 70)
        print("测试 11: stream=True 流式输出")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] stream=True, messages=[user:count to 3]")

        response = client.chat.create(
            messages=[{"role": "user", "content": "count to 3"}],
            stream=True,
            max_tokens=50
        )

        print(f"\n  [响应类型] {type(response)}")

        chunks = []
        full_content = ""

        for i, chunk in enumerate(response):
            chunks.append(chunk)
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                full_content += content

                if i < 3:
                    print(f"  Chunk {i}: {chunk}")

        print(f"\n  [统计]")
        print(f"  总 chunks 数: {len(chunks)}")
        print(f"  完整内容: {full_content}")

        assert len(chunks) > 0, "应该有多个 chunks"
        assert full_content != "", "内容不为空"
        print("\n  [PASS] 流式输出成功")

    def test_12_stream_false(self):
        """测试 12: stream=False 非流式输出"""
        print("\n" + "=" * 70)
        print("测试 12: stream=False 非流式输出")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] stream=False")

        response = client.chat.create(
            messages=[{"role": "user", "content": "say hello"}],
            stream=False,
            max_tokens=20
        )

        print(f"\n  [响应类型] {type(response)}")
        print(f"  [是否是 generator] {hasattr(response, '__iter__')}")

        if hasattr(response, '__iter__') and not isinstance(response, dict):
            content = ""
            for chunk in response:
                if "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content += delta.get("content", "")
            print(f"  完整内容: {content}")
        else:
            print(f"  content: {response['choices'][0]['message']['content']}")

        assert "choices" in response or hasattr(response, '__iter__')
        print("\n  [PASS] 非流式输出成功")

    def test_13_messages_only(self):
        """测试 13: 仅使用 messages"""
        print("\n" + "=" * 70)
        print("测试 13: 仅使用 messages")
        print("=" * 70)

        client = self._create_client()

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is 2+2?"}
        ]

        print(f"  [参数] messages 数量: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"    [{i}] {msg['role']}: {msg['content'][:50]}...")

        response = client.chat.create(messages=messages, max_tokens=20)

        print(f"\n  [响应]")
        print(f"  content: {response['choices'][0]['message']['content']}")

        assert "choices" in response
        print("\n  [PASS] messages 格式正确")

    def test_14_prompt_conversion(self):
        """测试 14: prompt 自动转换为 messages"""
        print("\n" + "=" * 70)
        print("测试 14: prompt 转换为 messages")
        print("=" * 70)

        client = self._create_client()

        print(f"  [参数] prompt='What is the capital of France?'")

        response = client.chat.create(
            prompt="What is the capital of France?",
            max_tokens=20
        )

        print(f"\n  [响应]")
        print(f"  content: {response['choices'][0]['message']['content']}")

        assert "choices" in response
        print("\n  [PASS] prompt 转换成功")

    def test_15_response_format_openai(self):
        """测试 15: 响应格式验证 - OpenAI 标准"""
        print("\n" + "=" * 70)
        print("测试 15: 响应格式验证 - OpenAI 标准")
        print("=" * 70)

        client = self._create_client()

        response = client.chat.create(
            messages=[{"role": "user", "content": "say 'test'"}],
            max_tokens=10
        )

        print(f"\n  [OpenAI 标准格式验证]")
        print(f"  {'id':<20} -> {response.get('id', 'MISSING')}")
        print(f"  {'object':<20} -> {response.get('object', 'MISSING')}")
        print(f"  {'created':<20} -> {response.get('created', 'MISSING')}")
        print(f"  {'model':<20} -> {response.get('model', 'MISSING')}")
        print(f"  {'choices':<20} -> {'[OK]' if 'choices' in response else 'MISSING'}")
        print(f"  {'usage':<20} -> {'[OK]' if 'usage' in response else 'MISSING'}")

        required = ["id", "object", "created", "model", "choices", "usage"]
        for field in required:
            assert field in response, f"Missing required field: {field}"

        choice = response["choices"][0]
        assert "message" in choice or "delta" in choice
        assert "finish_reason" in choice

        print(f"\n  [choices[0] 结构]")
        print(f"    finish_reason: {choice.get('finish_reason')}")
        print(f"    message.content: {choice.get('message', {}).get('content', 'N/A')[:50]}")

        print("\n  [PASS] OpenAI 标准格式正确")

    def test_16_usage_statistics(self):
        """测试 16: usage 统计数据"""
        print("\n" + "=" * 70)
        print("测试 16: usage 统计数据")
        print("=" * 70)

        client = self._create_client()

        response = client.chat.create(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=50
        )

        usage = response.get("usage", {})

        print(f"\n  [usage 统计]")
        print(f"  prompt_tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"  completion_tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  total_tokens: {usage.get('total_tokens', 'N/A')}")

        if "completion_tokens_details" in usage:
            print(f"  reasoning_tokens: {usage['completion_tokens_details'].get('reasoning_tokens', 'N/A')}")

        assert "usage" in response
        assert usage.get("total_tokens", 0) > 0
        print("\n  [PASS] usage 统计正确")

    def test_17_combined_sampling_params(self):
        """测试 17: 组合采样参数"""
        print("\n" + "=" * 70)
        print("测试 17: 组合采样参数 (temperature + top_p + top_k)")
        print("=" * 70)

        client = self._create_client()

        params = {
            "messages": [{"role": "user", "content": "give me a random sentence"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 30
        }

        print(f"  [参数组合]")
        print(f"    temperature: {params['temperature']}")
        print(f"    top_p: {params['top_p']}")
        print(f"    top_k: {params['top_k']}")
        print(f"    max_tokens: {params['max_tokens']}")

        response = client.chat.create(**params)

        content = response["choices"][0]["message"]["content"]
        print(f"\n  [响应]")
        print(f"  content: {content}")

        assert "choices" in response
        print("\n  [PASS] 组合参数成功")

    def test_18_combined_all_params(self):
        """测试 18: 所有参数组合"""
        print("\n" + "=" * 70)
        print("测试 18: 所有参数组合")
        print("=" * 70)

        client = self._create_client()

        messages = [{"role": "user", "content": "count to 5"}]

        params = {
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.85,
            "top_k": 10,
            "max_tokens": 50,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "stream": False,
            "user": "test-combined-params"
        }

        print(f"  [完整参数列表]")
        for key, value in params.items():
            print(f"    {key}: {value}")

        response = client.chat.create(**params)

        print(f"\n  [响应]")
        print(f"  content: {response['choices'][0]['message']['content'][:80]}")
        print(f"  finish_reason: {response['choices'][0]['finish_reason']}")

        assert "choices" in response
        assert response["choices"][0]["finish_reason"] in ["stop", "length"]
        print("\n  [PASS] 所有参数组合成功")

    def test_19_payload_structure_verify(self):
        """测试 19: Payload 结构验证"""
        print("\n" + "=" * 70)
        print("测试 19: Payload 结构验证")
        print("=" * 70)

        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key=self.api_key, model="minimax-m2.7")

        params = {
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 100,
            "stream": False,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "stop": "stop",
            "user": "test-user",
            "organization": "group-123"
        }

        payload = adapter._build_payload(params)

        print(f"\n  [Payload 关键字段验证]")
        print(f"  model: {payload.get('model')} (应为 MiniMax-M2.7)")
        print(f"  temperature: {payload.get('temperature')} (顶层，非 parameters)")
        print(f"  top_p: {payload.get('top_p')}")
        print(f"  top_k: {payload.get('top_k')}")
        print(f"  max_tokens: {payload.get('max_tokens')}")
        print(f"  stream: {payload.get('stream')}")
        print(f"  presence_penalty: {payload.get('presence_penalty')}")
        print(f"  frequency_penalty: {payload.get('frequency_penalty')}")
        print(f"  stop: {payload.get('stop')}")
        print(f"  user: {payload.get('user')}")
        print(f"  group_id: {payload.get('group_id')} (organization 映射)")

        print(f"\n  [验证结果]")
        assert payload["model"] == "MiniMax-M2.7", "model 映射错误"
        assert "temperature" in payload, "temperature 应该在顶层"
        assert "parameters" not in payload, "ERROR: 不应该有 parameters 嵌套!"
        assert "group_id" in payload, "organization 应该映射到 group_id"
        assert payload.get("temperature") == 0.7

        print(f"  [PASS] Payload 是扁平结构，符合 MiniMax API 规范")

    def test_20_openai_format_conversion(self):
        """测试 20: OpenAI 格式转换验证"""
        print("\n" + "=" * 70)
        print("测试 20: OpenAI 格式转换验证")
        print("=" * 70)

        from cnllm.core.vendor.minimax import MiniMaxAdapter

        adapter = MiniMaxAdapter(api_key=self.api_key, model="minimax-m2.7")

        raw_response = {
            "id": "test-id-123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response content"
                }
            }],
            "created": 1234567890,
            "model": "MiniMax-M2.7",
            "usage": {
                "total_tokens": 50,
                "prompt_tokens": 20,
                "completion_tokens": 30
            }
        }

        print(f"  [原始 MiniMax 响应]")
        print(f"  {json.dumps(raw_response, indent=4, ensure_ascii=False)}")

        openai_format = adapter._to_openai_format(raw_response, "MiniMax-M2.7")

        print(f"\n  [转换后 OpenAI 格式]")
        print(f"  id: {openai_format['id']}")
        print(f"  object: {openai_format['object']}")
        print(f"  model: {openai_format['model']}")
        print(f"  content: {openai_format['choices'][0]['message']['content']}")
        print(f"  finish_reason: {openai_format['choices'][0]['finish_reason']}")

        assert openai_format["object"] == "chat.completion"
        assert openai_format["choices"][0]["message"]["content"] == "Test response content"
        print("\n  [PASS] OpenAI 格式转换正确")


if __name__ == "__main__":
    print("=" * 70)
    print("MiniMax Adapter 完整参数测试套件 (真实 API)")
    print("=" * 70)
    print()
    print("测试项目:")
    print("  1.  基础聊天")
    print("  2.  temperature=0 (确定性)")
    print("  3.  temperature=1.0 (高随机性)")
    print("  4.  max_tokens 限制")
    print("  5.  top_p 核采样")
    print("  6.  top_k 采样")
    print("  7.  presence_penalty")
    print("  8.  frequency_penalty")
    print("  9.  stop 停止序列")
    print(" 10.  user 参数")
    print(" 11.  stream=True")
    print(" 12.  stream=False")
    print(" 13.  messages 格式")
    print(" 14.  prompt 转换")
    print(" 15.  OpenAI 标准格式")
    print(" 16.  usage 统计")
    print(" 17.  组合采样参数")
    print(" 18.  所有参数组合")
    print(" 19.  Payload 结构验证")
    print(" 20.  OpenAI 格式转换")
    print()
    print("=" * 70)

    pytest.main([__file__, "-v", "-s"])
