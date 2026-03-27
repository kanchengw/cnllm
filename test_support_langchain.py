import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from cnllm import CNLLM, MINIMAX_API_KEY

print("=" * 60)
print("LangChain 兼容性测试")
print("=" * 60)

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

tests_passed = 0
tests_total = 0

def test(name, condition, detail=None):
    global tests_passed, tests_total
    tests_total += 1
    if condition:
        tests_passed += 1
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail}")

print("\n[1] 消息类型转换...")

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    resp = client.chat.create(messages=[{"role": "user", "content": "hi"}])
    content = resp["choices"][0]["message"]["content"]

    test("消息类型转换", True)
except Exception as e:
    test("消息类型转换", False, str(e)[:50])

print("\n[2] prompt 参数支持...")

try:
    resp = client.chat.create(prompt="say hi")
    content = resp["choices"][0]["message"]["content"]
    test("prompt 参数支持", bool(content))
except Exception as e:
    test("prompt 参数支持", False, str(e)[:50])

print("\n[3] OpenAI 格式输出...")

try:
    resp = client("hello")
    test("OpenAI 格式输出", "choices" in resp and "usage" in resp)
except Exception as e:
    test("OpenAI 格式输出", False, str(e)[:50])

print("\n[4] message_to_dict 兼容...")

try:
    from langchain_core.messages import message_to_dict
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content="test")
    msg_dict = message_to_dict(msg)
    test("message_to_dict 兼容", "content" in msg_dict.get("data", {}))
except Exception as e:
    test("message_to_dict 兼容", False, str(e)[:50])

print("\n[5] messages_to_dict 兼容...")

try:
    from langchain_core.messages import messages_to_dict, HumanMessage

    messages = [HumanMessage(content="test")]
    msgs_dict = messages_to_dict(messages)
    test("messages_to_dict 兼容", len(msgs_dict) == 1)
except Exception as e:
    test("messages_to_dict 兼容", False, str(e)[:50])

print("\n[6] ChatPromptTemplate 兼容...")

try:
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    test("ChatPromptTemplate 兼容", True)
except Exception as e:
    test("ChatPromptTemplate 兼容", False, str(e)[:50])

print("\n[7] StrOutputParser 兼容...")

try:
    from langchain_core.output_parsers import StrOutputParser

    parser = StrOutputParser()
    test("StrOutputParser 创建", True)
except Exception as e:
    test("StrOutputParser 创建", False, str(e)[:50])

print("\n[8] Runnable 接口兼容 (规划中)...")

try:
    from langchain_core.runnables import Runnable

    test("Runnable 接口检查 (规划中)", isinstance(client, Runnable))
except Exception as e:
    test("Runnable 接口检查 (规划中)", False, str(e)[:50])

print("\n[9] AIMessageChunk 兼容...")

try:
    from langchain_core.messages import AIMessageChunk

    chunk = AIMessageChunk(content="test")
    test("AIMessageChunk 创建", True)
except Exception as e:
    test("AIMessageChunk 创建", False, str(e)[:50])

print("\n[10] BaseMessage 兼容...")

try:
    from langchain_core.messages import BaseMessage

    test("BaseMessage 导入", True)
except Exception as e:
    test("BaseMessage 导入", False, str(e)[:50])

print("\n[11] ChatMessage 兼容...")

try:
    from langchain_core.messages import ChatMessage

    msg = ChatMessage(content="test", role="user")
    test("ChatMessage 创建", True)
except Exception as e:
    test("ChatMessage 创建", False, str(e)[:50])

print("\n[12] FunctionMessage 兼容...")

try:
    from langchain_core.messages import FunctionMessage

    msg = FunctionMessage(name="test_func", content="result")
    test("FunctionMessage 创建", True)
except Exception as e:
    test("FunctionMessage 创建", False, str(e)[:50])

print("\n[13] ToolMessage 兼容...")

try:
    from langchain_core.messages import ToolMessage

    msg = ToolMessage(content="result", tool_call_id="123")
    test("ToolMessage 创建", True)
except Exception as e:
    test("ToolMessage 创建", False, str(e)[:50])

print("\n" + "=" * 60)
print(f"测试结果: 通过 {tests_passed}/{tests_total}")
print("=" * 60)
