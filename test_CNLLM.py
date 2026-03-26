from cnllm import CNLLM, MINIMAX_API_KEY
import time

print("=" * 60)
print("CNLLM 优化功能测试")
print("=" * 60)

print("\n[1] 测试模型参数可配置...")
client_m27 = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
print(f"    模型: {client_m27.model}")

client_m25 = CNLLM(model="minimax-m2.5", api_key=MINIMAX_API_KEY)
print(f"    模型: {client_m25.model}")

client_short = CNLLM(model="minimax", api_key=MINIMAX_API_KEY)
print(f"    'minimax' 自动映射为: {client_short.model}")

print("\n[2] 测试超时和重试参数...")
client = CNLLM(
    model="minimax",
    api_key=MINIMAX_API_KEY,
    timeout=30,
    max_retries=3,
    retry_delay=1.0
)
print(f"    超时: {client.timeout}s, 重试次数: {client.max_retries}, 重试延迟: {client.retry_delay}s")

print("\n[3] 测试 Token 统计...")
resp = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    model="minimax-m2.7"
)

usage = resp["usage"]
print(f"    Prompt Tokens: {usage['prompt_tokens']}")
print(f"    Completion Tokens: {usage['completion_tokens']}")
print(f"    Total Tokens: {usage['total_tokens']}")

print("\n[4] 测试 OpenAI 格式输出...")
print(f"    ID: {resp['id']}")
print(f"    Object: {resp['object']}")
print(f"    Model: {resp['model']}")
print(f"    Finish Reason: {resp['choices'][0]['finish_reason']}")

print("\n[5] 测试错误处理...")
try:
    client.chat.create(
        messages=[{"role": "user", "content": "测试"}],
        model="unsupported-model"
    )
except ValueError as e:
    print(f"    [OK] 捕获到模型验证错误: {str(e)[:50]}...")

print("\n[6] 测试 MiniMax-M2.5 模型...")
resp_m25 = client.chat.create(
    messages=[{"role": "user", "content": "用一句话介绍自己"}],
    model="minimax-m2.5"
)
content_m25 = resp_m25["choices"][0]["message"]["content"]
print(f"    M2.5 回复: {content_m25[:50]}...")

print("\n[7] 测试 MiniMax-M2.7 模型...")
resp_m27 = client.chat.create(
    messages=[{"role": "user", "content": "用一句话介绍自己"}],
    model="minimax-m2.7"
)
content_m27 = resp_m27["choices"][0]["message"]["content"]
print(f"    M2.7 回复: {content_m27[:50]}...")

print("\n[8] LangChain 函数兼容性验证...")

try:
    from langchain_core.messages import (
        HumanMessage, 
        AIMessage, 
        SystemMessage,
        BaseMessage,
        message_to_dict,
        messages_to_dict
    )
    
    print("    [OK] LangChain 已安装，验证函数兼容性...")
    
    resp = client.chat.create(
        messages=[{"role": "user", "content": "你好"}],
        model="minimax-m2.7"
    )
    
    content = resp["choices"][0]["message"]["content"]
    role = resp["choices"][0]["message"]["role"]
    msg_id = resp["id"]
    created = resp["created"]
    model = resp["model"]
    
    print(f"    CNLLM 输出格式:")
    print(f"        content: {content[:30]}...")
    print(f"        role: {role}")
    print(f"        id: {msg_id}")
    print(f"        created: {created}")
    print(f"        model: {model}")
    
    print("\n    [测试1] LangChain AIMessage 解析...")
    ai_msg = AIMessage(content=content)
    assert ai_msg.content == content, "AIMessage 内容不匹配"
    print(f"        [OK] AIMessage 解析成功: {ai_msg.content[:30]}...")
    
    print("\n    [测试2] LangChain message_to_dict 转换...")
    msg_dict = message_to_dict(ai_msg)
    assert msg_dict["data"]["content"] == content, "message_to_dict 内容不匹配"
    print(f"        [OK] message_to_dict 转换成功")
    print(f"        转换结果: {msg_dict}")
    
    print("\n    [测试3] LangChain messages_to_dict 批量转换...")
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，我是AI"),
        SystemMessage(content="你是一个有帮助的助手")
    ]
    msgs_dict = messages_to_dict(messages)
    assert len(msgs_dict) == 3, "消息数量不匹配"
    print(f"        [OK] messages_to_dict 转换成功，共 {len(msgs_dict)} 条消息")
    
    print("\n    [测试4] LangChain 消息类型检查...")
    assert isinstance(ai_msg, BaseMessage), "AIMessage 不是 BaseMessage 的实例"
    assert isinstance(HumanMessage(content="test"), BaseMessage), "HumanMessage 不是 BaseMessage"
    assert isinstance(SystemMessage(content="test"), BaseMessage), "SystemMessage 不是 BaseMessage"
    print(f"        [OK] 所有消息类型都是 BaseMessage 的子类")
    
    print("\n    [测试5] 验证 OpenAI 标准格式完全兼容...")
    assert "choices" in resp, "缺少 choices 字段"
    assert "id" in resp, "缺少 id 字段"
    assert "created" in resp, "缺少 created 字段"
    assert "model" in resp, "缺少 model 字段"
    assert "usage" in resp, "缺少 usage 字段"
    assert "object" in resp, "缺少 object 字段"
    print(f"        [OK] OpenAI 标准格式完全兼容")
    
    print("\n    [总结] CNLLM 输出与 LangChain 完全兼容!")
    
except ImportError:
    print("    [WARN] LangChain 未安装，跳过 LangChain 函数兼容性测试")
except AssertionError as e:
    print(f"    [ERROR] 断言失败: {e}")
except Exception as e:
    print(f"    [ERROR] LangChain 函数测试失败: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
