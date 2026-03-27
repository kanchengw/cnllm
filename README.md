# CNLLM - Chinese LLM Adapter

[English](README_en.md) | 中文

![PyPI Version](https://img.shields.io/pypi/v/cnllm)
![Python Versions](https://img.shields.io/pypi/pyversions/cnllm)
![License](https://img.shields.io/github/license/kanchengw/cnllm)
![GitHub Stars](https://img.shields.io/github/stars/kanchengw/cnllm?style=social)
![GitHub Forks](https://img.shields.io/github/forks/kanchengw/cnllm?style=social)

---

统一的中文大模型适配库，将各种国产大模型（如 MiniMax、字节豆包、Kimi 等）的 API 输出转换为统一的 OpenAI 格式，零成本接入 LangChain 等主流 AI 框架。

## 更新日志

### v0.2.0 (2026-03-27)
- ✨ **新增 `__call__` 方法**：支持 `client("提示词")` 极简调用
- ✨ **新增 prompt 参数**：支持直接传入字符串，无需手动构建 messages
- ✨ **模型覆盖机制**：支持调用时指定 API 可通用的其他模型
- ✨ **LangChain 兼容性测试**：13 个函数兼容性验证
- 📝 **README 重构**：精简结构，添加 API 参数说明、调用方式示例

## 特性

- **OpenAI 兼容** - 所有输出完全对齐 OpenAI API 标准格式，可直接接入 LangChain、LlamaIndex 等主流框架
- **统一接口** - 一套代码，无缝切换不同大模型
- **模型覆盖** - 支持调用时指定 API 可通用的其他模型
- **简洁调用** - 支持多种调用方式，最简只需一行代码
- **流式输出** - 支持流式响应（规划中）
- **重试机制** - 内置超时和自动重试

## 支持的模型

- **已验证**：MiniMax-M2.7、MiniMax-M2.5
- **更多厂商、模型支持正在开发中**

## 安装

```bash
pip install cnllm
```

或从源码安装：

```bash
git clone https://github.com/kanchengw/cnllm.git
cd cnllm
pip install -e .
```

## 快速开始

### 三种调用方式

**1. 极简调用 `client("提示词")`**

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
resp = client("用一句话介绍自己")
print(resp["choices"][0]["message"]["content"])
```

**2. 标准调用 `client.chat.create(prompt="提示词")`**

```python
resp = client.chat.create(prompt="用一句话介绍自己")
print(resp["choices"][0]["message"]["content"])
```

**3. 完整调用 `client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "用一句话介绍自己"}
    ]
)
print(resp["choices"][0]["message"]["content"])
```

### 模型覆盖

支持调用时覆盖默认模型（适用于同一 API 可访问的多个模型）：

```python
resp = client.chat.create(
    prompt="用一句话介绍自己",
    model="minimax-m2.5"  # 覆盖初始化时的模型
)
```

## API 参数

### CNLLM 客户端初始化

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | str | ✅ | - | 模型名称：`minimax-m2.7`、`minimax-m2.5` |
| `api_key` | str | ✅ | - | API 密钥 |
| `timeout` | int | - | 30 | 请求超时（秒） |
| `max_retries` | int | - | 3 | 最大重试次数 |
| `retry_delay` | float | - | 1.0 | 重试延迟（秒） |

### chat.create() 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `messages` | list[dict] | ⚠️ | - | OpenAI 格式消息列表（与 prompt 二选一） |
| `prompt` | str | ⚠️ | - | 简写参数，会自动转为 messages（与 messages 二选一） |
| `temperature` | float | - | 0.1 | 生成随机性，0-2 |
| `stream` | bool | - | False | 流式响应（规划中） |
| `model` | str | - | None | 覆盖默认模型 |

### __call__ 简写参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | str | ✅ | - | 提示词 |
| `temperature` | float | - | 0.1 | 生成随机性 |
| `model` | str | - | None | 覆盖默认模型 |

## 返回格式

所有 API 返回均为 OpenAI 标准格式：

```python
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "minimax-m2.7",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "我是 MiniMax-M2.7..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

## LangChain 兼容函数

| 函数/类 | 状态 | 说明 |
|---------|------|------|
| `HumanMessage` | ✅ | 人类消息类型 |
| `AIMessage` | ✅ | AI 消息类型 |
| `SystemMessage` | ✅ | 系统消息类型 |
| `BaseMessage` | ✅ | 消息基类 |
| `ChatPromptTemplate` | ✅ | 聊天提示模板 |
| `StrOutputParser` | ✅ | 字符串输出解析器 |
| `message_to_dict` | ✅ | 消息转字典 |
| `messages_to_dict` | ✅ | 批量消息转字典 |
| `AIMessageChunk` | ✅ | AI 消息块 |
| `ChatMessage` | ✅ | 聊天消息 |
| `FunctionMessage` | ✅ | 函数消息 |
| `ToolMessage` | ✅ | 工具消息 |

## 在 LangChain 中使用

CNLLM 返回标准 OpenAI 格式，可直接被 LangChain 函数使用：

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import message_to_dict, messages_to_dict
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

# 调用 CNLLM 获取响应
resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])
print(resp["choices"][0]["message"]["content"])

# CNLLM 返回格式符合 OpenAI 标准，可直接被 LangChain 使用
ai_msg = AIMessage(content=resp["choices"][0]["message"]["content"])
print(f"Role: {ai_msg.type}")  # "ai"
print(f"Content: {ai_msg.content}")

# 转换为 LangChain 字典格式
msg_dict = message_to_dict(ai_msg)
print(msg_dict)
# {'type': 'ai', 'data': {'content': '...', ...}}

# 批量转换
msgs = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!")
]
msgs_dict = messages_to_dict(msgs)
print(msgs_dict)
```

## 其他兼容框架

CNLLM 输出兼容所有使用 OpenAI 格式的 Python 库：

- **LangChain** - 消息类型、链式调用
- **LlamaIndex** - 索引和查询
- **AutoGen** - 多代理协作（规划中）
- **CrewAI** - 多代理工作流
- **Dify** - 平台集成

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: [https://github.com/kanchengw/cnllm/issues](https://github.com/kanchengw/cnllm/issues)
