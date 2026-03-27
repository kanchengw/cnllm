# CNLLM - Chinese LLM Adapter

[English](README_en.md) | 中文

[!\[PyPI Version\](https://img.shields.io/pypi/v/cnllm null)](https://pypi.org/project/cnllm/)
[!\[Python Versions\](https://img.shields.io/pypi/pyversions/cnllm null)](https://pypi.org/project/cnllm/)
[!\[License\](https://img.shields.io/github/license/kanchengw/cnllm null)](LICENSE)

***

统一的中文大模型适配库，将各种国产大模型API 输出转换为统一的 OpenAI 格式，无痛接入 LangChain 等主流 AI 框架。

## 更新日志

### v0.3.0 (2026-03-28) ✨

- ✨ **LangChain深度适配**
  - Runnable 适配器作为核心功能，一个函数接入Langchain chain
  - Runnable 流式输出、批量调用、异步调用支持
- ✨ **chat.create() 流式输出** - `stream=True` 参数支持
- ✨ **extra\_config** - 各大模型厂商特有参数统一管理
- 🔧 **适配器重构** - 模型适配器（中文大模型 Minimax 等）+ 框架适配器（LangChain 等）双层架构

### v0.2.0 (2026-03-27)

- ✨ __call__ 极简调用、prompt 参数、模型覆盖机制

## 特性

- **OpenAI 兼容** - 模型输出对齐 OpenAI API 标准格式
- **框架集成** - 适配 LangChain、LlamaIndex 等主流机器学习库
- **统一接口** - 一套代码，无缝切换不同国产大模型
- **简洁调用** - 支持多种调用方式，最简只需一行代码

## 支持的模型

- **已验证**：MiniMax-M2.7、MiniMax-M2.5
- **更多厂商、模型支持正在开发中**

## 安装

```bash
pip install cnllm
```

## 快速开始

### 三种调用方式

**1. 极简调用** **`client("提示词")`**

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
resp = client("用一句话介绍自己")
print(resp["choices"][0]["message"]["content"])
```

**2. 标准调用** **`client.chat.create(prompt="提示词")`**

```python
resp = client.chat.create(prompt="用一句话介绍自己")
print(resp["choices"][0]["message"]["content"])
```

**3. 完整调用** **`client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "用一句话介绍自己"}
    ]
)
print(resp["choices"][0]["message"]["content"])
```

### 模型覆盖

支持调用时覆盖默认模型：

```python
resp = client.chat.create(
    prompt="介绍自己",
    model="minimax-m2.5"
)
```

## LangChainRunnable实现

```python
from cnllm import CNLLM
from cnllm.adapters.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")
runnable = LangChainRunnable(client)

# 带输入变量的模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个热心的智能助手"),
    ("human", "{input}")
])
chain = prompt | runnable
result = chain.invoke({"input": "2+2等于几？"})
print(result.content)

# 同步流式输出
for chunk in runnable.stream("数到5"):
    print(chunk, end="", flush=True)

# 异步流式输出
async def async_stream_test():
    async for chunk in runnable.astream("数到3"):
        print(chunk, end="", flush=True)

asyncio.run(async_stream_test())

# 批量调用
results = runnable.batch(["Hello", "How are you?"])
for r in results:
    print(r.content)
```

此外，采用任意快速开始中的调用方法，输出皆适配 LangChain 消息类型（HumanMessage、AIMessage、SystemMessage 等）、提示模板（ChatPromptTemplate）、输出解析器（StrOutputParser）等组件。

## 统一接口参数

### CNLLM 客户端接口

| 参数             | 类型    | 必填 | 默认值      | 说明                             |
| -------------- | ----- | -- | -------- | ------------------------------ |
| `model`        | str   | ✅  | -        | 模型名称：minimax-m2.7、minimax-m2.5 |
| `api_key`      | str   | ✅  | -        | API 密钥                         |
| `base_url`     | str   | -  | API 默认地址 | 自定义 API 地址                     |
| `organization` | str   | -  | None     | OpenAI 组织标识（多团队追踪）             |
| `extra_config` | dict  | -  | {}       | 厂商特有参数（见下方说明）                  |
| `timeout`      | int   | -  | 30       | 请求超时（秒）                        |
| `max_retries`  | int   | -  | 3        | 最大重试次数                         |
| `retry_delay`  | float | -  | 1.0      | 重试延迟（秒）                        |

#### extra\_config 参数

| 厂商      | 参数                          | 说明              |
| ------- | --------------------------- | --------------- |
| MiniMax | `group_id`                  | 多用户/计费用 GroupId |
| 豆包      | `app_id`, `space_id`        | 应用/空间标识         |
| Kimi    | `project_id`, `api_version` | 项目/版本标识         |

### chat.create() 调用接口

| 参数                  | 类型          | 必填 | 默认值   | 说明                                  |
| ------------------- | ----------- | -- | ----- | ----------------------------------- |
| `messages`          | list\[dict] | ⚠️ | -     | OpenAI 格式消息列表（与 prompt 二选一）         |
| `prompt`            | str         | ⚠️ | -     | 简写参数，会自动转为 messages（与 messages 二选一） |
| `model`             | str         | -  | None  | 覆盖默认模型                              |
| `temperature`       | float       | -  | 0.7   | 生成随机性，0-2                           |
| `max_tokens`        | int         | -  | None  | 最大生成 token 数                        |
| `top_p`             | float       | -  | 1.0   | 核采样参数，0-1                           |
| `n`                 | int         | -  | 1     | 生成候选数量                              |
| `stream`            | bool        | -  | False | 流式响应                                |
| `stop`              | str/list    | -  | None  | 停止词                                 |
| `presence_penalty`  | float       | -  | 0.0   | 存在惩罚，-2 to 2                        |
| `frequency_penalty` | float       | -  | 0.0   | 频率惩罚，-2 to 2                        |
| `user`              | str         | -  | None  | 用户标识                                |
| `extra_config`      | dict        | -  | None  | 厂商特有参数                              |

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

## 其他兼容框架

- **LlamaIndex** - 索引和查询（规划中）
- **AutoGen** - 多代理协作（规划中）
- **CrewAI** - 多代理工作流

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>

