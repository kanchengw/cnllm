# CNLLM - Chinese LLM Adapter

[English](README_en.md) | 中文

[!\[PyPI Version\](https://img.shields.io/pypi/v/cnllm null)](https://pypi.org/project/cnllm/)
[!\[Python Versions\](https://img.shields.io/pypi/pyversions/cnllm null)](https://pypi.org/project/cnllm/)
[!\[License\](https://img.shields.io/github/license/kanchengw/cnllm null)](LICENSE)

***

<<<<<<< HEAD
中文大模型适配库，将模型 API 响应封装为 OpenAI 格式，无缝协作langchain、LlamaIndex、Pydantic等机器学习库
=======
中文大模型适配库，将模型 API返回封装为 OpenAI 格式，无缝协作langchain、LlamaIndex、Pydantic等机器学习库
>>>>>>> origin/main

## 更新日志

### v0.3.2 (2026-04-01) ✨

- ✨ **参数统一** - 客户端初始化参数与调用入口参数统一化，调用入口灵活覆写
- ✨ **架构优化** - 核心逻辑抽象，模型适配器BaseAdapter和响应转换器Responder处理通用逻辑
- ✨ **可扩展性** - 接入新厂商只需配置相应 YAML 文件，自动实现请求和响应双端字段映射、错误码映射，无需修改其他上层组件
- ✨ **YAML 功能集成** - 关联字段映射、模型支持验证、必填项验证、参数支持验证、厂商错误码映射逻辑
- ✨ **MiniMax 支持优化** - 支持 MiniMax 原生接口所有参数，如 `top_p`、`tools`、`thinking` 等

### v0.3.1 (2026-03-29) ✨

- ✨ **LangChain深度适配**
  - Runnable 适配器作为核心功能，一个函数接入Langchain chain
  - Runnable 流式输出、批量调用、异步调用支持
- ✨ **chat.create() 流式输出** - `stream=True` 参数支持
- ✨ **Fallback 机制** - 主模型失败时自动切换到备用模型
- ✨ **响应入口** - `client.chat.still` 轻松获取纯净会话响应，`client.chat.raw` 获取完整响应
- 🔧 **适配器重构** - 模型适配器（中文大模型 MiniMax 等）+ 框架适配器（LangChain 等）双层架构

<<<<<<< HEAD
=======
### v0.3.0 (2026-03-28)

- ✨ __call__ 极简调用、prompt 参数、模型覆盖机制

>>>>>>> origin/main
## 特性

- **OpenAI 兼容** - 模型输出对齐 OpenAI API 标准格式
- **框架集成** - 适配 LangChain、LlamaIndex 等主流机器学习库
- **统一接口** - 一套代码，无缝切换不同国产大模型
- **简洁调用** - 支持多种调用方式，最简只需一行代码

## 支持的模型

- **已验证**：MiniMax-M2.7、MiniMax-M2.5、MiniMax-M2.1、MiniMax-M2
- **更多厂商、模型支持正在开发中**

## 安装

```bash
pip install cnllm
```

## 快速开始

### 初始化接口

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
```

### 三种调用入口

**1. 极简调用** **`client("提示词")`**

```python
resp = client("用一句话介绍自己")  # 极简调用不接受其他参数
```

**2. 标准调用** **`client.chat.create(prompt="提示词")`**

```python
resp = client.chat.create(prompt="用一句话介绍自己")
```

**3. 完整调用** **`client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "用一句话介绍自己"}
    ]
)
```

### 调用时快捷切换模型：

```python
resp = client.chat.create(
    prompt="介绍自己",
    model="minimax-m2.5",  # 可选，覆盖模型
<<<<<<< HEAD
    api_key="your_other_api_key"  # 可选，覆盖 API Key，不填则默认使用客户端入口处的key
=======
    api_key="your_other_api_key"  # 可选，覆盖 API Key
>>>>>>> origin/main
)
```

### 响应入口
<<<<<<< HEAD

**1. 获取纯净会话响应**

=======
**1. 获取纯净会话响应**
>>>>>>> origin/main
```python
# 传统方式
print(resp["choices"][0]["message"]["content"])

# 使用 still 属性（推荐）
print(client.chat.still)
```

**2. 获取完整响应**

```python
print(client.chat.raw)  # 模型返回的原始响应
```

## 统一接口参数

| 参数                  | 类型          | 必填 | 默认值      | 客户端入口 | 调用入口 | 说明                              |
| ------------------- | ----------- | -- | -------- | :------: | :------: | ------------------------------- |
| `model`             | str         | ✅  | -        |    ✅    |    ✅    | 如minimax-m2.7或MiniMax-m2.7  |
| `api_key`           | str         | ✅  | -        |    ✅    |    ✅    | API 密钥                          |
| `messages`          | list\[dict] | ⚠️ | -        |    ❌    |    ✅    | OpenAI 格式消息列表（与 prompt 二选一）     |
| `prompt`            | str         | ⚠️ | -        |    ❌    |    ✅    | 简写参数（与 messages 二选一） |
| `fallback_models`   | dict        | -  | {}       |    ✅    |    ❌    | 备用模型配置                          |
| `base_url`          | str         | -  | API 默认地址 |    ✅    |    ✅    | 自定义 API 地址                      |
| `timeout`           | int         | -  | 60       |    ✅    |    ✅    | 请求超时（秒）                         |
| `max_retries`       | int         | -  | 3        |    ✅    |    ✅    | 最大重试次数                          |
| `retry_delay`       | float       | -  | 1.0      |    ✅    |    ✅    | 重试延迟（秒）                         |
| `temperature`       | float       | -  | 0.7      |    ✅    |    ✅    | 生成随机性，0-2                       |
| `max_tokens`        | int         | -  | None     |    ✅    |    ✅    | 最大生成 token 数                    |
| `stream`            | bool        | -  | False    |    ✅    |    ✅    | 流式响应                            |
| `top_p`             | float       | -  | 0.95     |    ✅    |    ✅    | 核采样阈值                           |
| `top_k`             | int         | -  | -        |    ✅    |    ✅    | Top-K 采样                        |
| `tools`             | list        | -  | -        |    ✅    |    ✅    | 函数工具定义                          |
| `tool_choice`       | str         | -  | -        |    ✅    |    ✅    | 工具选择模式：none / auto              |
| `thinking`          | bool        | -  | -        |    ✅    |    ✅    | 思考模式（MiniMax-M1）                |
| `presence_penalty`  | float       | -  | -        |    ✅    |    ✅    | 存在惩罚                            |
| `frequency_penalty` | float       | -  | -        |    ✅    |    ✅    | 频率惩罚                            |
| `stop`              | str/list    | -  | -        |    ✅    |    ✅    | 停止序列                            |
| `user`              | str         | -  | -        |    ✅    |    ✅    | 用户标识                            |
| `organization`      | str         | -  | -        |    ✅    |    ✅    | 使用MiniMax时会自动映射为MiniMax标准字段group_id               |

<<<<<<< HEAD
**说明**：

- 调用入口参数优先，复用客户端建议传入常用参数，单次调用时可灵活覆盖和传入更多参数。
- 必填项仅需保证请求发起时不为空，即客户端和调用入口至少有一次传入该参数。
=======
| 参数                | 类型    | 必填 | 默认值      | 说明                                                                           |
| ----------------- | ----- | -- | -------- | ---------------------------------------------------------------------------- |
| `model`           | str   | ✅  | -        | 模型名称：minimax-m2.7、minimax-m2.5                                               |
| `api_key`         | str   | ✅  | -        | API 密钥                                                                       |
| `base_url`        | str   | -  | API 默认地址 | 自定义 API 地址                                                                   |
| `timeout`         | int   | -  | 30       | 请求超时（秒）                                                                      |
| `max_retries`     | int   | -  | 3        | 最大重试次数                                                                       |
| `retry_delay`     | float | -  | 1.0      | 重试延迟（秒）                                                                      |
| `fallback_models` | dict  | -  | {}       | 备用模型配置，格式：`{"备用模型": "api_key", ...}`，api\_key 为 None 时与主模型共用API key，接受多个备用模型 |

### 两种调用接口

#### client.chat.create() 参数

| 参数            | 类型          | 必填 | 默认值   | 说明                          |
| ------------- | ----------- | -- | ----- | --------------------------- |
| `messages`    | list\[dict] | ⚠️ | -     | OpenAI 格式消息列表（与 prompt 二选一） |
| `prompt`      | str         | ⚠️ | -     | 简写参数（与 messages 二选一）        |
| `model`       | str         | -  | None  | 覆盖默认模型                      |
| `api_key`     | str         | -  | None  | 覆盖默认 API Key                |
| `temperature` | float       | -  | 0.7   | 生成随机性，0-2                   |
| `max_tokens`  | int         | -  | None  | 最大生成 token 数                |
| `stream`      | bool        | -  | False | 流式响应                        |
>>>>>>> origin/main

#### 极简调用 client()

直接传入提示词字符串，无额外参数。

## 返回格式

通过快速开始中的任意 API 调用方式，模型的响应将会封装为 OpenAI 标准格式：

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

<<<<<<< HEAD
OpenAI 标准响应结构适配 LangChain 库（深度集成Runnable组件），其他库如 Pydantic、LlamaIndex、Instructor 等支持 OpenAI 标准结构的库应都能直接使用（未验证）。
=======
适配 LangChain 库（深度集成Runnable组件），其他库如 Pydantic、LlamaIndex、Instructor 等支持 OpenAI 标准结构的库应都能直接使用（未验证）。
>>>>>>> origin/main

## LangChainRunnable实现

```python
from cnllm import CNLLM
from cnllm.core.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")

# 使用LangChainRunnable 包裹客户端
runnable = LangChainRunnable(client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个热心的智能助手"),("human", "{input}")])

# 构建LangChain chain
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

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>

