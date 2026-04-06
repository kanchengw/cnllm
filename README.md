# CNLLM - Chinese LLM Adapter

[English](README_en.md) | 中文

[![PyPI Version](https://img.shields.io/pypi/v/cnllm null)](https://pypi.org/project/cnllm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/cnllm null)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm null)](LICENSE)

***
## 项目背景

在学术研究与实际生产环境中，如何将中文大模型高效接入成熟的机器学习框架一直是开发者面临的核心挑战。

当前主流方案存在明显局限：使用 OpenAI 兼容接口虽简单易用，但无法充分发挥各厂商原生能力；而直接调用原生接口则意味着需要自行处理响应解析、格式转换等繁琐工作。

CNLLM 致力于解决这一两难困境——通过提供一个调用中文大模型的**统一的接口**与一套**一致的参数规范**，在完整释放中文大模型原生能力的同时，将形态各异的响应自动转换为 OpenAI 标准格式。无论是 langchain、LlamaIndex 或其他机器学习框架，都能以相同的方式接入各类大模型；另外，在需要多模型协作的场景下，也能保持一致的接口、参数和响应格式。

> 由于深感能力和精力有限，欢迎志同道合的朋友共同参与 CNLLM 的建设：[wangkancheng1122@163.com](mailto:wangkancheng1122@163.com)

### 我们期待这样的合作

| 方向 | 说明 |
|------|------|
| 🌐 **新厂商适配** | 接入更多中文大模型（如阿里千问、字节豆包、Kimi 等） |
| 🔗 **框架适配** | 深化与 LlamaIndex、LiteLLM 等框架的集成 |
| 🐛 **能力扩展** | Embedding、多模态等功能的适配框架开发 |
| 📖 **文档完善** | 补充使用案例、优化开发指南 |
| 💡 **功能建议** | 提出您的想法与需求 |

快速入门：[贡献者指南](docs/CONTRIBUTOR.md)
详细架构：[系统架构](docs/ARCHITECTURE.md)


## 更新日志

### v0.5.0 (2026-04-06)

- ✨ **KIMI (Moonshot AI) 适配** - Kimi 模型适配开发，支持 kimi-k2.5、kimi-k2 系列和 moonshot-v1 系列（8k/32k/128k）
- ✨ **DeepSeek 适配** - DeepSeek 模型适配开发，支持 `deepseek-chat` 和 `deepseek-reasoner` 两个模型
- 现在 CNLLM 在标准响应中会包含 `system_fingerprint` 和 `choices[0].logprobs` 字段

### v0.4.3 (2026-04-06)

- ✨ **豆包Doubao适配** - 豆包Doubao Seed系列模型适配开发，支持 seed-2.0系列、seed-1.6系列和 seed-1.8等 9 个模型(具体见`支持的模型列表`)，支持豆包原生参数，如`stream_options`、`reasoning_effort`、`service_tier` 等
  - 支持 `reasoning_effort` 推理长度字段， `minimal`、`low`、`medium`、`high`四档位切换
  - 支持 `thinking` 字段，`true`(enabled)、`false`(disabled)、`auto`三档位切换，其中`thinking="auto"`仅在 doubao-seed-1-6 模型中生效
- 🔧 **已知 bug 修复** - 修复流式响应中 `_collect_stream_result` 重复调用导致内容累积异常的 bug

### v0.4.2 (2026-04-05)

- ✨ **智谱GLM适配** - 智谱GLM模型适配开发，支持"glm-4.6"、"glm-5"、"glm-5-turbo"和 GLM 4.7系列模型
  - 支持智谱原生参数，如`do_sample`、`request_id`、`response_format`、`tool_stream`、`thinking`等
- 🔧 **已知 bug 修复** - 修复`id`字段的响应映射

### v0.4.1 (2026-04-04)

- 🔧 **已知 bug 修复**

### v0.4.0（2026-04-03）

- ✨ **mimo适配** - 小米mimo模型适配开发，支持"mimo-v2-pro"、"mimo-v2-omni"、"mimo-v2-flash"
- ✨ **架构重构** - BaseAdapter + Responder + VendorError 三层架构分离，职责清晰
- ✨ **.think 属性** - `client.chat.think` 获取 reasoning_content，支持流式累积
- ✨ **.tools 属性** - `client.chat.tools` 获取 tool_calls，支持流式累积
- ✨ **流式累积** - `.think`、`.still`、`.tools` 支持在流式响应中实时滚动积累


### v0.3.3 (2026-04-02) ✨

- ✨ **参数统一** - 客户端初始化参数与调用入口参数统一化，调用入口灵活覆写
- ✨ **架构优化** - 核心逻辑抽象，模型适配器BaseAdapter和响应转换器Responder处理通用逻辑
- ✨ **可扩展性** - 接入新厂商只需配置相应 YAML 文件，自动实现请求和响应双端字段映射、错误码映射，无需修改其他上层组件
- ✨ **YAML 功能集成** - 关联字段映射、模型支持验证、必填项验证、参数支持验证、厂商错误码映射逻辑
- ✨ **MiniMax 支持优化** - 支持 MiniMax 原生接口所有参数，如 `top_p`、`tools`、`thinking` 等

### v0.3.2 (2026-03-29) ✨

- ✨ **LangChain深度适配**
  - Runnable 适配器作为核心功能，一个函数接入Langchain chain
  - Runnable 流式输出、批量调用、异步调用支持
- ✨ **chat.create() 流式输出** - `stream=True` 参数支持
- ✨ **Fallback 机制** - 主模型失败时自动切换到备用模型
- ✨ **响应入口** - `client.chat.still` 轻松获取纯净会话响应，`client.chat.raw` 获取完整响应
- 🔧 **适配器重构** - 模型适配器（中文大模型 MiniMax 等）+ 框架适配器（LangChain 等）双层架构

## 特性

- **OpenAI 兼容** - 模型输出对齐 OpenAI API 标准格式
- **框架集成** - 适配 LangChain、LlamaIndex 等主流机器学习库
- **统一接口** - 一套代码，无缝切换不同国产大模型
- **简洁调用** - 支持多种调用方式，最简只需一行代码

## 支持的模型

- **DeepSeek**：deepseek-chat、deepseek-reasoner
- **KIMI (Moonshot AI)**：kimi-k2.5、kimi-k2-thinking、kimi-k2-thinking-turbo、kimi-k2-turbo-preview、kimi-k2-0905-preview、moonshot-v1-8k、moonshot-v1-32k、moonshot-v1-128k
- **豆包Doubao**：doubao-seed-2-0-pro、doubao-seed-2-0-mini、doubao-seed-2-0-lite、doubao-seed-2-0-code、doubao-seed-1-8、doubao-seed-1-6、doubao-seed-1-6-lite、doubao-seed-1-6-flash
- **智谱GLM**：glm-4.6、glm-4.7、glm-4.7-flash、glm-4.7-flashx、glm-5、glm-5-turbo
- **小米mimo**：mimo-v2-pro、mimo-v2-omni、mimo-v2-flash
- **MiniMax**：MiniMax-M2.7、MiniMax-M2.5、MiniMax-M2.1、MiniMax-M2
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
    api_key="your_other_api_key"  # 可选，覆盖 API Key，不填则默认使用客户端入口处的key
)
```

### 响应入口

**会话响应入口**

```python
client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
resp = client.chat.create(prompt="用一句话介绍自己", ...)

# 传统方式
print(resp["choices"][0]["message"]["content"])

# 使用 still 属性（推荐）
print(client.chat.still)     # 返回：你好，我是minimax-m2.7模型...

# 获取 raw 原始响应
print(client.chat.raw)     # 返回：{厂商原生响应的 JSON 字符串}
```

**获取模型思考过程（reasoning_content）**

```python
resp = client.chat.create(thinking=True, ...)

print(client.chat.think)     # 返回：Let me think about this, user asked me to ...
```

**获取工具调用消息（tool_calls）**

```python
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]
resp = client.chat.create(tools=tools, ...)

print(client.chat.tools)     # 返回：{工具调用消息字典}
```

## CNLLM 统一接口参数

| 参数                  | 类型          | 必填 | 默认值      | 客户端入口 | 调用入口 | 说明                              |
| ------------------- | ----------- | -- | -------- | :------: | :------: | ------------------------------- |
| `model`             | str         | ✅  | -        |    ✅    |    ✅    | 客户端初始化必填  |
| `api_key`           | str         | ✅  | -        |    ✅    |    ✅    | API 密钥                          |
| `messages`          | list\[dict] | ⚠️ | -        |    ❌    |    ✅    | OpenAI 格式消息列表（与 prompt 二选一）     |
| `prompt`            | str         | ⚠️ | -        |    ❌    |    ✅    | 简写参数（与 messages 二选一） |
| `fallback_models`   | dict        | -  | -       |    ✅    |    ❌    | 备用模型配置（具体见文档底部的 FallbackManager 流程设计）                          |
| `base_url`          | str         | -  | 自动适配支持模型的默认地址 |    ✅    |    ✅    | 自定义 API 地址                      |
| `timeout`           | int         | -  | 60       |    ✅    |    ✅    | 请求超时（秒）                         |
| `max_retries`       | int         | -  | 3        |    ✅    |    ✅    | 最大重试次数                          |
| `retry_delay`       | float       | -  | 1.0      |    ✅    |    ✅    | 重试延迟（秒）                         |
| `temperature`       | float       | -  | 端口默认值 |    ✅    |    ✅    | 生成随机性                             |
| `max_tokens`        | int         | -  | 端口默认值 |    ✅    |    ✅    | 最大生成 token 数                    |
| `stream`            | bool        | -  | 端口默认值，一般为False |    ✅    |    ✅    | 流式响应                            |
| `top_p`             | float       | -  | 端口默认值 |    ✅    |    ✅    | 核采样阈值                           |
| `tools`             | list        | -  | -        |    ✅    |    ✅    | 函数工具定义                          |
| `tool_choice`       | str         | -  | -        |    ✅    |    ✅    | 工具选择模式：none / auto              |
| `thinking`          | bool        | -  | 端口默认值 |    ✅    |    ✅    | 思考模式，统一格式为`thinking=True/False`                            |
| `presence_penalty`  | float       | -  | 端口默认值 |    ✅    |    ✅    | 存在惩罚                            |
| `frequency_penalty` | float       | -  | 端口默认值 |    ✅    |    ✅    | 频率惩罚                            |
| `organization`      | str         | -  | -        |    ✅    |    ✅    | 使用 MiniMax 时自动映射为 group_id      |
| `stop`              | str/list    | -  | -        |    ✅    |    ✅    | 停止序列                            |
| `user`              | str         | -  | -        |    ✅    |    ✅    | 用户标识                            |
| `response_format`   | dict        | -  | 端口默认值，一般为{type:"text"} |    ✅    |    ✅    | 响应格式                            |

**说明**：

- 模型支持的更多参数请参考官方文档， CNLLM 会透传具体模型支持的所有参数。
- 调用入口参数优先，复用客户端建议传入常用参数，单次调用时可灵活覆盖和传入更多参数。
- 必填项仅需保证请求发起时不为空，即客户端和调用入口至少有一次传入该参数。

#### 极简调用 client()

直接传入提示词字符串，无额外参数。

## CNLLM 标准返回格式

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
        "logprobs": null,
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "system_fingerprint": "fp_xxx"
}
```

OpenAI 标准响应结构适配 LangChain 库（深度集成Runnable组件），其他库如 Pydantic、LlamaIndex、Instructor 等支持 OpenAI 标准结构的库应都能直接使用（未验证）。

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

##  FallbackManager 模型选择的流程设计

只有客户端初始化入口接受配置`fallback_models`参数，为追求程序或应用运行时的稳定性建议配置此项。
当客户端入口处的主模型不可用时，会按顺序尝试`fallback_models`中的模型。
代码示例：

```python
client = CNLLM(
    model="minimax-m2.7", api_key="minimax_key", 
    fallback_models={"mimo-v2-flash": "xiaomi-key", "minimax-m2.5": None}  # None 表示使用主模型配置的 API_key
    )   
resp = client.chat.create(prompt="2+2等于几？")  # 调用入口如再次配置模型，将会覆盖客户端入口处配置的所有模型
print(resp)
```

```mermaid
flowchart TD
    A[chat.create 调用入口] --> B{model 指定?}
    B -->|是| C[调用 adapter]
    C -->|成功| J[调用入口模型成功]
    C -->|失败| K[ModelNotSupportedError]
    B -->|否| D[调用 FallbackManager]
    D --> E{主模型可用?}
    E -->|是| F[主模型成功]
    E -->|否| G{按顺序尝试 fallback_models}
    G -->|全部失败| H[FallbackError]
    G -->|任一成功| I[该模型成功]
````
***

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
