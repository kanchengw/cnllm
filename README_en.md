# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

![PyPI Version](https://img.shields.io/pypi/v/cnllm null)
![Python Versions](https://img.shields.io/pypi/pyversions/cnllm null)
![License](https://img.shields.io/github/license/kanchengw/cnllm null)

***

A unified adapter library for Chinese large language models (LLMs). It converts API outputs from various Chinese LLMs into the unified OpenAI format, enabling zero-cost integration with LangChain and other mainstream AI frameworks.

## Changelog

### v0.3.0 (2026-03-28) ✨

- ✨ **Deep LangChain Integration**
  - Runnable adapter as core feature, one function to integrate with LangChain chain
  - Runnable streaming output, batch calls, async calls support
- ✨ **chat.create() Streaming Output** - `stream=True` parameter support
- ✨ **extra\_config** - Unified management of provider-specific parameters
- 🔧 **Adapter Refactoring** - Dual-layer architecture with model adapters (Chinese LLMs like Minimax) + framework adapters (LangChain, etc.)

### v0.2.0 (2026-03-27)

- ✨ __call__ Ultra Simple Calling, prompt parameter, model override mechanism

## Features

- **OpenAI Compatible** - All outputs fully align with OpenAI API standard format
- **Framework Integration** - Compatible with LangChain, LlamaIndex and other major ML frameworks
- **Unified Interface** - One codebase, seamless switching between different LLMs
- **Simple API** - Multiple calling styles, as simple as one line of code

## Supported Models

- **Verified**: MiniMax-M2.7, MiniMax-M2.5
- **More models and providers in development**

## Installation

```bash
pip install cnllm
```

## Quick Start

### Three Calling Styles

**1. Ultra Simple** **`client("prompt")`**

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
resp = client("Introduce yourself in one sentence")
print(resp["choices"][0]["message"]["content"])
```

**2. Standard** **`client.chat.create(prompt="prompt")`**

```python
resp = client.chat.create(prompt="Introduce yourself in one sentence")
print(resp["choices"][0]["message"]["content"])
```

**3. Full** **`client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Introduce yourself in one sentence"}
    ]
)
print(resp["choices"][0]["message"]["content"])
```

### Model Override

Support overriding default model at call time:

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5"
)
```

## LangChainRunnable Implementation

```python
from cnllm import CNLLM
from cnllm.adapters.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")
runnable = LangChainRunnable(client)

# Template with input variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
chain = prompt | runnable
result = chain.invoke({"input": "What is 2+2?"})
print(result.content)

# Sync streaming output
for chunk in runnable.stream("Count to 5"):
    print(chunk, end="", flush=True)

# Async streaming output
async def async_stream_test():
    async for chunk in runnable.astream("Count to 3"):
        print(chunk, end="", flush=True)

asyncio.run(async_stream_test())

# Batch calls
results = runnable.batch(["Hello", "How are you?"])
for r in results:
    print(r.content)
```

Additionally, using any of the quick start calling methods, the output is compatible with LangChain components (HumanMessage, AIMessage, SystemMessage, etc.), prompt templates (ChatPromptTemplate), output parsers (StrOutputParser), etc.

## Unified Interface Parameters

### CNLLM Client Interface

| Parameter    | Type   | Required | Default      | Description                                    |
|-------------|--------|----------|--------------|------------------------------------------------|
| `model`     | str    | ✅        | -            | Model name: minimax-m2.7, minimax-m2.5        |
| `api_key`   | str    | ✅        | -            | API key                                        |
| `base_url`  | str    | -        | API default  | Custom API address                             |
| `organization` | str | -        | None         | OpenAI organization identifier                 |
| `extra_config` | dict | -        | {}           | Provider-specific parameters (see below)        |
| `timeout`   | int    | -        | 30           | Request timeout in seconds                      |
| `max_retries` | int  | -        | 3            | Maximum retry attempts                         |
| `retry_delay` | float | -        | 1.0          | Retry delay in seconds                          |

#### extra\_config Parameters

| Provider | Parameters              | Description                   |
|----------|------------------------|-------------------------------|
| MiniMax  | `group_id`             | Multi-user/billing GroupId    |
| Doubao   | `app_id`, `space_id`   | App/Space identifier          |
| Kimi     | `project_id`, `api_version` | Project/version identifier |

### chat.create() Interface

| Parameter         | Type          | Required | Default   | Description                                      |
|-----------------|---------------|----------|-----------|------------------------------------------------|
| `messages`       | list\[dict]   | ⚠️        | -         | OpenAI format message list (mutually exclusive with prompt) |
| `prompt`         | str           | ⚠️        | -         | Short form, auto-converted to messages (mutually exclusive with messages) |
| `model`          | str           | -        | None      | Override default model                          |
| `temperature`    | float         | -        | 0.7       | Generation randomness, 0-2                       |
| `max_tokens`     | int           | -        | None      | Maximum generation tokens                       |
| `top_p`          | float         | -        | 1.0       | Nucleus sampling parameter, 0-1                 |
| `n`              | int           | -        | 1         | Number of candidates to generate                |
| `stream`         | bool          | -        | False     | Streaming response                              |
| `stop`           | str/list      | -        | None      | Stop words                                      |
| `presence_penalty` | float       | -        | 0.0       | Presence penalty, -2 to 2                       |
| `frequency_penalty` | float     | -        | 0.0       | Frequency penalty, -2 to 2                      |
| `user`           | str           | -        | None      | User identifier                                 |
| `extra_config`   | dict          | -        | None      | Provider-specific parameters                    |

## Response Format

All API responses follow OpenAI standard format:

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
            "content": "I am MiniMax-M2.7..."
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

## Other Compatible Frameworks

- **LlamaIndex** - Indexing and querying (planned)
- **AutoGen** - Multi-agent collaboration (planned)
- **CrewAI** - Multi-agent workflows
