# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

[![PyPI Version](https://img.shields.io/pypi/v/cnllm)](https://pypi.org/project/cnllm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/cnllm)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm)](LICENSE)

***

Adapter Library for Chinese Large Language Models (LLMs), converts model API responses to OpenAI format, seamlessly integrates with LangChain, LlamaIndex, Pydantic and other major ML frameworks.

## Contributors

CNLLM is actively under development, contributors are welcome!

We're looking for help with:
- 🌐 **New Vendor Adapters** - Develop adapters for more LLM providers (GLM, doubao, Kimi, etc.)
- 🔗 **ML Framework Deep Adaptation** - Adaptation verification for LlamaIndex, LiteLLM, etc.
- 🐛 **Capability Expansion** - Develop adapters for Embedding, Multimodal
- 📖 **Documentation** - Improve docs and add examples
- 💡 **Feature Suggestions** - Propose new feature ideas

Quick start: [Contributor Guide](docs/CONTRIBUTOR_en.md)
Detailed architecture: [System Architecture](docs/ARCHITECTURE_en.md)

## Changelog

### v0.4.0 (2026-04-03)

- ✨ **mimo Adapter** - Xiaomi mimo model adapter, supports "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"
- ✨ **Architecture Refactoring** - BaseAdapter + Responder + VendorError three-layer architecture separation, clear responsibilities
- ✨ **.think Property** - `client.chat.think` retrieves reasoning_content, not included in resp
- ✨ **.tools Property** - `client.chat.tools` retrieves tool_calls, supports streaming accumulation
- ✨ **Streaming Accumulation** - `.think`, `.still`, `.tools` support real-time accumulation during streaming response

### v0.3.3 (2026-04-02) ✨

- ✨ **Unified Parameters** - Client init parameters unified with call entry parameters, call entry flexibly overrides
- ✨ **Architecture Optimization** - Core logic abstraction, BaseAdapter and Responder handle common logic
- ✨ **Extensibility** - Adding new provider only requires configuring corresponding YAML file, automatically implements request and response field mapping, error code mapping, no need to modify other upper-level components
- ✨ **YAML Function Integration** - Related field mapping, model support validation, required param validation, param support validation, vendor error code mapping logic
- ✨ **MiniMax Support Optimization** - Supports all MiniMax native interface parameters, such as `top_p`, `tools`, `thinking`, etc.

### v0.3.1 (2026-03-29) ✨

- ✨ **Deep LangChain Integration**
  - Runnable adapter as core feature, one function integrates with LangChain chain
  - Runnable streaming output, batch calls, async calls support
- ✨ **chat.create() Streaming Output** - `stream=True` parameter support
- ✨ **Fallback Mechanism** - Automatically switch to backup model when primary fails
- ✨ **Response Entry** - `client.chat.still` easily gets clean chat response, `client.chat.raw` gets full response
- 🔧 **Adapter Refactoring** - Model adapters (Chinese LLMs like MiniMax) + framework adapters (LangChain, etc.) dual-layer architecture

## Features

- **OpenAI Compatible** - Model output aligned with OpenAI API standard format
- **Framework Integration** - Compatible with LangChain, LlamaIndex and other major ML frameworks
- **Unified Interface** - One codebase, seamless switching between different Chinese LLMs
- **Simple API** - Multiple calling styles, simplest only needs one line of code

## Supported Models

- **Xiaomi mimo**: mimo-v2-pro, mimo-v2-flash, mimo-v2-omni
- **MiniMax**: MiniMax-M2.7、MiniMax-M2.5、MiniMax-M2.1、MiniMax-M2
- **More providers and models in development**

## Installation

```bash
pip install cnllm
```

## Quick Start

### Initialization Interface

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2.7", api_key="your_api_key")
```

### Three Entry Points

**1. Simple Call** **`client("prompt")`**

```python
resp = client("Introduce yourself in one sentence")
```

**2. Standard Call** **`client.chat.create(prompt="prompt")`**

```python
resp = client.chat.create(prompt="Introduce yourself in one sentence")
```

**3. Full Call** **`client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Introduce yourself in one sentence"}
    ]
)
```

### Quick Model Switching at Call:

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5",
    api_key="your_other_api_key"
)
```

### Response Entry

**1. Get Clean Chat Response**

```python
print(resp["choices"][0]["message"]["content"])

print(client.chat.still)
```

**2. Get Full Response**

```python
print(client.chat.raw)
```

**3. Get Model Thinking Process (reasoning_content)**

```python
resp = client.chat.create(
    messages=[{"role": "user", "content": "Explain why sky is blue"}],
    thinking=True
)
print(client.chat.think)
```

**4. Get Tool Calls (tool_calls)**

```python
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]
resp = client.chat.create(
    messages=[{"role": "user", "content": "How's the weather in Beijing?"}],
    tools=tools
)
print(client.chat.tools)
```

## Unified Interface Parameters

| Parameter            | Type          | Required | Default      | Client Init | Call Entry | Description                                           |
| ------------------- | ------------- | -------- | ------------ | :---------: | :--------: | ----------------------------------------------------- |
| `model`             | str           | ✅        | -            |     ✅      |     ✅     | Such as minimax-m2.7 or MiniMax-m2.7                 |
| `api_key`           | str           | ✅        | -            |     ✅      |     ✅     | API key                                               |
| `messages`          | list\[dict]   | ⚠️       | -            |     ❌      |     ✅     | OpenAI format message list (mutually exclusive with prompt) |
| `prompt`            | str           | ⚠️       | -            |     ❌      |     ✅     | Short form (mutually exclusive with messages)          |
| `fallback_models`   | dict          | -        | {}           |     ✅      |     ❌     | Backup model configuration                            |
| `base_url`          | str           | -        | API default  |     ✅      |     ✅     | Custom API address                                    |
| `timeout`           | int           | -        | 60           |     ✅      |     ✅     | Request timeout (seconds)                            |
| `max_retries`       | int           | -        | 3            |     ✅      |     ✅     | Maximum retry count                                   |
| `retry_delay`       | float         | -        | 1.0          |     ✅      |     ✅     | Retry delay (seconds)                                 |
| `temperature`       | float         | -        | 0.7          |     ✅      |     ✅     | Generation randomness, 0-2                            |
| `max_tokens`        | int           | -        | None         |     ✅      |     ✅     | Maximum generation token count                        |
| `stream`            | bool          | -        | False        |     ✅      |     ✅     | Streaming response                                    |
| `top_p`             | float         | -        | 0.95         |     ✅      |     ✅     | Nucleus sampling threshold                            |
| `top_k`             | int           | -        | -            |     ✅      |     ✅     | Top-K sampling                                        |
| `tools`             | list          | -        | -            |     ✅      |     ✅     | Function tools definition                             |
| `tool_choice`       | str           | -        | -            |     ✅      |     ✅     | Tool choice mode: none / auto                         |
| `thinking`          | bool          | -        | -            |     ✅      |     ✅     | Thinking mode (MiniMax-M1)                           |
| `presence_penalty`  | float         | -        | -            |     ✅      |     ✅     | Presence penalty                                      |
| `frequency_penalty` | float         | -        | -            |     ✅      |     ✅     | Frequency penalty                                     |
| `stop`              | str/list      | -        | -            |     ✅      |     ✅     | Stop sequences                                        |
| `user`              | str           | -        | -            |     ✅      |     ✅     | User identifier                                       |
| `organization`      | str           | -        | -            |     ✅      |     ✅     | When using MiniMax, automatically maps to MiniMax standard field group_id |

## Response Format

Through any API call style in the quick start, the model's response will be converted to OpenAI standard format:

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
        "total_tokens": 30,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    }
}
```

OpenAI standard response structure compatible with LangChain library (deep integration with Runnable component), other libraries like Pydantic, LlamaIndex, Instructor that support OpenAI standard structure should work directly (not verified).

## LangChainRunnable Implementation

```python
from cnllm import CNLLM
from cnllm.core.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")

runnable = LangChainRunnable(client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),("human", "{input}")])

chain = prompt | runnable
result = chain.invoke({"input": "What is 2+2?"})
print(result.content)

for chunk in runnable.stream("Count to 5"):
    print(chunk, end="", flush=True)

async def async_stream_test():
    async for chunk in runnable.astream("Count to 3"):
        print(chunk, end="", flush=True)

asyncio.run(async_stream_test())

results = runnable.batch(["Hello", "How are you?"])
for r in results:
    print(r.content)
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
