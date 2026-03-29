# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

!\[PyPI Version]\(<https://img.shields.io/pypi/v/cnllm> null)
!\[Python Versions]\(<https://img.shields.io/pypi/pyversions/cnllm> null)
!\[License]\(<https://img.shields.io/github/license/kanchengw/cnllm> null)

***

Adapter Library for Chinese Large Language Models (LLMs), format API responds in OpenAI format, seamless integration with LangChain, LlamaIndex, Pydantic and other major ML frameworks.

## Changelog

### v0.4.0 (Planned)

- 🔧 Model adapter development (e.g., Doubao, Kimi, etc.)
- 🔧 Framework adapter validation and deep integration (LlamaIndex, Pydantic, LiteLLM, Instructor)

### v0.3.1 (2026-03-29) ✨

- ✨ **Deep LangChain Integration**
  - Runnable adapter as core feature, one function to integrate with LangChain chain
  - Runnable streaming output, batch calls, async calls support
- ✨ **chat.create() Streaming Output** - `stream=True` parameter support
- ✨ **Fallback Mechanism** - Automatic switch to backup model when primary fails
- ✨ **Attribute** - `client.chat.still` get clean chat response, `client.chat.raw` get full response
- 🔧 **Adapter Refactoring** - Dual-layer architecture with model adapters (Chinese LLMs like MiniMax) + framework adapters (LangChain, etc.)

### v0.3.0 (2026-03-28)

- ✨ __call__ Ultra Simple Calling, prompt parameter, model override mechanism


## Features

- **OpenAI Compatible** - All outputs fully align with OpenAI API standard format
- **Framework Integration** - Compatible with LangChain, LlamaIndex and other major ML frameworks
- **Unified Interface** - One codebase, seamless switching between different LLMs
- **Simple API** - Multiple calling styles, as simple as one line of code

## Supported Models

- **Verified**: MiniMax-M2.7, MiniMax-M2.5, MiniMax-M2.1, MiniMax-M2
- **More models and providers in development**

## Installation

```bash
pip install cnllm
```

## Quick Start

### Initialization Interface

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
```

### Three Calling Styles

**1. Simple Call** **`client("prompt")`**

```python
resp = client("Introduce yourself in one sentence")  # Simple call does not accept other parameters
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

### Quick Model Switching at Call

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5",  # Optional, override model
    api_key="your_other_api_key"  # Optional, override API Key
)
```

### Response Entry

**1. Get Clean Chat Response**

```python
# Traditional way
print(resp["choices"][0]["message"]["content"])

# Using still attribute (recommended)
print(client.chat.still)
```
**2. Get Full Response**

```python
print(client.chat.raw)  # Raw response from the model, including all details
```

## Unified Interface Parameters

### CNLLM Client Interface

| Parameter         | Type  | Required | Default     | Description                                                                                                           |
| ----------------- | ----- | -------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `model`           | str   | ✅        | -           | Model name: minimax-m2.7, minimax-m2.5                                                                                |
| `api_key`         | str   | ✅        | -           | API key                                                                                                               |
| `base_url`        | str   | -        | API default | Custom API address                                                                                                    |
| `timeout`         | int   | -        | 30          | Request timeout in seconds                                                                                            |
| `max_retries`     | int   | -        | 3           | Maximum retry attempts                                                                                                |
| `retry_delay`     | float | -        | 1.0         | Retry delay in seconds                                                                                                |
| `fallback_models` | dict  | -        | {}          | Fallback model config, format: `{"fallback_model": "api_key", ...}`, api\_key None means sharing API key with primary |

### Two Calling Interfaces

#### Simple Call client()

Directly pass prompt string, no extra parameters.

#### client.chat.create() Parameters

| Parameter     | Type        | Required | Default | Description                                                 |
| ------------- | ----------- | -------- | ------- | ----------------------------------------------------------- |
| `messages`    | list\[dict] | ⚠️       | -       | OpenAI format message list (mutually exclusive with prompt) |
| `prompt`      | str         | ⚠️       | -       | Short form (mutually exclusive with messages)               |
| `model`       | str         | -        | None    | Override default model                                      |
| `api_key`     | str         | -        | None    | Override default API Key                                    |
| `temperature` | float       | -        | 0.7     | Generation randomness, 0-2                                  |
| `max_tokens`  | int         | -        | None    | Maximum generation tokens                                   |
| `stream`      | bool        | -        | False   | Streaming response                                          |

## Response Format

Through any API call style in the quick start, the response will be transformed into OpenAI standard format:

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

Compatible with LangChain library (with Runnable integration), other libraries like Pydantic, LlamaIndex, and Instructor can also use the transformed response directly (not verified).

## LangChainRunnable Implementation

```python
from cnllm import CNLLM
from cnllm.adapters.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")

# Wrap client with LangChainRunnable
runnable = LangChainRunnable(client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),("human", "{input}")])

# Build LangChain chain
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

