# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

<<<<<<< HEAD
[![PyPI Version](https://img.shields.io/pypi/v/cnllm)](https://pypi.org/project/cnllm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/cnllm)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm)](LICENSE)

***

Adapter Library for Chinese Large Language Models (LLMs), converts model API responses to OpenAI format, seamlessly integrates with LangChain, LlamaIndex, Pydantic and other major ML frameworks.
=======
!\[PyPI Version]\(<https://img.shields.io/pypi/v/cnllm> null)
!\[Python Versions]\(<https://img.shields.io/pypi/pyversions/cnllm> null)
!\[License]\(<https://img.shields.io/github/license/kanchengw/cnllm> null)

***

Adapter Library for Chinese Large Language Models (LLMs), format API responds in OpenAI format, seamless integration with LangChain, LlamaIndex, Pydantic and other major ML frameworks.
>>>>>>> origin/main

## Changelog

### v0.3.2 (2026-04-01) ✨

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
<<<<<<< HEAD
- ✨ **Fallback Mechanism** - Automatically switch to backup model when primary fails
- ✨ **Response Entry** - `client.chat.still` easily gets clean chat response, `client.chat.raw` gets full response
- 🔧 **Adapter Refactoring** - Model adapters (Chinese LLMs like MiniMax) + framework adapters (LangChain, etc.) dual-layer architecture
=======
- ✨ **Fallback Mechanism** - Automatic switch to backup model when primary fails
- ✨ **Attribute** - `client.chat.still` get clean chat response, `client.chat.raw` get full response
- 🔧 **Adapter Refactoring** - Dual-layer architecture with model adapters (Chinese LLMs like MiniMax) + framework adapters (LangChain, etc.)

### v0.3.0 (2026-03-28)

- ✨ __call__ Ultra Simple Calling, prompt parameter, model override mechanism

>>>>>>> origin/main

## Features

- **OpenAI Compatible** - Model output aligned with OpenAI API standard format
- **Framework Integration** - Compatible with LangChain, LlamaIndex and other major ML frameworks
- **Unified Interface** - One codebase, seamless switching between different Chinese LLMs
- **Simple API** - Multiple calling styles, simplest only needs one line of code

## Supported Models

- **Verified**: MiniMax-M2.7、MiniMax-M2.5、MiniMax-M2.1、MiniMax-M2
- **More providers and models in development**

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

### Three Entry Points

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

<<<<<<< HEAD
### Quick Model Switching at Call:
=======
### Quick Model Switching at Call
>>>>>>> origin/main

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5",  # Optional, override model
<<<<<<< HEAD
    api_key="your_other_api_key"  # Optional, override API Key, if not filled, defaults to key at client entry
=======
    api_key="your_other_api_key"  # Optional, override API Key
>>>>>>> origin/main
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
<<<<<<< HEAD

**2. Get Full Response**

```python
print(client.chat.raw)  # Raw response from the model
=======
**2. Get Full Response**

```python
print(client.chat.raw)  # Raw response from the model, including all details
>>>>>>> origin/main
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

<<<<<<< HEAD
**Notes**:
=======
| Parameter         | Type  | Required | Default     | Description                                                                                                           |
| ----------------- | ----- | -------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `model`           | str   | ✅        | -           | Model name: minimax-m2.7, minimax-m2.5                                                                                |
| `api_key`         | str   | ✅        | -           | API key                                                                                                               |
| `base_url`        | str   | -        | API default | Custom API address                                                                                                    |
| `timeout`         | int   | -        | 30          | Request timeout in seconds                                                                                            |
| `max_retries`     | int   | -        | 3           | Maximum retry attempts                                                                                                |
| `retry_delay`     | float | -        | 1.0         | Retry delay in seconds                                                                                                |
| `fallback_models` | dict  | -        | {}          | Fallback model config, format: `{"fallback_model": "api_key", ...}`, api\_key None means sharing API key with primary |
>>>>>>> origin/main

- Call entry parameters take priority, recommend passing common parameters via client init, can flexibly override and pass more parameters at single call.
- Required parameters only need to ensure not empty when making request, i.e., either client init or call entry should have the parameter.

#### Simple Call client()

Directly pass prompt string, no extra parameters.

<<<<<<< HEAD
## Response Format

Through any API call style in the quick start, the model's response will be converted to OpenAI standard format:
=======
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
>>>>>>> origin/main

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

<<<<<<< HEAD
OpenAI standard response structure compatible with LangChain library (deep integration with Runnable component), other libraries like Pydantic, LlamaIndex, Instructor that support OpenAI standard structure should work directly (not verified).
=======
Compatible with LangChain library (with Runnable integration), other libraries like Pydantic, LlamaIndex, and Instructor can also use the transformed response directly (not verified).
>>>>>>> origin/main

## LangChainRunnable Implementation

```python
from cnllm import CNLLM
from cnllm.core.framework import LangChainRunnable
from langchain_core.prompts import ChatPromptTemplate
import asyncio

client = CNLLM(model="minimax-m2.7", api_key="your_key")

# Wrap client with LangChainRunnable
runnable = LangChainRunnable(client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),("human", "{input}")])

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

<<<<<<< HEAD
## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
=======
>>>>>>> origin/main
