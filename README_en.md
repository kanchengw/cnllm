# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

[![PyPI Version](https://img.shields.io/pypi/v/cnllm)](https://pypi.org/project/cnllm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/cnllm)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm)](LICENSE)

***

## Project Background

Integrating Chinese large language models into established ML frameworks has become a core challenge for developers in both academic research and production environments.

Current mainstream approaches have clear limitations: using OpenAI-compatible interfaces is straightforward but cannot fully leverage each vendor's native capabilities; calling native interfaces directly means handling response parsing, format conversion, and other tedious work on your own.

CNLLM is dedicated to resolving this dilemma—by providing a **unified interface** and **consistent parameter specifications** for calling Chinese LLMs. While fully unleashing the native capabilities of these models, CNLLM automatically converts diverse responses into OpenAI standard format. Whether it's LangChain, LlamaIndex or other ML frameworks, you can integrate various LLMs in the same way; additionally, in scenarios requiring multi-model collaboration, you can maintain consistent interfaces, parameters, and response formats.

> Due to limited time and energy, we welcome like-minded friends to join us in building CNLLM: [wangkancheng1122@163.com](mailto:wangkancheng1122@163.com)

### Collaboration Opportunities

| Area | Description |
|------|-------------|
| 🌐 **New Vendor Adapters** | Integrate more Chinese LLMs (Alibaba Qwen, ByteDance Doubao, Kimi, etc.) |
| 🔗 **Framework Integration** | Deepen integration with LlamaIndex, LiteLLM, and other frameworks |
| 🐛 **Capability Expansion** | Adapter development for Embedding, Multimodal, and other features |
| 📖 **Documentation** | Add use cases and improve development guides |
| 💡 **Feature Suggestions** | Share your ideas and requirements |

Quick start: [Contributor Guide](docs/CONTRIBUTOR_en.md)
Detailed architecture: [System Architecture](docs/ARCHITECTURE_en.md)

## Changelog

### v0.6.0 (2026-04-08)

- ✨ **Async Support** - Full async support via `AsyncCNLLM` client with `await client.chat.create()` and `await client.embedding.create()`
  - httpx unified sync/async HTTP client with SSE streaming support
  - Streaming returns `AsyncIterator[dict]`, non-streaming returns `dict`
- ✨ **Batch Calls** - `client.chat.batch()` supports batch concurrent calls, returns `BatchResponse`, supports streaming/non-streaming, real-time statistics
  - Real-time stats: `request_counts` field shows real-time request status
  - Error isolation: single request failure doesn't affect other requests
  - Progress callbacks: `callbacks` custom callback functions
- ✨ **Embedding Support** - `client.embedding.create()` supports single/batch Embedding, returns `EmbeddingResponse`
  - Sync/async interfaces: `create()` / `acreate()`
  - Custom IDs: supports `custom_ids` parameter
  - OpenAI compatible: returns standard OpenAI embedding format

### v0.5.0 (2026-04-06)

- ✨ **KIMI (Moonshot AI) Adapter** - Kimi model adapter, supports kimi-k2.5, kimi-k2 series and moonshot-v1 series (8k/32k/128k)
- ✨ **DeepSeek Adapter** - DeepSeek model adapter, supports `deepseek-chat` and `deepseek-reasoner` models
- Now CNLLM includes `system_fingerprint` and `choices[0].logprobs` fields in standard response

### v0.4.3 (2026-04-06)

- ✨ **Doubao Adapter** - ByteDance Doubao Seed series model adapter, supports seed-2.0 series, seed-1.6 series and seed-1.8, totaling 8 models (see `Supported Models` for details), with support for Doubao native parameters like `stream_options`, `reasoning_effort`, `service_tier`, etc.
  - Supports `reasoning_effort` inference length field with four-level switching: `minimal`, `low`, `medium`, `high`
  - Supports `thinking` field with three-level switching: `true` (enabled), `false` (disabled), `auto`; `thinking="auto"` only takes effect on doubao-seed-1-6 model
- 🔧 **Bug Fix** - Fixed streaming response `_collect_stream_result` duplicate call causing content accumulation anomaly

### v0.4.2 (2026-04-05)

- ✨ **GLM Adapter** - Zhipu GLM model adapter, supports "glm-4.6", "glm-5", "glm-5-turbo" and GLM 4.7 series
  - Supports GLM native parameters: `do_sample`, `request_id`, `response_format`, `tool_stream`, `thinking`, etc.
- 🔧 **Bug Fix** - Fixed `id` field response mapping

### v0.4.1 (2026-04-04)

- 🔧 **Bug fixes**

### v0.4.0 (2026-04-03)

- ✨ **mimo Adapter** - Xiaomi mimo model adapter, supports "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"
- ✨ **Architecture Refactoring** - BaseAdapter + Responder + VendorError three-layer architecture separation, clear responsibilities
- ✨ **.think Property** - `client.chat.think` retrieves reasoning_content, not included in resp
- ✨ **.tools Property** - `client.chat.tools` retrieves tool_calls, supports streaming accumulation
- ✨ **Streaming Accumulation** - `.think`, `.still`, `.tools` support real-time accumulation during streaming response

## Features

- **OpenAI Compatible** - Model output aligned with OpenAI API standard format
- **Framework Integration** - Compatible with LangChain, LlamaIndex and other major ML frameworks
- **Unified Interface** - One codebase, seamless switching between different Chinese LLMs
- **Simple API** - Multiple calling styles, simplest only needs one line of code

## Supported Models

- **DeepSeek**: deepseek-chat, deepseek-reasoner
- **KIMI (Moonshot AI)**: kimi-k2.5, kimi-k2-thinking, kimi-k2-thinking-turbo, kimi-k2-turbo-preview, kimi-k2-0905-preview, moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
- **Doubao**: doubao-seed-2-0-pro, doubao-seed-2-0-mini, doubao-seed-2-0-lite, doubao-seed-2-0-code, doubao-seed-1-8, doubao-seed-1-6, doubao-seed-1-6-lite, doubao-seed-1-6-flash
- **Zhipu GLM**: glm-4.6, glm-4.7, glm-4.7-flash, glm-4.7-flashx, glm-5, glm-5-turbo
- **Xiaomi MiMo**: mimo-v2-pro, mimo-v2-omni, mimo-v2-flash
- **MiniMax**: MiniMax-M2.7, MiniMax-M2.5, MiniMax-M2.1, MiniMax-M2
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

OpenAI standard response structure compatible with LangChain library (deep integration with Runnable component), other libraries like Pydantic, LlamaIndex, Instructor that support OpenAI standard structure should work directly (not verified).

### Streaming Response Format

When streaming is enabled, the response is a `StreamResultAccumulator` iterable object:

```python
response = client.chat.create(messages=[...], stream=True)
```

Each chunk follows OpenAI standard format:

```python
# print(response) output:
{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}

{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'content': 'Hi'}, 'finish_reason': None}]}

 # ... middle chunks

{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}

'[DONE]'
```

### Accessing Properties in Streaming Response

```python
# print(client.chat.still) output:
"Hello, I am MiniMax-M-2.7..."

# print(client.chat.tools) output:
[{'id': 'call_xxx', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"Tokyo"}'}}]
 # ... more chunks

# print(client.chat.think) output:
"The user is asking me to greet them..."

# print(client.chat.raw) output (keeps vendor-specific fields like reasoning_content):
[{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'reasoning_content': 'The user'}, 'finish_reason': None}]},
 # ... more chunks
```

## Unified Interface Parameters

| Parameter            | Type          | Required | Default                      | Client Init | Call Entry | Description                                           |
| ------------------- | ------------- | -------- | ---------------------------- | :---------: | :--------: | ----------------------------------------------------- |
| `model`             | str           | ✅        | -                            |     ✅      |     ✅     | Required at client initialization                      |
| `api_key`           | str           | ✅        | -                            |     ✅      |     ✅     | API key                                               |
| `messages`          | list\[dict]   | ⚠️       | -                            |     ❌      |     ✅     | OpenAI format message list (mutually exclusive with prompt) |
| `prompt`            | str           | ⚠️       | -                            |     ❌      |     ✅     | Short form (mutually exclusive with messages)          |
| `fallback_models`   | dict          | -        | -                            |     ✅      |     ❌     | Backup model configuration (see FallbackManager design) |
| `base_url`          | str           | -        | Auto-adapted model default   |     ✅      |     ✅     | Custom API address                                    |
| `timeout`           | int           | -        | 60                           |     ✅      |     ✅     | Request timeout (seconds)                             |
| `max_retries`       | int           | -        | 3                            |     ✅      |     ✅     | Maximum retry count                                  |
| `retry_delay`       | float         | -        | 1.0                          |     ✅      |     ✅     | Retry delay (seconds)                                |
| `temperature`       | float         | -        | Vendor default               |     ✅      |     ✅     | Generation randomness                                 |
| `max_tokens`        | int           | -        | Vendor default               |     ✅      |     ✅     | Maximum generation token count                        |
| `stream`            | bool          | -        | Vendor default, usually False|     ✅      |     ✅     | Streaming response                                   |
| `top_p`             | float         | -        | Vendor default               |     ✅      |     ✅     | Nucleus sampling threshold                           |
| `tools`             | list          | -        | -                            |     ✅      |     ✅     | Function tools definition                             |
| `tool_choice`       | str           | -        | -                            |     ✅      |     ✅     | Tool choice mode: none / auto                         |
| `thinking`          | bool          | -        | Vendor default               |     ✅      |     ✅     | Thinking mode, unified format as `thinking=True/False` |
| `presence_penalty`  | float         | -        | Vendor default               |     ✅      |     ✅     | Presence penalty                                     |
| `frequency_penalty` | float         | -        | Vendor default               |     ✅      |     ✅     | Frequency penalty                                    |
| `organization`      | str           | -        | -                            |     ✅      |     ✅     | Organization identifier                              |
| `stop`              | str/list      | -        | -                            |     ✅      |     ✅     | Stop sequences                                       |
| `user`              | str           | -        | -                            |     ✅      |     ✅     | User identifier                                      |
| `response_format`   | dict          | -        | Vendor default, usually {type:"text"} |     ✅      |     ✅     | Response format                                      |

#### Simple Call client()

Pass the prompt string directly, no additional parameters.

**Notes**:

- Not all supported models support all CNLLM standard request parameters. For specific support and other parameters, please refer to the vendor's official documentation.

- For more parameters supported by specific models, please refer to the official documentation. CNLLM will pass through all parameters supported by the specific model.
- Call entry parameters take priority. It is recommended to pass commonly used parameters at the client level, and flexibly override or pass more parameters at individual calls.
- Required parameters only need to be ensured not empty when the request is initiated, i.e., the client and call entry should have at least one provide the parameter.

## FallbackManager Model Selection Flow

With the `fallback_models` parameter at client initialization entry, if  the primary model specified in `model` is unavailable, it will try models in `fallback_models` in order.
Configure this option for program or application runtime stability.

```python
client = CNLLM(
    model="minimax-m2.7", api_key="minimax_key",
    fallback_models={"mimo-v2-flash": "xiaomi-key", "minimax-m2.5": None}  # Apply None to use the API_key configured for the primary model
    )
resp = client.chat.create(prompt="What is 2+2?")
print(resp)
```

```mermaid
flowchart TD
    A[chat.create Call Entry] --> B{model specified?}
    B -->|Yes| C[Call adapter]
    C -->|Success| J[Entry model success]
    C -->|Failure| K[ModelNotSupportedError]
    B -->|No| D[Call FallbackManager]
    D --> E{Primary model available?}
    E -->|Yes| F[Primary model success]
    E -->|No| G{Try fallback_models in order}
    G -->|All fail| H[FallbackError]
    G -->|Any succeeds| I[That model succeeds]
```

***

### Quick Model Switching at Call:

Note: If model is overridden at the call entry, the FallbackManager flow will not be entered.

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5",  # Override models in client initialization
    api_key="your_other_api_key"  # Override, or leave api_key out to use the one in client initialization
)
```
***

## Batch Calls

Supports sync/async, streaming/non-streaming batch calls with various advanced parameters.

```python
results = client.chat.batch(
    ["Hello", "How's the weather?", "Who are you?"],
    stream=True,
    max_concurrent=3,       # max concurrent requests
    timeout=60,               # per-request timeout (seconds)
    stop_on_error=False,      # stop on error
    callbacks=None            # progress callback functions
)
for chunk in results:
    print(chunk)  # streaming chunk
print(results.still)               # batch response content {"request_0": "...", ...}
print(results.tools["request_0"])  # access single result's tool calls
print(result[0])                   # or print(result["request_0"]) single response
print(result["doc_001"])           # access by custom_ids
```

#### to_dict():

```python
result.to_dict()                        # results only (default)
result.to_dict(stats=True)              # results + stats (request_counts, elapsed)
result.to_dict(stats=True, think=True, still=True, tools=True, raw=True)  # results + any fields
```

### Embedding Calls

#### Batch Embedding

```python
result = client.embedding.create(["Hello", "world", "你好"])
```

#### Standard Embedding Response Structure

```python
{
    "request_counts": {
        "total": 2,
        "dimension": 1024
    },
    "elapsed": 0.35,
    "results": {
        "request_0": {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
            "model": "embedding-2",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
    }
}
```

#### Response Access

```python
print(result[0])  # or print(result["request_0"]) single response, standard OpenAI Embedding response
print(result["doc_001"])                 # access by custom_ids
print(result.elapsed)                    # request time
print(result.request_counts)             # statistics
```

#### to_dict():
```python
result.to_dict()                        # results only (default)
result.to_dict(stats=True)              # results + stats (request_counts, elapsed)
result.to_dict(results=True, stats=True, usage=True)  # all information
```

### Quick Response Properties

Quick access properties for specific fields, optimizing complex access like `response.choices[0].message.content`.

| Property | Description | Example |
|----------|-------------|---------|
| `client.chat.still` | Clean response | "Hello, I am MiniMax-M2.7..." |
| `client.chat.tools` | Tool calls | [{'id': '...', 'type': 'function', 'function': {...}}] |
| `client.chat.think` | Reasoning content | "Let me think about this..." |
| `client.chat.raw` | Vendor raw response | {'id': '...', 'choices': [{'message': {'reasoning_content': '...'}}} |

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

# Synchronous streaming output
for chunk in runnable.stream("Count to 5"):
    print(chunk, end="", flush=True)

# Asynchronous streaming output
async def async_stream_test():
    async for chunk in runnable.astream("Count to 3"):
        print(chunk, end="", flush=True)

asyncio.run(async_stream_test())

# Batch calls
results = runnable.batch(["Hello", "How are you?"])
for r in results:
    print(r.content)
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
