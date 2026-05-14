# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

[![PyPI Version](https://img.shields.io/pypi/v/cnllm)](https://pypi.org/project/cnllm/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776A4?style=flat)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm)](https://github.com/kanchengw/cnllm/blob/main/LICENSE)

***

## Why CNLLM?

Chinese LLMs have reached the top tier in capabilities, yet in real production environments they face a lack of infrastructure. An unavoidable **dilemma** is:

When using OpenAI SDK/LiteLLM with vendor-provided compatible interfaces, **unsupported native parameters are silently ignored**, leading to **uncontrollable results and missing features**; using vendor proprietary SDKs requires **extra field parsing and structure transformation**. When workflows involve multiple models from different vendors, different code adaptations are needed for each model, resulting in **increased engineering workload and maintenance costs**.

CNLLM provides a **unified OpenAI-compatible interface layer** and a set of **standardized parameter rules and response format specifications**. CNLLM achieves **bidirectional mapping** of requests and responses through standardized YAML configuration files tailored for each vendor, mapping CNLLM standard parameters to vendor-accepted parameter names, passing through other native parameters, and finally automatically encapsulating heterogeneous model responses into OpenAI standard responses.

This implementation path uniformly defines CNLLM standard parameters, aligns with OpenAI standard response structures, preserves the complete capabilities of Chinese LLMs, and ensures scalability for integrating more vendors. Compared to OpenAI SDK and vendor proprietary SDKs, CNLLM also implements **systematic enhancements** for key field parsing, frontend streaming rendering, and engineering batch processing scenarios.

Through CNLLM, developers can seamlessly use Chinese LLMs in the OpenAI ecosystem — LangChain, LlamaIndex, AutoGen, Haystack, DeepEval and other mainstream large language model application frameworks. Especially in development and application scenarios requiring multi-model collaboration, using CNLLM can **significantly reduce adaptation, parsing, feature implementation, and maintenance workload, and effectively lower token consumption in AI agent development**.

- **Unified Interface** - One set of interfaces and parameters to call different Chinese LLMs, returns OpenAI API standard format
- **Complete Model Capabilities** - Calls Chinese LLMs' native interfaces (or backward-compatible interfaces), supports all model native parameters, preserving complete model capabilities
- **Mainstream Framework Integration** - Deeply integrated with LangChain Runnable, more framework deep adaptation development in progress
- **Encapsulated Key Fields** - Provides `.still`/`.tools`/`.think` property access for content/tool_calls/reasoning_content fields, supporting real-time updates and accumulation in streaming and batch requests
- **Batch Capability Enhancement** - Supports independent configuration for single requests in batch tasks, real-time statistics, callbacks, stop on error, custom indices, field storage, and various other engineered batch processing features

### Collaboration Opportunities

Welcome developers to participate in CNLLM's development. Please submit an Issue to discuss your solution before creating a Pull Request.

Or contact us at: <wangkancheng1122@163.com>

| Direction | Description |
| -------- | ----------- |
| 🌐 **New Vendor Adapters** | Integrate more Chinese LLMs (Alibaba Qwen, Baidu Wenxin, Tencent Hunyuan, etc.) |
| 🔗 **Framework Integration** | Deepen integration with LlamaIndex, LiteLLM, and other frameworks |
| 🐛 **Capability Expansion** | Adapter framework development for multimodal capabilities |
| 📖 **Documentation** | Add use cases and improve development guides |
| 💡 **Feature Suggestions** | Share your ideas and requirements |

Project Documentation:

- [System Architecture](docs/ARCHITECTURE.md)
- [Vendor Development Guide](docs/CONTRIBUTOR.md)
- [Feature Documentation](docs/feature/)

***

## Changelog

### v0.9.3 (2026-05-14)

- ✨ **New Vendors**
  - Qwen: qwen3.6/qwen3.5/qwen-plus/qwen-turbo/qwen-max and 13 models total + Embedding models
  - Baidu: ernie-5.1/ernie-4.5/ernie-speed/ernie-lite/ernie-x1 and 11 models total + Embeddings models
  - Hunyuan: hy3-preview/hunyuan-2.0-thinking/hunyuan-2.0-instruct
- ✨ **LangChain Integration**
  - `LangChainRunnable(BaseChatModel)` adds support for `bind_tools()` / `with_structured_output()` methods
  - New `LangChainEmbeddings`: adapts `langchain_core.embeddings.Embeddings`, supports `embed_documents()` / `embed_query()`
- ✨ **MiniMax Dual Interface Adaptation**
  - Added `MiniMaxNativeAdapter`: CNLLM now adapts MiniMax models with both native and OpenAI-compatible dual interfaces
  - Native interface supports `stream_options`, `group_id` vendor-specific pass-through parameters, and CNLLM returns OpenAI standard responses
  - In OpenAI-compatible interface, the `.think` property extracts and accumulates model's thinking content in real-time

### v0.9.2 (2026-05-14)

- 🔧 **Framework Use Case Tests**
  - Added `tests/key_needed/framework` directory, containing test cases for CNLLM integration with langchain, llamaindex, autogen, haystack, deepeval frameworks in production scenarios
- 🔧 **Refactoring**
  - Removed `StreamChunks`, merged into `StreamAccumulator`
  - Removed seamless async support (`_SyncProxy` and 5 other classes), now async clients must use async syntax
  - `StreamAccumulator._accumulate()` caching, `from_chunks()` class method etc.

### v0.9.1 (2026-05-14)

- ✨ **`keep`** **parameter — Storage Control**
  - `batch()` adds `keep` parameter to control persistent storage of batch response fields
  - All fields in batch responses can be accessed in real-time during iteration, with results updated and accumulated in real-time; after iteration, accessing fields not specified in `keep` returns empty container + warning
  - Default strategy (when `keep` is not configured):
    - `chat.batch()` responses default to keeping key fields `still`/`think`/`tools` and batch metadata, releasing other redundant fields
    - `embeddings.batch()` responses default to keeping key field `vectors` and batch metadata, releasing other redundant fields
- ✨ **`drop_params`** **parameter — Unknown Parameter Handling Strategy**
  - `create()` and `batch()` add `drop_params` parameter, supporting three-tier parameter handling strategies:
    - `drop_params="warn"`: warns that parameters are not taking effect, ignores and continues, default strategy
    - `drop_params="ignore"`: silently ignores unknown parameters and continues execution
    - `drop_params="strict"`: throws exception, terminates request execution
- ✨ **`usage`** **field — Usage Statistics**
  - `batch()` response now includes `usage` field, storing full Token consumption statistics for batch processing, accessed via `.usage`
- ✨ **batch embeddings response format**
  - `embeddings.batch()` response now includes `vectors` field, storing embedding vectors returned from batch requests, accessed via `.vectors`
  - `embeddings.batch()` response now includes `batch_info` field, storing batch metadata like `batch_size`, accessed via `.batch_info`

## Supported Models

### Chat Completions:

- **DeepSeek**
  - `deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-pro`, `deepseek-v4-flash`
- **KIMI (Moonshot AI)**
  - `kimi-k2.6`, `kimi-k2.5`, `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`, `moonshot-v1-vision-preview`
- **Doubao**
  - `doubao-seed-2-0-pro`, `doubao-seed-2-0-mini`, `doubao-seed-2-0-lite`, `doubao-seed-2-0-code`, `doubao-seed-1-8`, `doubao-seed-1-6`, `doubao-seed-1-6-flash`, `doubao-seed-1-6-vision`, `doubao-1-5-vision-pro-32k`, `doubao-seed-1-5-lite-32k`, `doubao-seed-1-5-pro-32k`, `doubao-seed-1-5-pro-256k`
- **GLM**
  - `glm-4.6`, `glm-4.7`, `glm-4.7-flash`, `glm-4.7-flashx`, `glm-5`, `glm-5-turbo`, `glm-5.1`, `glm-4.5`, `glm-4.5-x`, `glm-4.5-air`, `glm-4.5-airx`, `glm-4.5-flash`, `glm-5v-turbo`, `glm-4.5v`, `glm-4.6v`, `glm-4.6v-flash`
- **Xiaomi mimo**
  - `mimo-v2-pro`, `mimo-v2-omni`, `mimo-v2-flash`, `mimo-v2.5-pro`, `mimo-v2.5`
- **MiniMax**
  - `MiniMax-M2`, `MiniMax-M2.1`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`
- **Qwen**
  - `qwen3.6-max-preview`, `qwen3.6-plus`, `qwen3.6-flash`, `qwen3.5-plus`, `qwen3.5-flash`, `qwen3.5-397b-a17b`, `qwen3.5-122b-a10b`, `qwen3.5-27b`, `qwen3.5-35b-a3b`
- **Baidu**
  - `ernie-5.1`, `ernie-5.0`, `ernie-4.5-turbo-128k`, `ernie-4.5-turbo-32k`, `ernie-4.5-turbo-vl`, `ernie-4.5-turbo-vl-32k`, `ernie-4.5-0.3b`, `ernie-speed-pro-128k`, `ernie-lite-pro-128k`, `ernie-x1.1`, `ernie-x1-turbo-32k`
- **Hunyuan**
  - `hy3-preview`, `hunyuan-2.0-thinking-20251109`, `hunyuan-2.0-instruct-20251111`

### Embeddings:

- **GLM**: `embedding-2`, `embedding-3`, `embedding-3-pro`
- **Qwen**: `text-embedding-v4`, `text-embedding-v3`, `text-embedding-v2`, `text-embedding-v1`
- **Baidu**: `embedding-v1`, `bge-large-zh`, `bge-large-en`

## 1. Quick Start

### 1.1 Installation

#### 1.1.1 SDK Installation
```bash
pip install cnllm
```

#### 1.1.2 Install as Agent Skill

**One-Click Install**:
```bash
npx skills add https://github.com/kanchengw/cnllm
```

Or manually copy the `SKILL.md` file from the project root to your agent's skill directory. When **calling Chinese LLMs, CNLLM will be used as the preferred option**.

### 1.2 Client Initialization

#### 1.2.1 Sync Client

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2.7", api_key="your_api_key")
resp = client.chat.create(...)
```

#### 1.2.2 Async Client

Async clients need to be called via `await`, and streaming responses are iterated via `async for`:

```python
from cnllm import asyncCNLLM
import asyncio

async def main():
    client = asyncCNLLM(
        model="minimax-m2.7", api_key="your_api_key")
    resp = await client.chat.create(...)
    print(resp)

asyncio.run(main())
```

### 1.3 Context Management

Two context management modes are supported:

- **Persistent Session** maintains session state across multiple calls, suitable for applications that need to maintain context
- **Temporary Session** is single-use, does not maintain session state, auto-closes

**Persistent Session**:

```Python
client = CNLLM(
    model="minimax-m2.7", api_key="your_api_key")
resp = client.chat.create(...)
client.close()                         # Manual close; async client uses client.aclose()
```

**Temporary Session**:

```Python
with CNLLM(
    model="deepseek-chat", api_key="your_api_key") as client:
    resp = client.chat.create(...)     # Auto-closes session
```

## 2. Call Scenarios

All methods support both sync and async clients:

| Type | Scenario | Method | Return Type |
| -- | -- | --------------- | --------------------- |
| **chat completions** | Non-streaming single | `chat.create()`        | `Dict`                |
|   | Streaming single | `chat.create(stream=True)`          | `Iterator[Dict]`      |
|   | Non-streaming batch | `chat.batch()`         | `BatchResponse`       |
|   | Streaming batch | `chat.batch(stream=True)`          | `Iterator[Dict]`      |
|   | Mixed streaming batch | `chat.batch(requests=[{"stream": True}, {"stream": False}])` | `BatchResponse`       |
| **embeddings** | Embeddings single | `embeddings.create()` | `Dict`                |
|   | Embeddings batch | `embeddings.batch()` | `EmbeddingResponse`   |

### 2.1 Chat Completions Single Call

Three calling methods are supported, with the simplest being one line of code, one parameter:

**Simplified Call:**
Does not support any parameters other than strings (streaming can be configured at client level with `stream=True` parameter).

```python
resp = client("Introduce yourself in one sentence")
```

**Standard Call:**

```python
resp = client.chat.create(prompt="Introduce yourself in one sentence", stream=True)
```

**Full Call:**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Introduce yourself in one sentence"},
        {"role": "assistant", "content": "I am an intelligent assistant"},
        {"role": "user", "content": "Hello"},
    ]
)
```

#### 2.1.1 Non-Streaming Call

```python
resp = client.chat.create(
    messages=[{"role": "user", "content": "Introduce yourself in one sentence"}],
)
```

#### 2.1.2 Streaming Call

```python
resp = client.chat.create(
    prompt="Introduce yourself in one sentence",
    stream=True
)
for chunk in resp:
    print(resp.still)  # Real-time accumulated model response text
print(resp.raw)  # Complete accumulated model native response
```

#### 2.1.3 Response Access

In streaming calls, access via `for` loop with **real-time accumulation** for responses or the following key fields; non-streaming calls do not support `for` iteration, and access returns complete field content:

| Response Field | Access Method | Return Format | Example |
| ------------------------------ | ------------ | ----------------- | ------------------------------------------------ |
| **resp**: standard response | `resp`        | `Dict`/`List[Dict]` | `{non-streaming standard response}`/`[streaming chunks list]` |
| **think**: `reasoning_content` | `resp.think` | `str`             | `"reasoning content..."`                                      |
| **still**: `content`           | `resp.still` | `str`             | `"response content..."`                                      |
| **tools**: `tool_calls`        | `resp.tools` | `Dict[int, Dict]` | `{0: {"id": "...", "function": {...}}, 1: {...}` |
| **raw**: model native response | `resp.raw`   | `Dict`            | `{"id": "...", "choices": [...], ...}`           |

**repr():** 
During streaming, displays **real-time merged chunks and accumulated field results**, not the real-time streaming chunks list; does not change the streaming response object type, which is an **iterator** containing all standard streaming chunks.
```python
for chunk in resp:
    print(resp)
# {'id': '...', 'object': '...', 'created': '...', 'model': '...', 'choices': [{'delta': {'content': 'real-time accumulated model response', 'reasoning_content': 'real-time accumulated reasoning process'}, 'finish_reason': 'None'}]}
```


### 2.2 Chat Completions Batch Call

You can use `prompt` and `messages` parameters for quick global configuration, or use `requests` parameter for independent configuration of individual requests.

**prompt parameter:**

```python
resp = client.chat.batch(
    prompt=["Hello", "How's the weather today", "Who are you"],
    stream=True
)
```

**messages parameter:**

```python
resp = client.chat.batch(
    messages=[
        [{"role": "user", "content": "How's the weather in Beijing?"},
         {"role": "assistant", "content": "It's sunny in Beijing"},
         {"role": "user", "content": "What about Shanghai?"}],
        [{"role": "user", "content": "How's the weather in Shanghai?"}],
    ],
    tools=[get_weather]
)
```

**requests parameter:**

Configure **independent strategy** for individual requests within batch, global parameters are inherited when not configured per-request, supports using `requests.messages` parameter to manage context.

```python
resp = client.chat.batch(
    requests=[
        {"prompt": "How's the weather in Beijing?", "tools": [get_weather], "stream": True},  # Inherits thinking parameter from global config
        {"prompt": "What is 1+1?", "tools": [calc], "thinking": False},  # Does not inherit any global parameters
        {"prompt": "How's the weather in Guangzhou?", "model": "deepseek-chat", "api_key": "key"}  # Inherits tools and thinking parameters from global config
    ],
    # Global parameters (used when per-request not configured):
    tools=[default_tool],
    thinking=True,
    max_concurrent=2  # Max concurrent: batch-level parameter, not inherited by individual requests
)
```

#### 2.2.1 Chat Batch Response Structure

BatchResponse outer structure, where each response under `results[request_id]` is in **OpenAI standard streaming/non-streaming response structure**:

```python
{
    "status": {"elapsed": "3.42s", "success_count": 2, "fail_count": 1, "total": 3},  # Statistics
    "usage": {"prompt_tokens": 5, "total_tokens": 5},  # Batch processing total usage info
    "errors": {"request_2": "error message"},  # Mapping of all failed requests' request_id and error messages
    "results": {     # Mapping of all successful requests' request_id and standard responses
        "request_0": {...},
        "request_1": {...}
    },
    "think": {"request_0": "...", "request_1": "..."},
    "still": {"request_0": "...", "request_1": "..."},
    "tools": {"request_0": [...], "request_1": [...]},
    "raw": {"request_0": {...}, "request_1": {...}}
}
```

#### 2.2.2 Chat Batch Response Access

Supports iterative access to **response results, metadata, and key field contents** within `for` loop, with content **real-time accumulation and updates**:

- In batch streaming calls, updates build chunk by chunk; in batch non-streaming calls and **batch calls with mixed streaming strategies** (see `requests` parameter), updates build request by request.
- In batch non-streaming calls and batch calls with mixed streaming strategies, if real-time access to batch response fields is not needed, you can access complete results directly, skipping the `for` loop.
- Supports access by `request_id` or by integer index.

**Access methods:**

```python
resp = client.chat.batch(
    prompt=["Hello", "How's the weather today", "Who are you"]
)

for r in resp:
    print(resp.status)  # Real-time statistics, request by request real-time update

print(resp.still)  # Response content for all requests in batch task

# Or access via client.chat.batch_result:
for r in client.chat.batch(
    prompt=["Hello", "How's the weather today", "Who are you"], stream=True
):
    print(client.chat.batch_result.results)  # OpenAI standard streaming responses for all requests in batch task, chunk by chunk real-time accumulation

print(client.chat.batch_result.think["request_0"])  # Reasoning content for first request in batch task, or use .think[0] integer index access
```

**Access fields:**

| Category | Field Description | Access Method | Return Format | Example |
| ----------- | ----------- | --------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| **Metadata** | Real-time statistics | `resp.status` / `batch_result.status`         | `Dict`                       | `{"success_count": 2, "fail_count": 0, "total": 2, "elapsed": "3.42s"}` |
| <br />      | Real-time Token usage | `resp.usage` / `batch_result.usage`           | `Dict[str, int]`             | `{"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}`  |
| **errors**  | Error information for failed requests | `resp.errors` / `batch_result.errors`         | `Dict[str, str]`             | `{"request_0": "error message","request_1": "error message"}`                                        |
| <br />      | Error information for single request   | `resp.errors[0]` / `batch_result.errors[0]`   | `str`                        | `"error message"`                                                             |
| **results** | Standard response for successful requests | `resp.results` / `batch_result.results`       | `Dict[str, Dict]`            | `{"request_0": {...}, "request_1": {...}}`                              |
| <br />      | Standard response for each request   | `resp.results[0]` / `batch_result.results[0]` | `Dict`                       | `{"id": "...", "choices": [...], ...}`                                  |
| **think**   | Reasoning process content      | `resp.think` / `batch_result.think`           | `Dict[str, str]`             | `{"request_0": "...", "request_1": "..."}`                              |
| <br />      | Reasoning content for single request   | `resp.think[0]` / `batch_result.think[0]`     | `str`                        | `"reasoning content..."`                                                             |
| **still**   | Response content        | `resp.still` / `batch_result.still`           | `Dict[str, str]`             | `{"request_0": "...", "request_1": "..."}`                              |
| <br />      | Response content for single request   | `resp.still[0]` / `batch_result.still[0]`     | `str`                        | `"response content..."`                                                             |
| **tools**   | Tool calls        | `resp.tools` / `batch_result.tools`           | `Dict[str, Dict[int, Dict]]` | `{"request_0": {...}, "request_1": {...}}`                              |
| <br />      | Tool calls for single request   | `resp.tools[0]`                               | `Dict[int, Dict]`            | `{0: {"id": "...", "function": {...}}, 1: {...}`                        |
| **raw**     | Model native response      | `resp.raw` / `batch_result.raw`               | `Dict[str, Dict]`            | `{"request_0": {...}, "request_1": {...}}`                              |
| <br />      | Model native response for single request | `resp.raw[0]` / `batch_result.raw[0]`         | `Dict`                       | `{"id": "...", "choices": [...], ...}`                                  |

**repr():** Displays batch processing metadata fields or response content:

```python
print(resp)
# BatchResponse(status={...}, usage={...})

print(resp.results)
```

### 2.3 Embeddings Batch Call

**prompt parameter:**
```python
resp = client.embeddings.batch(
    input=["Hello", "World", "你好"],
)
print(resp.vectors)   # Embedding vectors for all requests
print(resp.status)    # Statistics
print(resp.usage)     # Token usage statistics
```

**custom_ids parameter:**
```python
resp = client.embeddings.batch(
    input=["Text 1", "Text 2", "Text 3"],
    custom_ids=["doc_001", "doc_002", "doc_003"]
)

resp.results["doc_001"]          # Get response for doc_001
resp.vectors["doc_002"]          # Get embedding vector for doc_002
```

**to_dict():** Converts response to dictionary:

```python
resp.to_dict()               # Default: keeps vectors field + metadata (status/usage/batch_info)
resp.to_dict(results=True)   # Keeps results field + metadata (status/usage/batch_info)
```

### 2.4 Batch Call Control Parameters

Batch calls support **retry strategy, concurrency control** parameter configuration:

| Parameter | Type | Default | Description |
| ---------------- | ------- | -------- | ------------------------------------------ |
| `batch_size`     | `int`   | Dynamic  | Batch size, only supported for Embeddings calls                  |
| `max_concurrent` | `int`   | `12`/`3` | Max concurrent, Embeddings default 12, Chat completions default 3 |
| `rps`            | `float` | `10`/`2` | Requests per second, Embeddings default 10, Chat completions default 2 |
| `timeout`        | `int`   | 30       | Per-request timeout (seconds)                                   |
| `max_retries`    | `int`   | 3        | Max retry times                                     |
| `retry_delay`    | `float` | 1.0      | Retry delay (seconds)                                    |

**batch\_size**:
Only supported for batch Embeddings calls, defaults to adaptive calculation based on request count, manual configuration not recommended.

### 2.5 Batch Call Advanced Features

Both batch chat completions/Embeddings calls support **progress callbacks, custom request IDs, stop on error, field storage control, unknown parameter handling strategy**.

#### 2.5.1 Custom Request ID

Use `custom_ids` parameter to specify custom IDs for batch requests, which will replace the original request_id in batch responses.

```python
resp = client.embeddings.batch(
    input=["Text 1", "Text 2", "Text 3"],
    custom_ids=["doc_001", "doc_002", "doc_003"]
)

resp.results["doc_001"]          # Get response for doc_001
resp.vectors["doc_002"]          # Get embedding vector for doc_002
```

#### 2.5.2 Progress Callback

Callbacks are invoked **when each request completes**, which can be used for:

- Real-time display of processing progress
- Recording completed tasks
- Dynamically adjusting subsequent tasks
- ...

```python
def on_complete(request_id, status):          # Callback function example, supports customization
    print(f"[{request_id}] {status}")

resp = client.chat.batch(
    requests,
    callbacks=[on_complete]
)
```

#### 2.5.3 Stop on Error

When a batch request encounters the first error, it immediately throws an exception and interrupts subsequent tasks. If there are successful requests in the batch, it also returns a batch object containing already processed request results, which can be accessed normally:

```python
resp = client.embeddings.batch(
    input=requests,
    stop_on_error=True
)
# Error message: {request_id} request failed, reason: {error}

# If there are successful requests in the batch, you can access the batch object normally:
resp.status
resp.vectors
```

#### 2.5.4 Field Storage Control

Batch calls (Chat / Embeddings) can access all fields within the `for` loop. After iteration ends, some redundant fields are automatically released to save memory.
The `keep` parameter specifies which fields need to be retained after iteration:

**Default behavior (when keep parameter is not specified):**

| Call Type                        | Default Retention                    | Auto-released after Iteration              |
| --------------------------- | ----------------------- | -------------------- |
| `client.chat.batch()`       | `still/think/tools` and metadata | `results/errors/raw` |
| `client.embeddings.batch()` | `vectors` and metadata           | `results/errors`     |

**Notes:**

- When `keep=[]`, all fields are released after iteration, only metadata is retained; when `keep=["*"]`, all fields are retained after iteration.
- In `chat.batch()`, metadata fields include `status/usage`; in `embeddings.batch()`, metadata fields include `status/usage/batch_info`.

**Usage:**

```python
resp = client.embeddings.batch(
    input=["Text 1", "Text 2", "Text 3"],
    keep=["vectors"]         # Only retain vectors field after iteration
)
for _ in resp:
    print(resp.results)      # Any field can be accessed during iteration, request by request real-time accumulation

resp.vectors["request_0"]    # Accessible after iteration 
resp.results["request_0"]    # Not accessible after iteration, returns warning
```

Can also set global default at client initialization:

```python
client = CNLLM(..., keep=["vectors"])
```

#### 2.5.5 Unknown Parameter Handling Strategy

Use `drop_params` to control the handling behavior of **incompatible parameters and other unknown parameters** held by the client during actual calls. The default strategy is `warn` mode.

| Strategy | Configuration | Behavior |
| -------- | ---------------------- | ----------------------------- |
| Warning mode (default) | `drop_params="warn"`   | Prints warning log, parameter is discarded, request continues             |
| Strict mode     | `drop_params="strict"` | Throws `TypeError`, request terminated |
| Silent ignore mode   | `drop_params="ignore"` | Silently discards unknown parameters, no logs generated              |

**Notes:**
- When doing batch calls, if global parameters contain unknown parameters, `drop_params="strict"` directly throws an exception without actually starting the batch task;
If a single request within the batch task contains unknown parameters, `drop_params="strict"` directly puts that request into the `errors` field without actually executing that request, and continues executing subsequent batch tasks.

- Specifically, when configured with `drop_params="strict"` and `stop_on_error=True`, the first error encountered in batch requests immediately interrupts the batch task while returning already processed request results. See [Stop on Error](#253-stop-on-error).
- The `drop_params` parameter supports client configuration and all calling methods (including `create` single-call method).

## 3. CNLLM Standard Response Format

CNLLM's streaming, non-streaming, and Embeddings response formats for single requests are fully aligned with OpenAI standard structure.

### 3.1 Non-Streaming Response Format

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
            "content": "Hello, I am MiniMax-M2.7...",
            "reasoning_content": "reasoning process content..."    # Model reasoning process, if any
            "tool_calls": [{                        # Tool calls, if any
                "id": "call_xxx",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\":\"Beijing\"}"}
            }]
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

### 3.2 Streaming Response Format

```python
{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}

# reasoning_content chunks (model reasoning process, if any):
{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'reasoning..'}, 'finish_reason': None}]}

# tool_calls chunks (tool calls, if any):
{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call_xxx', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '...'}}]}, 'finish_reason': None}]}

{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {'content': 'Hello...'}, 'finish_reason': None}]}

# ... chunks

{'id': 'chatcmpl-xxx', 'object': 'chat.completion.chunk', 'created': 1234567890, 'model': 'minimax-m2.7', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}
```

### 3.3 Embeddings Response Format

```python
{
    "object": "list",
    "data": [{
        "object": "embedding",
        "embedding": [0.1, 0.2, ...],
        "index": 0
    }],
    "model": "embedding-2",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

## 4. CNLLM Unified Interface Parameters

Except for parameters specially noted below, other parameters can be configured at **both client initialization and call entry**. Call entry configuration will **override** client initialization configuration.

### 4.1 CNLLM Request Parameters

CNLLM request parameters are basically consistent with **OpenAI standard parameters**, with slight extensions based on domestic vendor situations. For uncovered parameters, vendor naming is used and **passed through**.
Note: Not all supported models support all request parameters. Please refer to vendor official documentation for confirmation, or configure `drop_params="ignore"` to ignore unsupported parameters.

#### 4.1.1 Basic Parameters

| Parameter | Type | Default | Description |
| ------------------- | ------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `model`             | `str`                           | -                               | Model name, required at client initialization, can be overridden at call entry         |
| `api_key`           | `str`                           | -                               | API key                                                 |
| `base_url`          | `str`                           | Auto-adapted                            | Customizable API address                                            |
| `messages`          | `list[dict]`/`list[list[dict]]` | -                               | `chat()` input parameter, supports context management/image recognition (call entry configuration only)                           |
| `prompt`            | `str`/`list[str]`               | -                               | `chat()` input parameter (call entry configuration only)                            |
| `requests`          | `list[dict]`                    | -                               | `chat.batch()` input parameter, supports per-request independent configuration (call entry configuration only) |
| `input`             | `str`/`list[str]`               | -    | `embeddings()` input parameter (call entry configuration only) |
| `stream`            | `bool`                          | `False`                         | Streaming response                                                   |
| `thinking` ¹         | `bool/dict`                     | Determined by model endpoint, most default to `False`            | Thinking mode, supports `True`/`False`, some models support `"auto"`                 |
| `tools`             | `list`                          | -                               | Tool/function definition list                                              |

¹ `thinking` mapping:
   - GLM, DeepSeek, Baidu, Hunyuan, Xiaomi, Kimi: `True` → `{"type": "enabled"}`, `False` → `{"type": "disabled"}`
   - Doubao: `True` → `"enabled"`, `False` → `"disabled"`, `"auto"` → `"auto"`
   - Qwen: `True` → `enable_thinking: true`, `False` → `enable_thinking: false`

#### 4.1.2 Advanced Parameters

| Parameter | Type | Default | Description |
| ------------------- | ------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `temperature`       | `float`                         | Determined by model endpoint                         | Generation randomness                                                  |
| `max_completion_tokens`        | `int`                           | Determined by model endpoint                         | Max generated token count (including thinking chain)                                           |
| `max_tokens`        | `int`                           | Determined by model endpoint                         | Max generated token count (excluding thinking chain)                                           |
| `top_p`             | `float`                         | Determined by model endpoint                         | Nucleus sampling threshold                                                  |
| `stop`              | `str/list`                      | -                               | Stop sequence                                                   |
| `reasoning_effort`  | `str`                           | Determined by model endpoint                         | Reasoning depth control                                                 |
| `tool_choice`       | `str/dict`                      | -                               | Tool selection strategy                                                 |
| `response_format`   | `dict`                          | Determined by model endpoint, most default to `{"type": "text"}` | Response format                                                   |
| `n`                 | `int`                           | `1`                             | Number of generated candidates                                                  |
| `presence_penalty`  | `float`                         | -                               | Presence penalty                                                   |
| `frequency_penalty` | `float`                         | -                               | Frequency penalty                                                   |
| `logit_bias`        | `dict`                          | -                               | Token-level bias                                             |
| `user` ¹             | `str`                           | -                               | User identifier                                                   |
| `seed`              | `int`                           | -                               | Random seed, same seed can reproduce results                                   |
| `stream_options`    | `dict`                          | -                               | Streaming output config, such as `{"include_usage": true}`                      |
| `logprobs`          | `bool`                          | `False`                         | Whether to return log probabilities of output tokens                                   |
| `top_logprobs`      | `int`                           | `0`                             | Number of highest probability candidate tokens to return for each position                              |

¹ `user` mapping:
   - GLM: `user` → `user_id`

### 4.1.3 Vendor Pass-through Parameters

Parameters supported by models but not covered in 4.1.1/4.1.2 will be passed through by CNLLM to the model endpoint.

| Vendor | Pass-through Parameters |
|------|---------|
| **KIMI** | `prompt_cache_key`, `safety_identifier`, `stream_options` |
| **Doubao** | `service_tier`, `stream_options` |
| **GLM** | `do_sample`, `request_id`, `tool_stream`, `dimensions` |
| **MiniMax** | `stream_options`, `group_id` |
| **Qwen** | `enable_thinking`, `preserve_thinking`, `thinking_budget`, `top_k`, `repetition_penalty`, `vl_high_resolution_images`, `enable_code_interpreter`, `enable_search`, `search_options`, `parallel_tool_calls`, `dimensions` |
| **Baidu** | `enable_thinking`, `thinking_budget`, `thinking_strategy`, `penalty_score`, `repetition_penalty`, `parallel_tool_calls`, `web_search`, `metadata` |

### 4.2 SDK Control Parameters

Parameters defined internally by CNLLM to control internal execution behavior or strategy, not transmitted to API endpoint.

#### 4.2.1 General Parameters

| Parameter | Type | Default | Description |
| ----------------- | ------- | -------- | ------------------ |
| `timeout`         | `int`   | `60`     | Request timeout (seconds)            |
| `max_retries`     | `int`   | `3`      | Max retry times             |
| `retry_delay`     | `float` | `1.0`    | Retry delay (seconds)            |
| `fallback_models`¹ | `dict`  | -        | Fallback models (client initialization only), see below for details |
| `drop_params`     | `str`   | `"warn"` | See [Unknown Parameter Handling Strategy](#255-unknown-parameter-handling-strategy) |

¹`fallback_models` model fallback strategy:

Fallback models are only supported at **client initialization**. If the primary `model` does not respond successfully, it will sequentially try the provided `fallback_models`. For application **robustness**, it is recommended to configure this option and set `drop_params="ignore"` to avoid parameter compatibility issues.

```python
fallback_models = {
    "deepseek-chat": {
        "api_key": "ds-key-456",     # required
        "base_url": "https://api.deepseek.com/v1",
    },
    "qwen-plus": {
        "api_key": "my-key",         # when base_url is not configured, default URL is used
    },
}
```

**Notes**:
- Specifying `model` again at the call entry overrides the client's primary model configuration. When the call entry's `model` fails, it will still try `fallback_models`
- In `chat.batch()`, fallback is tried per-req independently
- Non-retryable errors (model not found, missing params, content filtered) are raised directly without triggering fallback
- When all models fail, `FallbackError` is raised, aggregating all failure information

#### 4.2.2 Batch Method Parameters

Only effective for `chat.batch()` and `embeddings.batch()` calls:

| Parameter | Type | Default | Description |
| ---------------- | ----------- | ---------------------------- | --------------------- |
| `max_concurrent` | `int`       | Chat: `3` / Embeddings: `12` | Max concurrent                 |
| `rps`            | `float`     | Chat: `2` / Embeddings: `10` | Requests per second limit               |
| `batch_size`     | `int`       | Dynamic calculation                         | Batch size, only supported by Embeddings |
| `stop_on_error`  | `bool`      | `False`                      | Stop subsequent requests on error, return already processed results     |
| `callbacks`      | `list`      | -                            | Progress callback function list              |
| `custom_ids`     | `list[str]` | -                            | Custom request ID list           |
| `keep`           | `set/list`  | See [Field Storage Control](#254-field-storage-control)             | Data fields to retain after iteration            |

## 5. Framework Integration

### 5.1. LangChainRunnable Implementation

`LangChainRunnable` inherits `BaseChatModel`, natively supports `invoke`/`stream`/`batch` as well as `bind_tools`/`with_structured_output`.

```python
from cnllm import CNLLM
from cnllm.core.framework import LangChainRunnable, LangChainEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import asyncio

# Create CNLLM client
client = CNLLM(model="deepseek-chat", api_key="your_key")

# Create Runnable instance
runnable = LangChainRunnable(client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}")
])

# Build LangChain chain
chain = prompt | runnable

# Sync calls with invoke/stream/batch
resp = chain.invoke({"input": "What is 2+2?"})
print(resp.content)

for chunk in chain.stream({"input": "Count to 5"}):
    print(chunk.content, end="", flush=True)

resp = chain.batch([{"input": "Hello"}, {"input": "How are you?"}])
for r in resp:
    print(r.content)

# bind_tools — tool calling
@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return "Sunny 20°C"

llm_with_tools = runnable.bind_tools([get_weather])
resp = llm_with_tools.invoke("Weather in Beijing")
print(resp.content)

# with_structured_output — structured output
# deepseek-v4 series requires thinking=False to receive tool_choice from with_structured_output(); other models/vendors do not have this requirement
class Person(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")

structured = runnable.with_structured_output(Person)
result = structured.invoke("Zhang San is 28 years old")
print(result) # → Person(name="Zhang San", age=28)

# LangChainEmbeddings — embeddings
embeddings = LangChainEmbeddings(client)
vectors = embeddings.embed_documents(["Hello", "World"])
query_vec = embeddings.embed_query("query")

# Async calls with ainvoke/astream/abatch
async def main():
    async with client:
        resp = await chain.ainvoke({"input": "What is 2+2?"})
        print(resp.content)

        async for chunk in chain.astream({"input": "Count to 5"}):
            print(chunk.content, end="", flush=True)

        results = await chain.abatch([{"input": "A"}, {"input": "B"}])
        for r in results:
            print(r.content)

asyncio.run(main())
```

### 5.2. LlamaIndex — Response Consumption

CNLLM responses can be used to construct LlamaIndex's ChatMessage:

```python
from cnllm import CNLLM
from llama_index.core.llms import ChatMessage, MessageRole

client = CNLLM(model="deepseek-chat", api_key="your_key")
resp = client.chat.create(prompt="Introduce yourself in one sentence")

msg = ChatMessage(role=MessageRole.ASSISTANT, content=resp.still)
print(msg.content)
```

### 5.3. AutoGen — LLM Backend

CNLLM integrates with AutoGen via OpenAI-compatible responses:

```python
from cnllm import CNLLM
from autogen_agentchat.messages import TextMessage

client = CNLLM(model="deepseek-chat", api_key="your_key")
resp = client.chat.create(prompt="1+1=?")

msg = TextMessage(content=resp.still, source="assistant")
print(msg.content)
```

### 5.4. Haystack — Document & ChatMessage

CNLLM embeddings feed into Haystack Document, chat output constructs ChatMessage:

```python
from cnllm import CNLLM
from haystack import Document
from haystack.dataclasses import ChatMessage

client = CNLLM(model="deepseek-chat", api_key="your_key")

# embedding → Document
text = "CNLLM is a Chinese LLM adapter"
resp = client.embeddings.create(input=text)
doc = Document(content=text, embedding=resp.vectors)
print(f"Vector dimension: {len(doc.embedding)}")

# chat → ChatMessage
resp = client.chat.create(prompt="1+1=?")
msg = ChatMessage.from_assistant(resp.still)
print(msg.text)
```

### 5.5. DeepEval — Evaluation Test Cases

CNLLM output feeds into DeepEval evaluation:

```python
from cnllm import CNLLM
from deepeval.test_case import LLMTestCase

client = CNLLM(model="deepseek-chat", api_key="your_key")
resp = client.chat.create(messages=[{"role": "user", "content": "1+1=?"}])

test_case = LLMTestCase(
    input="1+1=?", actual_output=resp.still, expected_output="2",
)
print(test_case.actual_output)
```

### License

MIT License - See [LICENSE](LICENSE) file

### Contact

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
- Author Email: <wangkancheng1122@163.com>