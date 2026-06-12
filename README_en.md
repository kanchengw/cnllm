![Figure 1][Figure 1]

[Figure 1]: pics/figure_1.png

# CNLLM - Chinese LLM Adapter

[English](README.md) | [中文](README_zh.md)

[![PyPI Version](https://img.shields.io/pypi/v/cnllm)](https://pypi.org/project/cnllm/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776A4?style=flat)](https://pypi.org/project/cnllm/)
[![License](https://img.shields.io/github/license/kanchengw/cnllm)](https://github.com/kanchengw/cnllm/blob/main/LICENSE)

***

## Why CNLLM?

The CNLLM Python toolkit offers a **unified OpenAI-compatible interface layer** for all Chinese LLMs, alongside a suite of **enhanced utilities** to streamline LLM development workflows.. 

Through CNLLM, developers can seamlessly use Chinese LLMs in the OpenAI ecosystem — LangChain, LlamaIndex, AutoGen, Haystack, DeepEval and other mainstream large language model application frameworks. Especially in development and application scenarios requiring multi-model collaboration, using CNLLM can **significantly reduce adaptation, parsing, feature implementation, and maintenance workload, and effectively lower token consumption in AI agent development**.

- **Unified Interface** - One set of interfaces and parameters to call different Chinese LLMs, returns OpenAI API standard format response
- **Parameter Validation** - Validation and explicit feedback for all parameters, especially vendor native parameters, with support for parameter handling behavior control (`drop_params`)
- **Streaming Response** - Streaming lifecycle monitoring via `repr()`, and automatic accumulation of incremental fields via `.still`/`.think`/`.tools` property access
- **Batch Capability** - Independent configuration for single requests in batch tasks, with real-time batch progress statistics (`.status`), and configurable failure policy (`stop_on_error`) and memory management (`keep`).

**Example: Streaming Lifecycle View and Incremental Extraction/Automatic Accumulation**

![Figure 2][repr]

[repr]: pics/repr_demo.gif

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

### v0.9.10 (2026-06-12)

- ⚡ **Adaptive Scheduling + Pooling Algorithm**
  - `chat.batch(stream=False)` supports adaptive controller: dynamic concurrency, RPS limiting, RPM learning, 429 freeze/thaw
  - Streaming/mixed batch still requires manual `max_concurrent` and `rps` or defaults; non-streaming batch can override adaptive scheduling by specifying these params
  - With `fallback_models` + `performance=True`: weighted distribution by model throughput, fast/slow models don't block each other; `max_concurrent` and `rps` cannot be configured
  - With `performance=False` or default: primary model priority, failed requests auto-retry fallback models
- ✨ **Step (阶跃星辰)** new vendor
  - Models: `step-3-5-flash`, `step-3-7-flash`
  - Supports streaming/non-streaming Chat Completions, Tools, reasoning effort (`reasoning_effort`)
- ✨ **MiniMax** adds `minimax-m3` (`MiniMax-M3`) model
  - `thinking` parameter supports `True`/`False` for thinking mode (M3 only)
- ✨ **Qwen** adds `qwen3.7-max`, `qwen3.7-plus` models

### v0.9.3 (2026-05-29)

- ✨ **Context Building Tool**
  - New `ContextBox` class: one line of code to automatically format model responses, reasoning process, and tool call messages, and add them to the `messages` context list.
  - Supports `executor` parameter for custom tool executor function.
- ✨ **LangChain Integration**
  - `LangChainRunnable(BaseChatModel)` adds support for `bind_tools()` / `with_structured_output()` methods
  - New `LangChainEmbeddings`: adapts `langchain_core.embeddings.Embeddings`, supports `embed_documents()` / `embed_query()`

## Supported Models

### Chat Completions:

- **DeepSeek**
  - `deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-pro`, `deepseek-v4-flash`
- **KIMI (Moonshot AI)**
  - `kimi-k2.6`, `kimi-k2.5`, `moonshot-v1-128k` (`moonshot-v1`), `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-vision-preview`
- **Doubao**
  - `doubao-seed-2-0-pro-260215` (`doubao-seed-2-0-pro`), `doubao-seed-2-0-mini-260215` (`doubao-seed-2-0-mini`), `doubao-seed-2-0-lite-260215` (`doubao-seed-2-0-lite`), `doubao-seed-2-0-code-preview-260215` (`doubao-seed-2-0-code`), `doubao-seed-1-8-251228` (`doubao-seed-1-8`), `doubao-seed-1-6-251015` (`doubao-seed-1-6`), `doubao-seed-1-6-flash-250828` (`doubao-seed-1-6-flash`), `doubao-seed-1-6-vision-250815` (`doubao-seed-1-6-vision`), `doubao-1-5-vision-pro-32k-250115` (`doubao-1-5-vision-pro`), `doubao-seed-1-5-lite-32k-250115` (`doubao-seed-1-5-lite`), `doubao-seed-1-5-pro-32k-250115` (`doubao-seed-1-5-pro-32k`), `doubao-seed-1-5-pro-256k-250115` (`doubao-seed-1-5-pro`)
- **GLM**
  - `glm-4.6`, `glm-4.7`, `glm-4.7-flash`, `glm-4.7-flashx`, `glm-5`, `glm-5-turbo`, `glm-5.1`, `glm-4.5`, `glm-4.5-x`, `glm-4.5-air`, `glm-4.5-airx`, `glm-4.5-flash`, `glm-5v-turbo`, `glm-4.5v`, `glm-4.6v`, `glm-4.6v-flash`
- **Xiaomi mimo**
  - `mimo-v2-pro`, `mimo-v2-omni`, `mimo-v2-flash`, `mimo-v2.5-pro`, `mimo-v2.5`
- **MiniMax**
  - `MiniMax-M3`, `MiniMax-M2`, `MiniMax-M2.1`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`
- **Qwen**
  - `qwen3.7-max`, `qwen3.7-plus`, `qwen3.6-max-preview`, `qwen3.6-plus`, `qwen3.6-flash`, `qwen3.5-plus`, `qwen3.5-flash`, `qwen3.5-397b-a17b`, `qwen3.5-122b-a10b`, `qwen3.5-27b`, `qwen3.5-35b-a3b`
- **Baidu**
  - `ernie-5.1`, `ernie-5.0`, `ernie-5.0-thinking-perview`, `ernie-4.5-8k-preview`, `ernie-4.5-turbo-128k` (`ernie-4.5-turbo`), `ernie-4.5-turbo-32k`, `ernie-4.5-turbo-vl`, `ernie-4.5-turbo-vl-32k`, `ernie-4.5-0.3b`, `ernie-speed-pro-128k` (`ernie-speed-pro`), `ernie-lite-pro-128k` (`ernie-lite-pro`), `ernie-x1.1`, `ernie-x1-turbo-32k` (`ernie-x1-turbo`)
- **Step (阶跃星辰)**
  - `step-3-5-flash`, `step-3-7-flash`
- **Hunyuan**
  - `hy3-preview`, `hunyuan-2.0-thinking-20251109` (`hunyuan-2.0-thinking`), `hunyuan-2.0-instruct-20251111` (`hunyuan-2.0-instruct`)

### Embeddings:

- **GLM**: `embedding-2`, `embedding-3`, `embedding-3-pro`
- **Qwen**: `text-embedding-v4`, `text-embedding-v3`, `text-embedding-v2`, `text-embedding-v1`
- **Baidu**: `embedding-v1`, `bge-large-zh`, `bge-large-en`

## 1. Quick Start

### 1.1 Installation

#### 1.1.1 Install as Agent Skill (Recommended)

CNLLM now provides a dedicated Agent Skill following the Claude Skills / Agent Skills standard.

**Install the skill**:
```bash
npx skills add kanchengw/cnllm-skill
```

📖 For full documentation and examples, visit the dedicated skill repository:
https://github.com/kanchengw/cnllm-skill

#### 1.1.2 SDK Installation
```bash
pip install cnllm
```

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
|   | Mixed streaming batch | `chat.batch(requests=[{"stream": True}, {"stream": False}])` | `Iterator[Dict]`       |
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

Streaming responses provide **two access layers** for different usage scenarios:

```python
from cnllm import ToolCollector

resp = client.chat.create(
    prompt="Introduce yourself in one sentence",
    stream=True,
    thinking=True,
    tools=tools,
)

# ── During iteration: chunk.* returns per-frame increments, suitable for frontend real-time rendering / streaming process monitoring ──
with resp as view:   # Complete view merged chunk by chunk
    for chunk in resp:
        frontend_content.append(chunk.still)    # delta.content, character-level increment
        frontend_reasoning.append(chunk.think)  # delta.reasoning_content, character-level increment
        frontend_tools.update(chunk.tools)      # delta.tool_calls, per index merge
        view.refresh()                          # Real-time refresh view

# ── After stream ends: resp.* returns complete accumulated results, suitable for getting final values ──
print(resp.still)   # Complete model response text
print(resp.think)   # Complete reasoning process
print(resp.tools)   # Complete tool calls
print(resp)         # Complete merged OpenAI dict
```

#### 2.1.3 Response Access

**Non-streaming / Streaming common** (can be accessed directly when `stream=False`; recommended to access after stream ends when `stream=True`):

| Access Method | Return Content | Return Format | Example |
|-------------|-------------|-------------|---------|
| `resp` | OpenAI standard response | `Dict` / `Iterator[Dict]` | Non-streaming returns complete dict / streaming returns chunk list |
| `resp.still` | Model response text (`content`) | `str` | `"Hello, I'm..."` |
| `resp.think` | Reasoning process (`reasoning_content`) | `str` | `"reasoning content..."` |
| `resp.tools` | Tool calls (`tool_calls`) | `List[Dict]` | `[]` |
| `resp.raw` | Model native response | `Dict` / `List[Dict]` | Non-streaming returns complete dict / streaming returns chunks list |

**Streaming-exclusive** (only accessible during iteration when `stream=True`, returns per-chunk increments):

| Access Method | Return Content | Return Format | Example |
|-------------|-------------|-------------|---------|
| `chunk.still` | Current chunk's `delta.content` increment | `str` | `"Y"`, `"ou"` |
| `chunk.think` | Current chunk's `delta.reasoning_content` increment | `str` | `"Th"`, `"ink"` |
| `chunk.tools` | Current chunk's `delta.tool_calls` increment | `List[Dict]` | `[]` |
| `with resp as view` | Complete view merged chunk by chunk (real-time refresh) | `LiveDict` context manager | `{real-time view}` |

#### 2.1.4 Context Building for Multi-turn Conversation

`ContextBox` automatically formats `resp.still` / `resp.think` / `resp.tools` containing complete context content into the `messages` list for the next round of conversation.

```python
from cnllm import ContextBox

# Build assistant message (think + still auto-concatenated, tool_calls auto-attached)
messages += ContextBox(resp.still, resp.think)

# Or in tool calling scenario, pass executor to auto-execute and append tool result
def execute_weather_tool(tc):
    """tc: {"id": "call_xxx", "function": {"name": "get_weather", "arguments": "..."}}"""
    args = json.loads(tc["function"]["arguments"])
    return json.dumps(get_weather(args["location"]))

messages += ContextBox(resp.still, resp.think, resp.tools,
                       executor=execute_weather_tool)
# → Auto produces:
#   {"role": "assistant", "content": "think...\n\nstill...", "tool_calls": resp.tools}
#   {"role": "tool", "tool_call_id": "call_xxx", "content": "Tool execution result"}


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
    "usage": {"prompt_tokens": 5, "total_tokens": 5},     # Batch processing total usage info
    "errors": {"request_2": "error message"},             # Mapping of all failed requests' request_id and error messages
    "results": {"request_0": {...}, "request_1": {...}},  # Mapping of all successful requests' request_id and standard responses
    "think": {"request_0": "...", "request_1": "..."},
    "still": {"request_0": "...", "request_1": "..."},
    "tools": {"request_0": {...}, "request_1": {...}},
    "raw": {"request_0": {...}, "request_1": {...}}
}
```

#### 2.2.2 Chat Batch Response Access

**Terminal real-time observation:**

```python
resp = client.chat.batch(
    prompt=["Hello", "How's the weather today", "Who are you"],
    stream=True,
)

with resp as view:   # Real-time refresh metadata view
    for r in resp:
        view.refresh()
```

**Real-time increment during iteration** (streaming batch / mixed streaming batch available):

```python
# chunk.* returns per-frame increments, request_id auto-routes
for chunk in resp:
    rid = chunk["request_id"]
    frontend_still[rid].append(chunk.still)
    frontend_think[rid].append(chunk.think)
```

**Get full content after stream ends:**

```python
print(resp.still)   # {"request_0": "Hello", "request_1": "...", "request_2": "..."}
print(resp.think)   # {"request_0": "reasoning...", "request_1": "..."}
print(resp.tools)   # {"request_0": [{"function": {"name": "get_weather", ...}}]}
print(resp)   # Complete metadata view accumulated result
```

**Common access fields:**

| Access Method | Return Content | Return Format | Example |
|-------------|-------------|-------------|---------|
| `resp.status` | Real-time statistics | `Dict` | `{"success_count":2,"elapsed":"3.42s"}` |
| `resp.usage` | Token usage | `Dict[str, int]` | `{"total_tokens":150}` |
| `resp.errors` | Failed request info | `Dict[str, str]` | `{"request_0": "error"}` |
| `resp.results` | Standard response | `Dict[str, Dict]` | `{"request_0": {...}}` |
| `resp.still` | All requests' responses | `Dict[str, str]` | `{"request_0": "Hello", "request_1": "..."}` |
| `resp.think` | All requests' reasoning | `Dict[str, str]` | `{"request_0": "reasoning..."}` |
| `resp.tools` | All requests' tool calls | `Dict[str, List[Dict]]` | `{"request_0": [{"function": {...}}]}` |
| `with resp as view` | Metadata view (real-time refresh) | `LiveBatchDict` context manager | `{"status": {...}, "usage": {...}}` |

**Streaming / Mixed streaming batch** (accessible during iteration, returns per-chunk increments for streaming requests in batch):

| Access Method | Return Content | Return Format | Example |
|-------------|-------------|-------------|---------|
| `chunk.still` | Current chunk increment | `str` | `"Y"` |
| `chunk.think` | Current chunk reasoning increment | `str` | `"Th"` |
| `chunk.tools` | Current chunk's `delta.tool_calls` increment | `List[Dict]` | `[]` |

**to_dict():** Converts response to dictionary, preserving specified fields; fields not declared in keep will generate warnings if retained:

```python
resp.to_dict()  # Default: keeps still/think/tools fields + metadata (status/usage)
resp.to_dict(errors=True, results=True)  # Keeps results/errors fields + metadata (status/usage)
```

### 2.3 Embeddings Call

### 2.3.1 Single Call

```python
resp = client.embeddings.create(input="Hello world")
print(resp.vectors)  # Embedding vector result
```

### 2.3.2 Embeddings Batch Call

```python
resp = client.embeddings.batch(
    input=["Hello", "world", "你好"]
)
```

#### 2.3.3 Embeddings Batch Response Structure

BatchEmbeddingResponse outer structure, where each response under `results[request_id]` is in **OpenAI standard Embeddings response structure**:

```python
{   
    "status": {"elapsed": "3.35s", "success_count": 1, "fail_count": 1, "total": 2},
    "batch_info": {"batch_size": 2, "batch_count": 2, "dimension": 1024},
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
    "results": {"request_0": {...}, "request_1": {...}}
    "errors": {"request_2": "error message"},
    "vectors": {"request_0": [...]}    # Mapping of all successful requests' request_id and embedding vectors
}
```

#### 2.3.4 Embeddings Batch Response Access

```python
resp = client.embeddings.batch(
    input=["Hello", "How's the weather today", "Who are you"]
)
```

**Access fields:**

| Access Method | Return Content | Return Format | Example |
|-------------|-------------|-------------|---------|
| `resp.status` | Real-time statistics | `Dict` | `{"total":2,"elapsed":"3.42s"}` |
| `resp.usage` | Token usage | `Dict[str, int]` | `{"total_tokens":10}` |
| `resp.batch_info` | Batch info | `Dict` | `{"batch_size":2,"batch_count":3,"dimension":1024}` |
| `resp.errors` | Failed request info | `Dict[str, str]` | `{"request_0":"error"}` |
| `resp.results` | Standard response | `Dict[str, Dict]` | `{"request_0": {...}}` |
| `resp.vectors` | Embedding vector representation | `Dict[str, List[float]]` | `{"request_0":[0.1,0.2,...]}` |
| `with resp as view` | Metadata view (real-time refresh) | `LiveEmbeddingDict` context manager | `{"status": {...}, "usage": {...}, "batch_info": {...}}` |

**to_dict():** Converts response to dictionary, preserving specified fields; fields not declared in keep will generate warnings if retained:

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
| `performance`    | `bool`  | `False`  | Pooled distribution, weighted by model throughput, fast/slow models don't block each other |

**batch_size**:
Only supported for batch Embeddings calls, defaults to adaptive calculation based on request count, manual configuration not recommended.

**max_concurrent, rps, performance**:
Non-streaming batch (`chat.batch(stream=False)`) uses adaptive scheduler by default; manual `max_concurrent` and `rps` override is not recommended.
Setting `performance=True` enables pooled distribution (requires `fallback_models`), weighted by model throughput; `max_concurrent` and `rps` cannot be configured.

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
- If a single request within the batch task contains unknown parameters, `drop_params="strict"` directly puts that request into the `errors` field without actually executing that request, and continues executing subsequent batch tasks.

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
| `model`             | `str`                           | -                               | Model name, see [Supported Models](#supported-models)|
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
   - MiniMax (M3): `True` → `{"type": "adaptive"}`, `False` → `{"type": "disabled"}`

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
| **MiniMax** | `stream_options`(native API), `group_id`(native API) |
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

`LangChainRunnable` inherits `BaseChatModel`, natively supports `(a)invoke`/`(a)stream`/`(a)batch` as well as `bind_tools`/`with_structured_output`.

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

### 5.5. DeepEval — Evaluation Testing

CNLLM output used for DeepEval evaluation:

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

Apache License 2.0 - See [LICENSE](LICENSE) file for details

### Contact

- GitHub Issues: <https://github.com/kanchengw/cnllm/issues>
- Author Email: <wangkancheng1122@163.com>
