# CNLLM 批量调用

## 1. 功能特性

CNLLM 提供统一的批量调用接口，支持：

| 特性 | 说明 |
|------|------|
| 统一接口 | `batch()` / `abatch()` 方法，通过 `stream` 参数控制流式/非流式 |
| 并发控制 | `max_concurrent` 参数控制最大并发数 |
| 错误隔离 | 单个请求失败不影响其他请求 |
| 进度回调 | `callbacks` 参数监听任务进度 |
| 停止控制 | `stop_on_error` 参数遇错时停止其他任务 |
| 混合流式 | 每个请求可独立控制 `stream`（per-request 级别） |
| per-request 超时 | 每个请求可独立设置 `timeout` |
| per-request 重试 | 每个请求可独立设置 `max_retries` / `retry_delay` |

## 2. 竞品对比

| 特性 | OpenAI SDK | LangChain | LiteLLM | CNLLM |
|------|------------|-----------|---------|-------|
| 批量调用 | ✅ batch | ✅ batch | ✅ batch | ✅ batch |
| 并发控制 | ✅ max_concurrent | ✅ max_concurrent | ✅ concurrency_limit | ✅ max_concurrent |
| 异步批量 | ✅ async batch | ✅ abatch | ✅ abatch | ✅ abatch |
| 进度回调 | ❌ | ✅ with_config | ⚠️ | ✅ callbacks |
| 错误隔离 | ⚠️ | ✅ | ⚠️ | ✅ stop_on_error |
| 流式批量 | ❌ | ⚠️ | ✅ | ✅ stream=True |
| per-request 流式 | ❌ | ❌ | ❌ | ✅ per-request stream |
| per-request 超时 | ❌ | ❌ | ❌ | ✅ per-request timeout |

## 3. 架构

```mermaid
flowchart TD
    subgraph 用户层["用户层"]
        U1[client.chat.batch] --> U11[stream=False]
        U1 --> U12[stream=True]
        U2[client.chat.abatch] --> U21[stream=False]
        U2 --> U22[stream=True]
    end

    subgraph 批量调度器["批量调度器"]
        B1[BatchScheduler] --> B11[非流式批量]
        B2[StreamBatchScheduler] --> B21[流式批量]
        B3[AsyncBatchScheduler] --> B31[异步非流式批量]
        B4[AsyncStreamBatchScheduler] --> B41[异步流式批量]
    end

    subgraph 执行层["执行层"]
        E1[create<br/>同步调用]
        E2[acreate<br/>异步调用]
    end

    U11 --> B1
    U12 --> B2
    U21 --> B3
    U22 --> B4

    B1 --> E1
    B2 --> E1
    B3 --> E2
    B4 --> E2
```

### 参数分类

所有 `kwargs` 在 `batch()` 入口处按 `BATCH_LEVEL_KEYS` 集合分为两组：

```python
BATCH_LEVEL_KEYS = frozenset({
    "max_concurrent", "rps", "stop_on_error",
    "callbacks", "custom_ids", "requests",
})

per_request_defaults = {k: v for k, v in kwargs.items() if k not in BATCH_LEVEL_KEYS}
batch_level_kwargs  = {k: v for k, v in kwargs.items() if k in BATCH_LEVEL_KEYS}
```

| 类别 | 参数 | 性质 | 处理方式 |
|------|------|------|---------|
| **Per-Request** | `prompt`/`messages`、`thinking`、`tools`、`temperature`、`max_tokens`、`top_p`、`stop`、`model`、`stream`、`timeout`、`max_retries`、`retry_delay` | 描述「发给 API 的数据」 | 进入请求 dict → create() → YAML 验证 |
| **Batch-Level** | `max_concurrent`、`rps`、`stop_on_error`、`callbacks`、`custom_ids` | 描述「如何调度这些请求」 | **不进请求 dict** → 直接用于 BatchScheduler，不传给 create() |

**关键原则**：Per-Request 参数可在 `requests` 列表中每个请求独立配置，也可作为全局参数传入 `batch()`，未配置时自动继承全局默认值（逐字段继承）。

### 重试机制

`_execute_with_retry` 已移除，不再有 scheduler 层重试循环。重试由 `create()` 内 HTTP 层处理：

- Per-request 中配置 `max_retries`/`retry_delay` → 传递到 `create()` → HTTP 层重试
- `batch()` 全局参数中配置 `max_retries`/`retry_delay` → 作为 per-request 的 fallback 填充
- YAML 中配置的默认值（如 `timeout: 60.0`, `max_retries: 3`）→ 最终兜底

## 4. 接口定义

### 4.1 同步批量 `client.chat.batch()`

```python
def batch(
    self,
    requests: list = None,
    *,
    prompt: list = None,
    messages: list = None,
    stream: bool = False,
    max_concurrent: int = 3,
    timeout: Optional[float] = None,
    stop_on_error: bool = False,
    callbacks: Optional[List[Callable]] = None,
    custom_ids: Optional[List[str]] = None,
    **kwargs,  # 其他 per-request 参数：thinking, tools, temperature, max_retries ...
) -> Union[BatchResponse, BatchStreamAccumulator]:
    """
    批量执行多个请求（同步）

    Args:
        requests: 请求字典列表，支持独立参数 {prompt/messages, stream, tools, ...}
        prompt: prompt 字符串列表（向后兼容）
        messages: messages 列表的列表（向后兼容）
        stream: 是否使用流式处理，默认 False
        max_concurrent: 最大并发数，默认 3
        timeout: 单个请求超时（秒），默认 None
        stop_on_error: 遇到错误是否停止，默认 False
        callbacks: 进度回调列表，默认 None
        custom_ids: 自定义请求 ID 列表，默认 None

    Returns:
        stream=False: BatchResponse
        stream=True:  BatchStreamAccumulator（可迭代 chunk）
    """
```

### 4.2 异步批量 `client.chat.abatch()`

```python
async def abatch(
    self,
    requests: list = None,
    *,
    prompt: list = None,
    messages: list = None,
    stream: bool = False,
    max_concurrent: int = 3,
    timeout: Optional[float] = None,
    stop_on_error: bool = False,
    callbacks: Optional[List[Callable]] = None,
    custom_ids: Optional[List[str]] = None,
    **kwargs,
) -> Union[BatchResponse, AsyncBatchStreamAccumulator]:
    """
    批量执行多个请求（异步）
    """
```

## 5. 参数说明

| 参数 | 类型 | 默认值 | 层级 | 说明 |
|------|------|--------|------|------|
| `requests` | `list[dict]` | - | per-request | 请求列表，每项可含 `prompt`/`messages` 及独立参数 |
| `prompt` | `list[str]` | - | 向后兼容 | prompt 字符串列表 |
| `messages` | `list[list]` | - | 向后兼容 | messages 列表的列表 |
| `stream` | `bool` | `False` | per-request | 是否使用流式处理（也可 per-request 独立控制） |
| `max_concurrent` | `int` | `3` | batch-level | 最大并发执行数 |
| `timeout` | `float` | `None` | per-request | 单请求超时时间（秒），YAML 默认兜底 |
| `max_retries` | `int` | `None` | per-request | 重试次数，YAML 默认兜底 |
| `retry_delay` | `float` | `None` | per-request | 重试间隔，YAML 默认兜底 |
| `stop_on_error` | `bool` | `False` | batch-level | 遇错误时停止其他任务 |
| `callbacks` | `List[Callable]` | `None` | batch-level | 进度回调函数列表 |
| `custom_ids` | `List[str]` | `None` | batch-level | 自定义请求 ID 列表 |

### 参数继承规则

`requests` 中每个 dict 的参数与 `batch()` 全局参数合并：

```python
# 每个 per-request dict 缺失的字段从全局参数继承
defaults = {k: v for k, v in per_request_defaults.items() if k not in per_request}
per_request = {**defaults, **per_request}  # 全局 → per-request 覆盖
```

Batch-Level 参数（`max_concurrent`, `stop_on_error`, `callbacks`, `custom_ids`）**不进入**请求 dict，不会传给 `create()`，不被 per-request 继承。

## 6. 返回值

### 6.1 外层结构

```python
# print 输出:
print(result)
# BatchResponse(success=2, fail=0, total=2, elapsed=0.35s)

print(result.results)
# BatchResults(count=2, ids=['request_0', 'request_1'])
```

### 6.2 非流式批量响应格式

```python
{
    "success": ["request_0", "request_1"],  # 成功的 request_id 列表
    "errors": [],                                 # 失败的 request_id 列表
    "request_counts": {
        "success_count": 2,
        "fail_count": 0,
        "total": 2
    },
    "elapsed": 2.1,
    "results": {              # {request_id: OpenAI 格式 dict}
        "request_0": {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "created": 1742112345,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "回复内容"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9
            }
        },
        "request_1": {
            "error": "参数错误"
        }
    },
    "think": {"request_0": "推理内容", "request_1": "推理内容"},
    "still": {"request_0": "回复内容", "request_1": "回复内容"},
    "tools": {"request_0": {0: {...}}, "request_1": {0: {...}}},
    "raw": {"request_0": {...}, "request_1": {...}}
}
```

### 6.3 流式批量响应格式

流式批量 yield 标准 OpenAI 格式 chunk：

```python
# 开始 chunk:
{"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}

# 中间 chunk:
{"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "你"}, "finish_reason": null}]}

# 结尾 chunk:
{"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
```

### 6.4 响应访问

```python
# 统计字段:
result.total              # int: 总数
result.success_count      # int: 成功数
result.fail_count         # int: 失败数
result.elapsed            # float: 耗时
result.success            # List[str]: 成功的 request_id 列表
result.fail               # List[str]: 失败的 request_id 列表
result.request_counts     # Dict: {"success_count": ..., "fail_count": ..., "total": ...}

# 累积字段（支持索引和 key 访问）:
result.think[0]                  # 推理内容
result.think["request_0"]        # 同上
result.still[0]                  # 回复内容
result.still["request_0"]        # 同上
result.tools[0]                  # 工具调用 {0: {...}}
result.tools["request_0"]        # 同上
result.raw[0]                    # 原始数据
result.raw["request_0"]          # 同上

# 响应访问:
result.results["request_0"]      # 单个结果 dict
result.results[0]                # 同上
result["request_0"]              # 同上
result[0]                        # 同上

# 遍历:
for request_id, item in result.results.items():
    if "error" in item:
        print(f"失败: {item['error']}")
    else:
        print(f"成功: {item['choices'][0]['message']['content']}")

# 转换为标准 JSON:
result.to_dict()
result.to_dict(stats=True, think=True, still=True, tools=True, raw=True)
```

## 7. 使用示例

### 7.1 同步批量（非流式）

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2.5", api_key="xxx")

results = client.chat.batch(
    prompt=["你好", "今天天气怎么样", "你是谁"],
    thinking=True,
)

print(f"成功: {results.success_count}/{results.total}")
print(f"耗时: {results.elapsed:.2f}s")

for cid, item in results.results.items():
    if "error" in item:
        print(f"{cid} 失败: {item['error']}")
    else:
        print(f"{cid} 回复: {item['choices'][0]['message']['content']}")
```

### 7.2 同步批量（流式）

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2.5", api_key="xxx")

acc = client.chat.batch(
    prompt=["数到3", "说你好", "介绍自己"],
    stream=True,
)

for chunk in acc:
    print(chunk)  # OpenAI 标准 chunk

# 流中访问累积数据:
print(acc.still[0])   # 实时累积
print(acc.think[0])   # 实时累积
print(f"成功: {acc.success_count}")
```

### 7.3 异步批量（非流式）

```python
from cnllm import AsyncCNLLM
import asyncio

async def main():
    client = AsyncCNLLM(model="minimax-m2.5", api_key="xxx")
    results = await client.chat.abatch(
        prompt=["你好", "今天天气怎么样", "你是谁"],
    )
    print(f"成功: {results.success_count}/{results.total}")
    for cid, item in results.results.items():
        if "error" not in item:
            print(f"{cid}: {item['choices'][0]['message']['content']}")
    await client.aclose()

asyncio.run(main())
```

### 7.4 异步批量（流式）

```python
from cnllm import AsyncCNLLM
import asyncio

async def main():
    client = AsyncCNLLM(model="minimax-m2.5", api_key="xxx")
    async for chunk in client.chat.abatch(
        prompt=["数到3", "说你好"],
        stream=True,
    ):
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        print(content, end="", flush=True)
    await client.aclose()

asyncio.run(main())
```

### 7.5 per-request 混合流式

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2.5", api_key="xxx")

# 混合 stream: 部分请求流式，部分非流式
resp = client.chat.batch(requests=[
    {"prompt": "hello", "stream": True},   # 流式调用
    {"prompt": "hi"},                       # 非流式调用
    {"prompt": "good", "stream": True},     # 流式调用
])

# 迭代时 req-by-req 实时累积
for r in resp:
    print(f"已完成: {r.success_count}/{r.total}")

# 循环外完整数据
print(resp.still)   # 3 个请求的 still 全部正确
print(resp.think)   # 3 个请求的 think 全部正确
```

### 7.6 per-request 独立参数（请求覆盖全局）

```python
resp = client.chat.batch(
    requests=[
        {"prompt": "北京天气怎么样", "tools": tool_1, "stream": True},           # 独有参数覆盖全局
        {"prompt": "1+1等于多少", "tools": tool_2, "thinking": False},            # 不继承任何全局参数
        {"prompt": "广州天气怎么样"},                                              # 全部继承全局默认值
    ],
    tools=[default_tool],
    thinking=True,
    max_concurrent=2,  # batch-level，不被单个请求继承
)
```

## 8. 高级选项

### 8.1 并发控制

```python
results = client.chat.batch(
    prompt=["文本1", "文本2", "文本3"],
    max_concurrent=5,      # 最多5个并发
)
```

### 8.2 自定义请求 ID

```python
results = client.chat.batch(
    prompt=["文本1", "文本2", "文本3"],
    custom_ids=["doc_001", "doc_002", "doc_003"],
)

results["doc_001"]        # 获取 doc_001 的响应
results.think["doc_002"]  # 获取 doc_002 的推理内容
```

### 8.3 进度回调

```python
def on_complete(request_id, status):
    print(f"[{request_id}] {status}")

results = client.chat.batch(
    prompt=["文本1", "文本2", "文本3"],
    callbacks=[on_complete],
)
```

### 8.4 遇错停止

```python
results = client.chat.batch(
    prompt=["文本1", "文本2", "文本3"],
    stop_on_error=True,
)
```

### 8.5 per-request 超时和重试

```python
# 每个请求独立超时和重试策略
results = client.chat.batch(requests=[
    {"prompt": "快处理完成", "timeout": 10},
    {"prompt": "很慢的任务", "timeout": 60, "max_retries": 2},
])
```
