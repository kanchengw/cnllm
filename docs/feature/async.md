# CNLLM 异步支持

## 1. 技术选型

**httpx**：统一同步和异步 HTTP 客户端

| 特性 | 说明 |
|------|------|
| 同步支持 | `httpx.Client` - 与 requests API 高度相似 |
| 异步支持 | `httpx.AsyncClient` - 真正的异步 HTTP |
| 连接池 | 内置连接池管理，比 requests 更现代 |
| Streaming | `stream()` / `stream()` 支持 SSE |
| 社区 | well-maintained，~4M downloads/week |

## 2. 架构设计

### 2.1 设计理念

CNLLM 采用 **OpenAI SDK 风格的双客户端设计**：

| 客户端 | 类 | 用途 | 调用方式 |
| ------ | --- | ---- | -------- |
| 同步客户端 | `CNLLM` | 同步场景 | `client.chat.create(...)` |
| 异步客户端 | `asyncCNLLM` | 异步场景 | `await client.chat.create(...)` |

### 2.2 统一 stream 参数

无论是同步还是异步客户端，`stream=True` 参数都控制返回类型：

| 客户端 | `stream=False` | `stream=True` | 批量 `stream=True` |
| ------ | -------------- | ------------- | ----------------- |
| 同步 `CNLLM` | `dict` | `StreamResultAccumulator` | `Iterator[dict]` |
| 异步 `asyncCNLLM` | `dict` | `AsyncStreamResponse` | `AsyncIterator[dict]` |

### 2.3 httpx 统一方案

CNLLM 使用 **httpx** 作为 HTTP 客户端，同时支持同步和异步：

```python
# 同步请求
def post(self, ...):
    with httpx.Client() as client:
        response = client.post(url, json=payload, headers=headers)
        return response.json()

# 异步请求
async def apost(self, ...):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        return response.json()

# 同步流式
def post_stream(self, ...):
    with client.stream("POST", url=url, ...) as response:
        for line in response.iter_lines():
            yield line.encode('utf-8')

# 异步流式
async def apost_stream(self, ...):
    async with client.stream("POST", url=url, ...) as response:
        async for line in response.aiter_lines():
            yield line.encode('utf-8')
```

### 2.4 BaseHttpClient 统一客户端

```
┌─────────────────────────────────────────┐
│           BaseHttpClient (httpx)        │
├─────────────────────────────────────────┤
│  _sync_client: httpx.Client             │
│  _async_client: httpx.AsyncClient      │
├─────────────────────────────────────────┤
│  post()         - 同步非流式              │
│  post_stream() - 同步流式                │
│  apost()        - 异步非流式              │
│  apost_stream() - 异步流式               │
└─────────────────────────────────────────┘
```

### 2.5 异常映射

| requests | httpx |
|----------|-------|
| `requests.exceptions.Timeout` | `httpx.TimeoutException` |
| `requests.exceptions.ConnectionError` | `httpx.ConnectError` |
| `requests.exceptions.InvalidURL` | `httpx.InvalidURL` |
| `requests.exceptions.HTTPError` | `httpx.HTTPStatusError` |
| `requests.exceptions.RequestException` | `httpx.RequestError` |

## 3. 接口定义

### 3.1 同步 vs 异步接口对比

```python
# ========== 同步客户端 ==========
from cnllm import CNLLM

client = CNLLM(model="deepseek-chat", api_key="xxx")

# 非流式
resp = client.chat.create(messages=[...])
print(resp)

# 流式
resp = client.chat.create(messages=[...], stream=True)
for chunk in resp:
    print(chunk)


# ========== 异步客户端 ==========
from cnllm import asyncCNLLM

async_client = asyncCNLLM(model="deepseek-chat", api_key="xxx")

# 非流式
resp = await async_client.chat.create(messages=[...])
print(resp)

# 流式（统一接口，stream 参数控制返回类型）
resp = await async_client.chat.create(messages=[...], stream=True)
async for chunk in resp:
    print(chunk)
```

### 3.2 BaseHttpClient 同步方法

```python
class BaseHttpClient:
    def post(self, path: str, payload: Dict, extra_headers: Dict = None) -> Dict:
        """同步非流式请求"""

    def post_stream(self, path: str, payload: Dict, extra_headers: Dict = None) -> Iterator[bytes]:
        """同步流式请求"""
        with client.stream("POST", url=url, headers=headers, json=payload) as response:
            for line in response.iter_lines():
                yield line.encode('utf-8')

    def close(self):
        """关闭同步客户端"""
```

### 3.3 BaseHttpClient 异步方法

```python
class BaseHttpClient:
    async def apost(self, path: str, payload: Dict, extra_headers: Dict = None) -> Dict:
        """异步非流式请求"""

    async def apost_stream(self, path: str, payload: Dict, extra_headers: Dict = None) -> AsyncIterator[bytes]:
        """异步流式请求"""
        async with client.stream("POST", url=url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                yield line.encode('utf-8')

    async def aclose(self):
        """关闭异步客户端"""
```

### 3.4 AsyncStreamAccumulator 异步流式封装

`AsyncStreamAccumulator` 封装异步迭代器，实现实时流式处理：

```python
class AsyncStreamAccumulator:
    """异步流式响应，支持 async for 实时迭代"""

    def __init__(self, async_iterator, adapter):
        self._raw_iterator = async_iterator
        self._adapter = adapter
        self._chunks = []
        self._seen_tool_call_indices = set()
        self._seen_choice_indices = set()
        self._done = False

        if self._adapter._raw_response is None:
            self._adapter._raw_response = {}
        self._adapter._raw_response["chunks"] = []
        self._adapter._cnllm_extra = {}

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        try:
            raw_chunk = await self._raw_iterator.__anext__()
        except StopAsyncIteration:
            self._done = True
            self._add_done_marker()
            raise

        result = self._adapter._to_openai_stream_format(raw_chunk)

        # 从原生 chunk 提取额外字段（reasoning_content, content, tool_calls）
        self._accumulate_extra_fields(raw_chunk)

        self._post_process_chunk(result)
        self._chunks.append(result)
        self._adapter._raw_response["chunks"].append(result)
        return result
```

## 4. 流式处理

### 4.1 同步流式

```python
for line in client.http_client.post_stream(path, payload):
    # line 是 bytes
    chunk = json.loads(line.decode('utf-8'))
    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    print(content, end="", flush=True)
```

### 4.2 异步流式

```python
async for line in await client.http_client.apost_stream(path, payload):
    # line 是 bytes
    chunk = json.loads(line.decode('utf-8'))
    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    print(content, end="", flush=True)
```

## 5. 资源管理

### 5.1 异步资源管理

```python
# 方式1：手动关闭（适合长生命周期，如 FastAPI 全局客户端）
client = asyncCNLLM(model="deepseek-chat", api_key="xxx")
try:
    result = await client.chat.create("你好")
finally:
    await client.aclose()

# 方式2：上下文管理器（推荐，适合短生命周期）
async with asyncCNLLM(model="deepseek-chat", api_key="xxx") as client:
    result = await client.chat.create("你好")
# 自动调用 aclose
```

## 6. 使用示例

### 6.1 异步客户端创建

```python
from cnllm import asyncCNLLM

client = asyncCNLLM(model="deepseek-chat", api_key="xxx")

# 或使用上下文管理器
async with asyncCNLLM(model="deepseek-chat", api_key="xxx") as client:
    result = await client.chat.create("你好")
    print(result)
```

### 6.2 异步流式

```python
async with asyncCNLLM(model="deepseek-chat", api_key="xxx") as client:
    async for chunk in await client.chat.create("写一首诗", stream=True):
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        print(content, end="", flush=True)
```

### 6.3 异步批量

```python
async with asyncCNLLM(model="deepseek-chat", api_key="xxx") as client:
    results = await client.chat.batch([
        "问题1",
        "问题2",
        "问题3",
    ])
    print(f"成功: {results.success_count}/{results.total}")
```

### 6.4 异步 Embedding 单条

```python
async with asyncCNLLM(model="embedding-2", api_key="xxx") as client:
    result = await client.embeddings.create(input="要向量化的文本")
    # 返回: Dict (OpenAI 标准格式)
    embedding = result["data"][0]["embedding"]
    print(f"向量维度: {len(embedding)}")
```

### 6.5 异步 Embedding 批量

```python
async with asyncCNLLM(model="embedding-2", api_key="xxx") as client:
    results = await client.embeddings.batch(inputs=[
        "文本1",
        "文本2",
        "文本3",
    ])
    # 返回: EmbeddingResponse
    for request_id, result in results.results.items():
        embedding = result["data"][0]["embedding"]
        print(f"{request_id}: 维度={len(embedding)}")
```

## 7. 注意事项

1. **异步客户端需要关闭**：使用 `await client.aclose()` 或 `async with` 上下文管理器
2. **异步迭代器**：使用 `async for` 而非 `for`
3. **httpx 流式必须用上下文管理器**：`with client.stream()` 或 `async with client.stream()`
4. **iter_lines 返回值**：httpx 返回 str，需要手动 `encode('utf-8')`
