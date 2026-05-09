# CNLLM Embedding 实现文档

## 1. 概述

CNLLM 支持文本 embedding 功能，可以将文本转换为向量表示。支持多厂商模型。

### 1.1 支持的模型

| 厂商 | 模型 | 向量维度 | 限制 |
|------|------|----------|------|
| GLM | `embedding-2` | 1024 | 单条最多 512 tokens，数组总长度不超过 8K |
| GLM | `embedding-3` | 2048 | 单条最多 3072 tokens，数组最大 64 条 |
| MiniMax | `embo-01` | - | 需指定 `type` 参数 (db/query) |
| DeepSeek | `deepseek-embedding` | 1536 | API 暂不可用 |

### 1.2 功能特性

| 特性 | 说明 |
|------|------|
| 同步/异步接口 | `create()` / `acreate()` |
| 单条/批量输入 | 自动识别 `str` / `List[str]` |
| 自定义 ID | 支持 `custom_ids` 参数 |
| OpenAI 兼容 | 返回标准 OpenAI embedding 格式 |
| 配置驱动 | YAML 配置文件管理 |
| 模型验证 | 自动从 `model_mapping.embedding` 验证支持列表 |

### 1.3 文件清单

```
cnllm/
├── core/
│   ├── __init__.py           # 导出 Embedding 相关类
│   ├── embedding.py          # EmbeddingResponder, BaseEmbeddingAdapter,
│   │                         # EmbeddingAdapter, AsyncEmbeddingAdapter,
│   │                         # EmbeddingsNamespace, AsyncEmbeddingsNamespace
│   └── vendor/
│       ├── minimax.py        # MiniMaxEmbeddingAdapter, MiniMaxEmbeddingResponder
│       └── glm.py            # GLMEmbeddingAdapter, GLMEmbeddingResponder
├── utils/
│   └── accumulators/
│       └── embedding_accumulator.py  # EmbeddingResponse 类
└── entry/
    ├── client.py             # CNLLM 添加 embeddings 属性
    └── async_client.py       # AsyncCNLLM 添加 embeddings 属性

configs/
├── minimax/
│   ├── request_minimax.yaml  # 合并后的配置 (chat + embedding)
│   └── response_minimax.yaml # 合并后的配置 (chat + embedding)
├── glm/
│   ├── request_glm.yaml      # 合并后的配置 (chat + embedding)
│   └── response_glm.yaml     # 合并后的配置 (chat + embedding)
└── deepseek/
    ├── request_deepseek.yaml # 合并后的配置 (chat + embedding)
    └── response_deepseek.yaml
```

## 2. 配置设计

### 2.1 合并配置文件

Embedding 配置与 Chat 配置合并到同一个 yml 文件中，通过 `adapter` 标识区分。

**目录结构**：
- `request_{vendor}.yaml` - 请求配置（chat + embedding）
- `response_{vendor}.yaml` - 响应配置（chat + embedding）

### 2.2 请求配置结构

```yaml
request:
  method: "POST"
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${api_key}"

required_fields:
  api_key:
    adapter: [chat, embedding]  # chat 和 embedding 都使用
    body: "__skip__"             # 不加入请求体
  model:
    adapter: [chat, embedding]
  input:
    adapter: [embedding]          # 仅 embedding 使用

one_of:
  messages_or_prompt:
    adapter: [chat]              # 仅 chat 使用
    messages: ""
    prompt: ""

optional_fields:
  base_url:
    body: "__skip__"
    chat:
      default: "https://..."
      text: "path/to/chat"
    embedding:
      default: "https://..."
      path: "/path/to/embeddings"
  type:                          # 厂商特有参数
    adapter: [embedding]
    default: "db"                # 默认值
  dimensions:
    adapter: [embedding]

model_mapping:
  chat:
    glm-4: "glm-4"
  embedding:
    embedding-2: "embedding-2"
    embedding-3: "embedding-3"
```

### 2.3 响应配置结构

```yaml
fields:                          # Chat 字段
  id: "id"
  content: "choices[0].message.content"
  # ...

embedding_fields:                # Embedding 字段
  embedding: "data[0].embedding"
  index: "data[0].index"

defaults:                        # Chat 默认值
  object: "chat.completion"

embedding_defaults:              # Embedding 默认值
  object: "list"
  index: 0
```

### 2.4 adapter 标识

| 值 | 说明 |
|----|------|
| `[chat]` | 仅 Chat 使用 |
| `[embedding]` | 仅 Embedding 使用 |
| `[chat, embedding]` | Chat 和 Embedding 都使用 |

### 2.5 配置加载优先级

1. 优先加载合并后的配置文件 `request_{vendor}.yaml`
2. 如果不存在，回退到 `request_embedding_{vendor}.yaml`（兼容性）

## 3. 接口设计

### 3.1 客户端初始化

```python
from cnllm import CNLLM

# 使用 embedding 模型初始化
client = CNLLM(model="embedding-2", api_key="xxx")

# 或使用 chat 模型，通过方法参数指定 embedding 模型
client = CNLLM(model="glm-4", api_key="xxx")
```

### 3.2 接口数量：2 个

```python
# 同步接口：自动识别 单条/批量
client.embeddings.create(input: str | list[str])

# 异步接口：自动识别 单条/批量
await client.embeddings.acreate(input: str | list[str])
```

### 3.3 方法参数

```python
def create(
    self,
    input: Union[str, List[str]],      # 必填：单条文本或文本列表
    model: str = None,                 # 可选：指定 embedding 模型
    custom_ids: List[str] = None,     # 可选：自定义 ID
    **kwargs                          # 厂商特有参数 (如 dimensions, type)
) -> Union[Dict[str, Any], EmbeddingResponse]
```

## 4. 响应结构

### 4.1 单条输入 → OpenAI 标准结构

**完整结构**:
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, 0.2, ...],
            "index": 0
        }
    ],
    "model": "embedding-2",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

```python
# print 输出:
print(result)
# {"object": "list", "data": [{"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}], ...}

# 字段访问:
print(result["object"])        # "list"
print(result["data"][0]["embedding"])  # [0.1, 0.2, ...]
print(result["usage"]["total_tokens"])  # 5
```

### 4.2 批量输入（列表）→ EmbeddingResponse 结构

**完整结构**:
```python
{
    "status": {
        "success_count": 2,
        "fail_count": 0,
        "total": 2,
        "elapsed": "0.35s"
    },
    "usage": {
        "prompt_tokens": 11,
        "total_tokens": 11
    },
    "batch_info": {
        "batch_size": 2,
        "batch_count": 1,
        "dimension": 1024
    },
    "vectors": {
        "request_0": [0.1, 0.2, ...],
        "request_1": [0.3, 0.4, ...]
    },
    "results": {
        "request_0": {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
            "model": "embedding-2",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        },
        "request_1": {
            "object": "list",
            "data": [{"embedding": [0.3, 0.4, ...], "index": 0}],
            "model": "embedding-2",
            "usage": {"prompt_tokens": 6, "total_tokens": 6}
        }
    }
}
```

```python
# print 输出（简洁统计，不显示大文本）:
print(result)
# EmbeddingResponse(status={'success_count': 2, 'fail_count': 0, 'total': 2, 'elapsed': '0.35s'}, usage={'prompt_tokens': 11, 'total_tokens': 11}, batch_info={'batch_size': 2, 'batch_count': 1, 'dimension': 1024})

# 向量访问（推荐）:
result.vectors["request_0"]      # [0.1, 0.2, ...]
result.vectors[0]                # [0.1, 0.2, ...]

# 标准响应访问:
result.results["request_0"]      # {"object": "list", "data": [...], ...}
result.results[0]                # 同上
result.results["doc_001"]        # 自定义 request_ids 时

# 合并统计:
result.usage                     # {"prompt_tokens": 11, "total_tokens": 11}
result.batch_info                # {"batch_size": 2, "batch_count": 1, "dimension": 1024}
```

### 4.3 EmbeddingResponse 类

**位置**: `cnllm/core/accumulators/embedding_accumulator.py`

**统计属性**（无阻塞，实时可用）:

| 属性 | 类型 | 说明 |
|------|------|------|
| `status` | `Dict` | 统计信息: success_count, fail_count, total, elapsed |
| `status["success_count"]` | `int` | 成功数 |
| `status["fail_count"]` | `int` | 失败数 |
| `status["total"]` | `int` | 请求总数 |
| `status["elapsed"]` | `str` | 耗时（如 "0.35s"） |
| `usage` | `Dict[str, int]` | Token 用量汇总 (prompt_tokens, total_tokens) |
| `batch_info` | `Dict` | 批量信息 (batch_size, batch_count, dimension) |
| `batch_info["dimension"]` | `int` | 向量维度 |
| `vectors` | `Dict[str, list]` | 向量数据 (request_id → embedding list) |
| `errors` | `Dict[str, str]` | 错误信息 (request_id → error_msg) |
| `results` | `Dict[str, Dict]` | 完整响应 (需 keep=["*"] 保留) |

**数据属性**（支持 int/str 索引）:

| 属性 | 类型 | 说明 |
|------|------|------|
| `vectors` | `Dict[str, List[float]]` | request_id → 向量（推荐使用，默认保留） |
| `results` | `Dict[str, Dict]` | request_id → OpenAI 标准 embedding 响应（迭代后默认释放） |

**方法**:

| 方法 | 说明 |
|------|------|
| `to_dict()` | 转换为 dict，默认返回 vectors + 统计字段 |
| `__repr__()` | 显示如 `EmbeddingResponse(success=[...], fail=[...], request_counts={...}, elapsed=..., usage={...}, batch_info={...})` |
| `__getitem__(key)` | 支持整数索引 `response[0]` 或字符串 `response["request_0"]` |
| `for r in response` | 实时迭代，每次 yield 当前快照（迭代结束后释放非 keep 字段） |

**to_dict() 参数**:
```python
result.to_dict()                          # 默认：返回 vectors + 统计字段
result.to_dict(results=True)              # 返回 results + 统计字段
result.to_dict(stats=False)               # 只返回 vectors，不含统计字段
```

**keep 参数** — 控制迭代后保留的字段：

Embeddings 批量默认 `keep={"vectors"}`，即 `for` 循环迭代结束后只保留向量，`results` 被释放以节省内存。

```python
# 默认行为：迭代后 vectors 保留，results 释放
resp = client.embeddings.batch(input=["文本1", "文本2"])
for _ in resp:
    pass
resp.vectors["request_0"]    # 可访问 ✓
resp.results["request_0"]    # 已释放（访问会触发警告）

# 自定义 keep：迭代后同时保留 vectors 和 results
resp = client.embeddings.batch(
    input=["文本1", "文本2"],
    keep={"vectors", "results"}
)
for _ in resp:
    pass
resp.vectors["request_0"]    # 可访问 ✓
resp.results["request_0"]    # 可访问 ✓
```

## 5. 使用示例

### 5.1 单条文本

```python
from cnllm import CNLLM

client = CNLLM(model="embedding-2", api_key="xxx")

result = client.embeddings.create(input="要向量化的文本")

vector = result["data"][0]["embedding"]
print(f"向量维度: {len(vector)}")  # 1024
```

### 5.2 指定 embedding 模型

```python
client = CNLLM(model="glm-4", api_key="xxx")  # chat 模型

# 使用 embedding-3 (2048维)
result = client.embeddings.create(
    model="embedding-3",
    input="要向量化的文本"
)
print(f"向量维度: {len(result['data'][0]['embedding'])}")  # 2048
```

### 5.3 批量文本（自动 ID）

```python
result = client.embeddings.create(
    model="embedding-2",
    input=["文本1", "文本2", "文本3"]
)

print(result)  # EmbeddingResponse(request_counts={'total': 3, 'dimension': 1024}, elapsed=0.35)
print(f"总数: {result.total}, 维度: {result.dimension}")
print(f"耗时: {result.elapsed:.2f}s")

vec0 = result[0]["data"][0]["embedding"]
vec1 = result["request_1"]["data"][0]["embedding"]
```

### 5.4 批量文本（自定义 ID）

```python
result = client.embeddings.create(
    model="embedding-2",
    input=["文本1", "文本2", "文本3"],
    custom_ids=["doc_001", "doc_002", "doc_003"]
)

vec0 = result["doc_001"]["data"][0]["embedding"]
```

### 5.5 MiniMax 特有参数

```python
# MiniMax 需要 type 参数
result = client.embeddings.create(
    model="embo-01",
    input="要向量化的文本",
    type="db"  # 或 "query"
)
```

### 5.6 GLM 特有参数

```python
# GLM embedding-3 支持 dimensions 参数
result = client.embeddings.create(
    model="embedding-3",
    input="要向量化的文本",
    dimensions=512  # 可选: 256, 512, 1024, 2048
)
```

### 5.7 异步客户端

```python
import asyncio
from cnllm import AsyncCNLLM

async def main():
    async with AsyncCNLLM(model="embedding-2", api_key="xxx") as client:
        result = await client.embeddings.acreate(
            model="embedding-2",
            input=["文本1", "文本2"]
        )
        print(f"成功: {result.success_count}/{result.total}")

asyncio.run(main())
```

## 6. 厂商适配器

### 6.1 适配器注册

| 厂商 | Chat Adapter | Embedding Adapter |
|------|-------------|-------------------|
| GLM | `GLMAdapter` | `GLMEmbeddingAdapter` |
| MiniMax | `MiniMaxAdapter` | `MiniMaxEmbeddingAdapter` |
| DeepSeek | `DeepSeekAdapter` | (待实现) |

### 6.2 特殊处理

**MiniMax**:
- 响应格式不同：使用 `vectors` 字段而非 `data[0].embedding`
- 需要 `MiniMaxEmbeddingResponder.to_openai_format()` 转换
- 请求需要 `type` 参数

**GLM**:
- 响应格式与 OpenAI 兼容
- 可使用基类 `EmbeddingResponder.to_openai_format()`
- 支持 `dimensions` 参数

## 7. 错误处理

Embedding 批量请求是"全成功或全失败"模式，出错时直接抛出异常。

```python
result = client.embeddings.create(
    model="embedding-2",
    input=["文本1", "文本2", "文本3"]
)

# 出错时 results 为空，dimension 为 0
if result.dimension == 0:
    print("请求失败，results 为空")
```

### 7.3 不支持的模型

```python
try:
    result = client.embeddings.create(
        model="unsupported-model",
        input="text"
    )
except ModelNotSupportedError as e:
    print(f"不支持的模型: {e}")
```

## 8. 测试

```bash
# 运行 GLM embedding 测试
python test_embedding/test_glm_embedding.py

# 运行 MiniMax embedding 测试
python test_embedding/test_minimax_embedding.py
```

## 9. 与 Chat Batch 对比

| 特性 | Chat Batch | Embedding Batch |
|------|------------|-----------------|
| 响应类 | `BatchResponse` | `EmbeddingResponse` |
| 统计字段 | success/fail/request_counts/elapsed/usage | 同 |
| think/still/tools/raw | 支持 | 不需要 |
| vectors | 不支持 | 支持 |
| batch_info | 不支持 | 支持 (batch_size, batch_count, dimension) |
| 流式支持 | 支持 | 不需要 |
| 整数索引 | 支持 | 支持 |
| 自定义 request_id | 支持 | 支持 |
| 默认 keep (迭代后保留) | 全部保留 | 仅保留 vectors |
| `for` 循环实时迭代 | ✅ request by request | ✅ request by request |
| 配置方式 | 独立 yml | 合并到 chat yml |
| 厂商特有参数 | 通过 transform | 通过 adapter + default |
