# CNLLM 架构与设计文档

## 1. 架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/core/client.py)                  │
├─────────────────────────────────────────────────────────────┤
│  三种调用入口：                                                │
│  - 极简入口: client("prompt")                                 │
│  - 标准入口: client.chat.create(prompt="...")                 │
│  - 完整入口: client.chat.create(messages=[...])               │
│                                                             │
│  响应入口：                                                   │
│  - client.chat.still  → 纯净文本                             │
│  - client.chat.raw   → 原始响应（含平台特有字段）               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    模型适配层 (Model Adapter)                 │
│              (cnllm/adapters/{厂商}/chat.py)                 │
├─────────────────────────────────────────────────────────────┤
│  - 厂商协议转换                                               │
│  - 参数验证                                                  │
│  - 存储 raw 响应到 adapter._raw_response                      │
│  - 返回 OpenAI 格式响应                                       │
│                                                             │
│  如：MiniMaxAdapter                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                           │
│                     (cnllm/core/base.py)                    │
├─────────────────────────────────────────────────────────────┤
│  - HTTP 请求发送                                             │
│  - 重试机制                                                  │
│  - 错误处理                                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       [外部 API]                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    输出清洗层 (Output Cleaner)                │
│                     (cnllm/utils/cleaner.py)                │
├─────────────────────────────────────────────────────────────┤
│  - 清洗 Markdown 标记                                        │
│  - 提取 OpenAI 标准字段，将响应转换为标准格式                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 框架适配（LangChain 集成）

```
LangChain Chain
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│               LangChainRunnable (包装 CNLLM Client)         │
│              (cnllm/adapters/framework/langchain.py)        │
├─────────────────────────────────────────────────────────────┤
│  提供标准 LangChain 接口：                                    │
│  - invoke()    → 单次调用                                    │
│  - stream()    → 同步流式                                    │
│  - astream()   → 异步流式                                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 模块职责

**通用抽象层**包含多个组件，各司其职：

| 层级 | 组件 | 职责 | 示例 |
|------|------|------|------|
| **前端入口** | `CNLLM` (entry/client.py) | 统一入口、模型验证、参数标准化 | `CNLLM(model='minimax-m2.7')` |
| **请求预处理** | `BaseAdapter` (core/adapter.py) | 参数校验、Payload构建、厂商模型映射 | `_build_payload()`, `validate_model()` |
| **HTTP执行** | `BaseHttpClient` (entry/http.py) | 通用HTTP请求/重试 | `post_stream()`, `post()` |
| **响应后处理** | `Responder` (core/responder.py) | 统一OpenAI格式转换 | `to_openai_stream_format()` |
| **厂商特定层** | 各厂商 Adapter (core/vendor/) | 厂商特殊响应处理 | MiniMax 的 `reasoning_content` |

**原则**：
- **前端入口层** (`CNLLM`)：最早接收用户参数，统一做模型名小写转换和初步验证
- **预处理层** (`BaseAdapter`)：参数校验、过滤、Payload组装、调用HttpClient
- **HTTP层** (`BaseHttpClient`)：通用HTTP逻辑，不感知厂商差异
- **后处理层** (`Responder`)：将厂商原始响应转为OpenAI标准格式，所有厂商通用
- **厂商特定层**：厂商特有的响应字段（如MiniMax的`reasoning_content`）必须在厂商Adapter层处理

***

## 2. 三种调用入口

| 入口       | 调用方式                                                                |
| -------- | ------------------------------------------------------------------- |
| **极简入口** | `client("prompt")`                                                  |
| **标准入口** | `client.chat.create(prompt="prompt")`                               |
| **完整入口** | `client.chat.create(messages=[{"role": "user", "content": "..."}])` |

**调用链说明**：

- `client("prompt")` 通过 `__call__` 方法调用 `chat.create(prompt=prompt)`

***

## 3. Fallback 流程

### 3.1 调用决策流程

```
chat.create(messages, model, api_key, ...)
        │
        ▼
    model 指定?
    ├── 是 → 直接调用 adapter（跳过 fallback）
    │
    └── 否 → 调用 FallbackManager
                    │
                    ▼
            主模型可用?
            ├── 是 → 主模型成功
            │
            └── 否 → 按顺序尝试 fallback_models
                        │
                        ├── 全部失败 → FallbackError
                        └── 任一成功 → 该模型成功
```

### 3.2 模型与 Adapter 映射

```
SUPPORTED_MODELS = {
    "minimax-m2.7": "minimax",
    "minimax-m2.5": "minimax",
}

ADAPTER_MAP = {
    "minimax": MiniMaxAdapter,
}

模型验证流程:
1. 检查模型名是否在 SUPPORTED_MODELS
2. 通过映射获取 adapter_name
3. 通过 adapter_name 在 ADAPTER_MAP 获取 Adapter 类
4. 创建 Adapter 实例
```

### 3.3 raw 响应追踪

每次调用成功后，最后使用的 adapter 实例会保存到 `client._last_adapter`，可通过 `client.chat.raw` 访问原始响应：

```python
client.chat.create(messages=[...])
raw = client.chat.raw  # 原始 API 响应（含 base_resp 等平台字段）
```

***

## 4. 参数体系

### 4.1 参数分类 (params.py)

| 分类            | 定义   | 处理方式               |
| ------------- | ---- | ------------------ |
| **required**  | 必填参数 | Python 签名验证 + 类型检查 |
| **supported** | 可选参数 | ✅ 传递给 API          |
| **其他**        | 未知参数 | ⚠️ 警告 + 忽略后继续运行    |

未识别的参数统一警告+忽略后继续运行，简化逻辑同时提高兼容性。

### 4.2 params.py 注册表结构

```python
PROVIDER_PARAMS = {
    "minimax": {
        "init": {
            "required": ["api_key", "model"],
            "supported": ["base_url", "timeout", "max_retries", "retry_delay"],
        },
        "create": {
            "required": [],
            "supported": ["messages", "temperature", "max_tokens", "stream", "tools", "tool_choice", "group_id"],
        }
    }
}
```

***

## 5. 异常体系

### 5.1 异常类型

```
CNLLMError (基类)
├── AuthenticationError      # 认证失败 (401)
├── RateLimitError           # 限流 (429)
├── TimeoutError            # 超时 (408)
├── NetworkError            # 网络错误
├── ServerError             # 服务器错误 (5xx)
├── InvalidRequestError     # 请求错误 (400)
├── ParseError              # 解析错误
├── ModelNotSupportedError  # 模型不支持
├── MissingParameterError   # 缺少参数
├── ContentFilteredError    # 内容过滤 (403)
├── TokenLimitError        # Token 限制 (431)
├── ModelAPIError          # 模型 API 调用失败
└── FallbackError          # 所有模型均失败
```

### 5.2 异常属性

```python
class CNLLMError(Exception):
    message: str           # 错误消息
    error_code: ErrorCode  # 错误码枚举
    status_code: int       # HTTP 状态码
    provider: str          # 厂商标识
    details: dict          # 详细诊断信息
    suggestion: str        # 用户建议
```

***

## 6. 目录结构

```
cnllm/
├── entry/                    # 入口层 - 客户端初始化和调用入口
│   ├── __init__.py
│   ├── client.py             # CNLLM 主客户端类
│   └── http.py               # HTTP 请求客户端
├── core/                     # 核心层 - 适配器抽象和厂商实现
│   ├── __init__.py
│   ├── adapter.py            # BaseAdapter 基础适配器
│   ├── responder.py          # Responder 响应格式转换框架
│   ├── framework/
│   │   ├── __init__.py
│   │   └── langchain.py      # LangChain 集成
│   └── vendor/               # 厂商实现
│       ├── __init__.py
│       └── minimax.py        # MiniMax 厂商适配器
└── utils/                    # 工具层 - 通用工具
    ├── __init__.py
    ├── exceptions.py         # 异常定义
    ├── fallback.py           # Fallback 管理器
    ├── stream.py             # 流式处理工具
    ├── validator.py          # 参数验证器
    └── vendor_error.py       # 厂商错误处理

configs/
└── minimax/
    ├── request_minimax.yaml  # 请求配置
    └── response_minimax.yaml # 响应配置
```

***

## 7. YAML 厂商配置文件  

### 7.1 request_minimax.yaml

```yaml
request:
  method: "POST"
  url: "/text/chatcompletion_v2"
  base_url: "https://api.minimaxi.com/v1"
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${api_key}"

required_fields:
  api_key: ""
  model: ""

one_of:
  messages_or_prompt:
    messages: ""
    prompt: ""

optional_fields:
  fallback_models: ""
  # ... 更多可选参数

model_mapping:
  minimax-m2: "MiniMax-M2"
  # ... 更多模型映射关系

error_check:
  code_path: "base_resp.status_code"
  message_path: "base_resp.status_msg"
  success_code: 0
  error_codes:
    1001: { type: "timeout", message: "请求超时", suggestion: "请检查网络连接" }
    1002: { type: "rate_limit", message: "触发RPM限流", suggestion: "请降低请求频率" }
    # ... 更多错误码映射
```

### 7.2 response_minimax.yaml

```yaml
fields:
  id: "id"
  created: "created"
  model: "model"
  content: "choices[0].message.content"
  # ...

defaults:
  object: "chat.completion"
  # ...

stream_fields:
  # ...
```

### 7.3 YAML 功能集成

| 用途 | 访问点 | YAML 路径 |
|------|--------|-----------|
| 获取默认值 | `defaults`, `timeout`, `max_retries`, `retry_delay` | `default_values` |
| 厂商请求字段映射 | `build_payload` | `body_mapping` (在 request 中) |
| OpenAI响应字段映射 | `responder` | `fields` |
| 必填参数校验 | `validate_required_params` | `required_fields` |
| 参数支持校验 | `filter_supported_params` | `optional_fields` |
| 互斥参数校验 | `validate_one_of` | `one_of` |
| API配置 | `get_base_url`, `get_api_path` | `request.base_url`, `request.url` |
| 模型名映射 | `model_mapping` | `model_mapping` |

***

## 8. 异常处理系统

### 8.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Vendor Error (cnllm/core/vendor/)                 │
│  MiniMaxVendorError.from_response()                         │
│  职责：解析厂商原始响应 → code, message                      │
└─────────────────────────┬───────────────────────────────────┘
                          │ Registry.create_vendor_error()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Error Translator (cnllm/utils/vendor_error.py)   │
│  ErrorTranslator.translate()                               │
│  职责：查 YAML → type → CNLLM Error                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ raise CNLLM Error
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: CNLLM Error (cnllm/utils/exceptions.py)          │
│  RateLimitError, ServerError, AuthenticationError...        │
│  职责：统一异常类型，与厂商无关                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: User Code (用户应用层)                            │
│  try: ... except CNLLMError: ...                           │
│  职责：用户捕获并处理异常                                    │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `VendorError` | `utils/vendor_error.py` | 数据类：code, message, vendor, raw_response |
| `VendorErrorRegistry` | `utils/vendor_error.py` | 注册厂商错误类，创建 VendorError 实例 |
| `ErrorTranslator` | `utils/vendor_error.py` | 查 YAML 翻译为 CNLLM Error |
| `MiniMaxVendorError` | `core/vendor/minimax.py` | MiniMax 特有响应解析逻辑 |

***

## 10. 版本规划

### v0.3.2 (2026-04-01) ✨

- ✨ **参数统一** - 客户端初始化参数与调用入口参数统一化，调用入口灵活覆写
- ✨ **架构优化** - 核心逻辑抽象，模型适配器BaseAdapter和响应转换器Responder处理通用逻辑
- ✨ **可扩展性** - 接入新厂商只需配置相应 YAML 文件，自动实现请求和响应双端字段映射、错误码映射，无需修改其他上层组件
- ✨ **YAML 功能集成** - 关联字段映射、模型支持验证、必填项验证、参数支持验证、厂商错误码映射逻辑
- ✨ **MiniMax 支持优化** - 支持 MiniMax 原生接口所有参数，如 `top_p`、`tools`、`thinking` 等

### v0.4.0 (规划中)

- [ ] 模型适配开发（如豆包、Kimi 等）
- [ ] 框架适配验证和深度集成（LlamaIndex、Pydantic、LiteLLM、Instructor等）