# CNLLM 架构与设计文档

## 1. 架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/entry/client.py)                 │
├─────────────────────────────────────────────────────────────┤
│  三种调用入口：                                                │
│  - 极简入口: client("prompt")                                 │
│  - 标准入口: client.chat.create(prompt="...")                 │
│  - 完整入口: client.chat.create(messages=[...])               │
│                                                             │
│  响应入口：                                                   │
│  - client.chat.still  → 纯净文本                             │
│  - client.chat.think → 思考过程（reasoning_content）          │
│  - client.chat.tools → 工具调用（tool_calls）                 │
│  - client.chat.raw   → 原始响应（含平台特有字段）               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseAdapter                             │
│              (cnllm/core/adapter.py)                        │
├─────────────────────────────────────────────────────────────┤
│  - 参数验证（YAML 配置驱动）                                  │
│  - Payload 构建                                              │
│  - HTTP 请求发送                                             │
│  - 厂商模型映射                                              │
│  - 委托 Responder 处理响应转换                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Responder                                │
│              (cnllm/core/responder.py)                      │
├─────────────────────────────────────────────────────────────┤
│  - 响应格式转换（厂商 → OpenAI 标准）                         │
│  - reasoning_content 提取与累积                              │
│  - tool_calls 提取与累积                                     │
│  - usage 信息处理                                           │
│  - 敏感内容检测（input_sensitive_type / output_sensitive）    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VendorError                              │
│              (cnllm/utils/vendor_error.py)                  │
├─────────────────────────────────────────────────────────────┤
│  - 厂商错误解析（code → CNLLM Error）                        │
│  - 错误码映射（YAML 配置驱动）                                │
│  - 敏感内容检测触发                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                           │
│                     (cnllm/entry/http.py)                   │
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
```

### 1.2 三层架构组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **BaseAdapter** | `core/adapter.py` | 核心适配逻辑：参数校验、Payload构建、请求发送 |
| **Responder** | `core/responder.py` | 响应转换：厂商格式 → OpenAI 标准格式 |
| **VendorError** | `utils/vendor_error.py` | 错误处理：厂商错误码 → CNLLM 统一异常 |

### 1.3 分层架构原则

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

## 2. 目录结构

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
│       ├── minimax.py        # MiniMax 厂商适配器
│       └── xiaomi.py         # Xiaomi 厂商适配器
└── utils/                    # 工具层 - 通用工具
    ├── __init__.py
    ├── exceptions.py         # 异常定义
    ├── fallback.py           # Fallback 管理器
    ├── stream.py             # 流式处理工具
    ├── validator.py          # 参数验证器
    └── vendor_error.py       # 厂商错误处理

configs/
├── minimax/
│   ├── request_minimax.yaml  # 请求配置
│   └── response_minimax.yaml # 响应配置
└── xiaomi/
    ├── request_xiaomi.yaml   # 请求配置
    └── response_xiaomi.yaml  # 响应配置
```

***

## 3. 模型选择流程

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

## 4. YAML 厂商配置文件

### 4.1 YAML 逻辑实现

| 用途 | 访问点 | YAML 路径 | YAML 表名 |
|------|--------|-----------|------|  
| 获取默认值 | `timeout`, `max_retries`... | `default_values` | request_{vendor}.yaml |
| 厂商请求字段映射 | `build_payload` | `body_mapping` (在 request 中) | request_{vendor}.yaml |
| 必填参数校验 | `validate_required_params` | `required_fields` | request_{vendor}.yaml |
| 参数支持校验 | `filter_supported_params` | `optional_fields` | request_{vendor}.yaml |
| 互斥参数校验 | `validate_one_of` | `one_of` | request_{vendor}.yaml |
| API配置 | `get_base_url`, `get_api_path` | `request.base_url`, `request.url` | request_{vendor}.yaml |
| 模型名映射 | `model_mapping` | `model_mapping` | request_{vendor}.yaml |
| OpenAI响应字段映射 | `responder` | `fields` | response_{vendor}.yaml |

### 4.2 request_{vendor}.yaml

```yaml
request:
  method: "POST"
  url: "/chat/completions"
  base_url: "https://api.{vendor}.com/v1"
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
  stream: ""
  thinking:
    path: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
  # ... 更多可选参数

model_mapping:
  minimax-m2: "MiniMax-M2"
  mimo-v2-flash: "mimo-v2-flash"
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

### 4.3 response_{vendor}.yaml

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
  delta:
    content: "delta.content"
    reasoning_content: "delta.reasoning_content"
  # ...

usage:
  prompt_tokens: "usage.prompt_tokens"
  completion_tokens: "usage.completion_tokens"
  total_tokens: "usage.total_tokens"
  prompt_tokens_details:
    cached_tokens: "usage.prompt_tokens_details.cached_tokens"
  ...
```

***

## 5. 异常处理系统

### 5.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Vendor Error (cnllm/core/vendor/)                 │
│  MiniMaxVendorError.from_response()                         │
│  职责：解析厂商原始响应 → code, message                      │
└─────────────────────────┬───────────────────────────────────┘
                          │ Registry.create_vendor_error()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Error Translator (cnllm/utils/vendor_error.py)     │
│  ErrorTranslator.translate()                                │
│  职责：查 YAML → type → CNLLM Error                         │
└─────────────────────────┬───────────────────────────────────┘
                          │ raise CNLLM Error
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: CNLLM Error (cnllm/utils/exceptions.py)           │
│  RateLimitError, ServerError, AuthenticationError...       │
│  职责：统一异常类型，与厂商无关                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: User Code (用户应用层)                             │
│  try: ... except CNLLMError: ...                            │
│  职责：用户捕获并处理异常                                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `VendorError` | `utils/vendor_error.py` | 数据类：code, message, vendor, raw_response |
| `VendorErrorRegistry` | `utils/vendor_error.py` | 注册厂商错误类，创建 VendorError 实例 |
| `ErrorTranslator` | `utils/vendor_error.py` | 查 YAML 翻译为 CNLLM Error |
| `MiniMaxVendorError` | `core/vendor/minimax.py` | MiniMax 特有响应解析逻辑 |

***

## 6. 版本更新

### v0.4.0 ✅ 已完成 (2026-04-03)

- ✨ **mimo适配** - 小米mimo模型适配开发，支持"mimo-v2-pro"、"mimo-v2-omni"、"mimo-v2-flash"
- ✨ **架构重构** - BaseAdapter + Responder + VendorError 三层架构分离，职责清晰
- ✨ **.think 属性** - `client.chat.think` 获取 reasoning_content，支持流式累积
- ✨ **.tools 属性** - `client.chat.tools` 获取 tool_calls，支持流式累积
- ✨ **流式累积** - `.think`、`.still`、`.tools` 支持在流式响应中实时滚动积累

### v0.3.3 (2026-04-02) ✨

- ✨ **参数统一** - 客户端初始化参数与调用入口参数统一化，调用入口灵活覆写
- ✨ **架构优化** - 核心逻辑抽象，模型适配器BaseAdapter和响应转换器Responder处理通用逻辑
- ✨ **可扩展性** - 接入新厂商只需配置相应 YAML 文件，自动实现请求和响应双端字段映射、错误码映射，无需修改其他上层组件
- ✨ **YAML 功能集成** - 关联字段映射、模型支持验证、必填项验证、参数支持验证、厂商错误码映射逻辑
- ✨ **MiniMax 支持优化** - 支持 MiniMax 原生接口所有参数，如 `top_p`、`tools`、`thinking` 等

### v0.3.2 (2026-03-29) ✨

- ✨ **LangChain深度适配**
  - Runnable 适配器作为核心功能，一个函数接入Langchain chain
  - Runnable 流式输出、批量调用、异步调用支持
- ✨ **chat.create() 流式输出** - `stream=True` 参数支持
- ✨ **Fallback 机制** - 主模型失败时自动切换到备用模型
- ✨ **响应入口** - `client.chat.still` 轻松获取纯净会话响应，`client.chat.raw` 获取完整响应
- 🔧 **适配器重构** - 模型适配器（中文大模型 MiniMax 等）+ 框架适配器（LangChain 等）双层架构
