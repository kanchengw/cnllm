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

| 模块           | 文件                                         | 职责                               |
| ------------ | ------------------------------------------ | -------------------------------- |
| **客户端入口**    | `cnllm/core/client.py`                     | 三种调用入口，参数透传，路由分发                 |
| **模型适配层**    | `cnllm/adapters/{厂商}/chat.py`              | 厂商协议转换 + 参数验证                    |
| **框架适配层**    | `cnllm/adapters/framework/*.py`            | 框架集成（LangChain 等）统一输出            |
| **模型映射**     | `cnllm/core/models.py`                     | SUPPORTED\_MODELS + ADAPTER\_MAP |
| **HTTP 基础**  | `cnllm/core/base.py`                       | 网络请求，重试                          |
| **异常定义**     | `cnllm/utils/exceptions.py`                | 统一异常体系                           |
| **参数注册**     | `cnllm/core/params.py`                     | 参数分类配置                           |
| **Fallback** | `cnllm/utils/fallback.py`                  | 降级机制                             |
| **输出清洗**     | `cnllm/utils/cleaner.py`                   | 统一模型输出格式                         |
| **模型验证**     | `cnllm/utils/validate_model_compatible.py` | 模型兼容性验证                          |

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
├── __init__.py              # 包入口，导出 CNLLM 和异常类
├── adapters/
│   ├── minimax/
│   │   └── chat.py         # MiniMax 适配器
│   └── framework/
│       └── langchain.py     # LangChain Runnable 适配器
├── core/
│   ├── client.py            # 客户端 (三种调用入口)
│   ├── models.py            # 模型映射 (SUPPORTED_MODELS, ADAPTER_MAP)
│   ├── params.py            # 参数注册表
│   └── base.py              # HTTP 基础层
└── utils/
    ├── config.py            # 环境配置
    ├── exceptions.py        # 异常定义
    ├── fallback.py          # Fallback 机制
    ├── validate_model_compatible.py  # 模型兼容性验证，用于引入新的模型支持
    └── cleaner.py           # 输出清洗
```

***

## 7. 版本规划

### v0.3.1 ✅ 已完成 (2026-03-29)

- [x] 结构化错误体系
- [x] 三种调用入口
- [x] stream 流式输出 (`stream=True`)
- [x] 简化参数验证（required/supported 两类）
- [x] LangChain Runnable 适配器
- [x] Fallback 机制
- [x] 模型兼容性验证工具
- [x] `client.chat.still` / `client.chat.raw` 响应入口

### v0.4.0 (规划中)

- [ ] 模型适配开发（如豆包、Kimi 等）
- [ ] 框架适配验证和深度集成（LlamaIndex、Pydantic、LiteLLM、Instructor）