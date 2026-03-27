# CNLLM 架构与设计文档

> 本文档用于团队内部参考，保证开发过程中架构和逻辑的一致性。
> 最后更新: 2026-03-28

***

## 1. 项目定位

**CNLLM (Chinese LLM Adapter)** - 中文大模型适配器

### 核心目标

将各种国产大模型（如 MiniMax、字节豆包、Kimi 等）的 API 输出转换为统一的 **OpenAI 格式**，实现：

- **零成本接入** LangChain、LlamaIndex 等主流框架
- **无缝切换** 不同大模型
- **统一接口** 简化多模型管理

***

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/client.py)                       │
├─────────────────────────────────────────────────────────────┤
│  两种统一接口：                                             │
│  - 极简接口: client("prompt")                              │
│  - 标准API接口: client.chat.create(messages=[...])         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Adapter Layer                            │
│               (cnllm/adapters/{厂商}/chat.py)              │
├─────────────────────────────────────────────────────────────┤
│  厂商适配层 (协议转换 + 参数验证)                            │
│  - MiniMaxAdapter                                          │
│  - DoubaoAdapter (规划中)                                  │
│  - KimiAdapter (规划中)                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                          │
│                     (cnllm/core/base.py)                   │
├─────────────────────────────────────────────────────────────┤
│  HTTP 基础层 (请求发送 / 重试机制 / 错误处理)               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块          | 文件                            | 职责          |
| ----------- | ----------------------------- | ----------- |
| **客户端入口**   | `cnllm/client.py`             | 两种统一接口，参数透传 |
| **适配器层**    | `cnllm/adapters/{厂商}/chat.py` | 协议转换 + 参数验证 |
| **HTTP 基础** | `cnllm/core/base.py`          | 网络请求，重试     |
| **异常定义**    | `cnllm/core/exceptions.py`    | 统一异常体系      |
| **参数注册**    | `cnllm/params.py`             | 参数分类配置      |
| **工具函数**    | `cnllm/utils/*.py`            | 辅助功能        |

***

## 3. 两种统一接口

| 接口            | 调用方式                                 | 对应 OpenAI 方式                              |
| ------------- | ------------------------------------ | ----------------------------------------- |
| **极简接口**      | `client("prompt")`                   | `openai_client("prompt")`                 |
| **标准 API 接口** | `client.chat.create(messages=[...])` | `openai_client.chat.completions.create()` |

**说明**：

- `client("prompt")` → 调用 `client.chat.create(prompt=prompt)` 实现
- 两种接口最终都通过 `chat.create()` 完成

***

## 4. 参数体系

### 4.1 参数分类 (params.py)

| 分类                     | 定义             | 处理方式                  |
| ---------------------- | -------------- | --------------------- |
| **required**           | OpenAI 必填参数    | Python 签名验证           |
| **supported**          | OpenAI 有且厂商支持  | ✅ 传递给 API             |
| **ignored**            | OpenAI 有但厂商不支持 | ⚠️ 警告 + 忽略            |
| **provider\_specific** | OpenAI 没有但厂商特有 | ✅ 通过 extra\_config 传递 |

### 4.2 params.py 注册表结构

```python
PROVIDER_PARAMS = {
    "minimax": {
        "init": {
            "required": [],
            "supported": [],
            "ignored": [],
            "provider_specific": []
        },
        "create": {
            "required": [],
            "supported": ["messages", "temperature", "max_tokens", "stream"],
            "ignored": ["top_p", "n", "stop", "presence_penalty", "frequency_penalty", "user"],
            "provider_specific": ["group_id"]
        }
    }
}
```

***

## 5. 统一接口参数传递逻辑

### 5.1 处理规则表格

| # | 参数情况                        | 传入方式          | 操作                         | 说明                                 |
| - | --------------------------- | ------------- | -------------------------- | ---------------------------------- |
| 1 | **缺失必备字段**                  | 用户未传          | 抛出 `MissingParameterError` | messages 或 prompt 必须传一个            |
| 2 | **supported 类**             | 普通参数          | 传递给 adapter → API          | 如 temperature, max\_tokens, stream |
| 3 | **ignored 类**               | 普通参数          | 警告 + 不传                    | 如 top\_p, n, stop 等                |
| 4 | **在 provider\_specific 中**  | 普通参数          | 警告"应使用 extra\_config" + 不传 | 如 group\_id 作为普通参数传入               |
| 5 | **未知参数**                    | 普通参数          | 警告 + 不传                    | 不在任何分类中                            |
| 6 | **在 provider\_specific 中**  | extra\_config | 传递给 adapter → API          | 如 group\_id                        |
| 7 | **不在 provider\_specific 中** | extra\_config | 警告"非有效 extra\_config" + 忽略 | extra\_config 中传了普通参数              |

**核心原则**：

- 除了必填字段缺失外，其他所有情况都是**警告 + 继续运行**
- 不在支持列表的参数都不传给 API，避免模型受到影响

### 5.2 验证流程

```
用户调用 (极简/标准/适配接口)
        │
        ▼
┌─────────────────────────────────────┐
│ client.py                            │
│ 1. 检查必填字段 (messages/prompt)     │
│ 2. 透传所有参数到 adapter             │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ adapter (minimax/chat.py)            │
│ 1. _validate_and_warn_params(kwargs)  │
│    - ignored → 警告                  │
│    - provider_specific → 警告         │
│    - 未知参数 → 警告                  │
│ 2. _warn_invalid_extra_config()      │
│    - 非 provider_specific → 警告      │
│ 3. 协议转换 → 调用厂商 API           │
└─────────────────────────────────────┘
```

***

## 6. 异常体系

### 6.1 异常类层次

```
CNLLMError (基类)
├── AuthenticationError      # 认证失败
├── RateLimitError           # 限流
├── TimeoutError            # 超时
├── NetworkError            # 网络错误
├── ServerError             # 服务器错误
├── InvalidRequestError     # 请求错误
├── ParseError              # 解析错误
├── ModelNotSupportedError  # 模型不支持
├── MissingParameterError   # 缺少参数
├── ContentFilteredError    # 内容过滤
└── TokenLimitError        # Token 限制
```

### 6.2 异常属性

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

## 7. 目录结构

```
cnllm/
├── __init__.py              # 包入口
├── client.py                # 客户端 (两种统一接口)
├── params.py                # 参数注册表
├── adapters/
│   ├── minimax/
│   │   └── chat.py         # MiniMax 适配器
│   └── framework/
│       └── langchain.py     # LangChain Runnable 适配器
├── core/
│   ├── base.py             # HTTP 基础层
│   ├── exceptions.py        # 异常定义
│   └── config.py            # 配置
└── utils/
    └── cleaner.py          # 输出清洗
```

***

## 8. 版本规划

### v0.3.0 ✅ 已完成

- [x] 结构化错误体系
- [x] 两种统一接口
- [x] stream 流式输出
- [x] extra\_config 厂商特有参数
- [x] LangChain Runnable 适配器

### v0.4.0 (规划中)

- [ ] Doubao 适配器
- [ ] Kimi 适配器

***

## 10. 设计原则

1. **OpenAI 兼容优先** - 输出格式完全对齐 OpenAI
2. **参数可扩展** - 通过 params.py 统一管理
3. **静默降级** - 不支持的参数警告但不阻断
4. **错误友好** - 异常包含诊断信息和解决建议
5. **职责分离** - client 做路由，adapter 做验证

