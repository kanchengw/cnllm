# Kimi (Moonshot AI) 适配

## 支持的模型

| 模型 | 说明 | 支持 thinking |
|------|------|--------------|
| `moonshot-v1-8k` | 8K 上下文版本 | ❌ |
| `moonshot-v1-32k` | 32K 上下文版本 | ❌ |
| `moonshot-v1-128k` | 128K 上下文版本 | ❌ |
| `kimi-k2.5` | K2.5 模型，temperature 固定 0.95 | ✅ |
| `kimi-k2-thinking` | K2 长思考模型，支持 256k 上下文 | ✅ |
| `kimi-k2-thinking-turbo` | K2 长思考模型高速版本 | ✅ |
| `kimi-k2-0905-preview` | 增强 Agentic Coding 能力，上下文 256k | ❌ |
| `kimi-k2-0711-preview` | MoE 架构基础模型，上下文 128k | ❌ |
| `kimi-k2-turbo-preview` | K2 高速版本，上下文 256k | ❌ |

## 接口配置

- **API 地址**: `https://api.moonshot.cn/v1`
- **端点**: `/chat/completions`
- **认证**: `Authorization: Bearer ${MOONSHOT_API_KEY}`
- **协议**: OpenAI 兼容 API

## 请求配置

### 配置文件

- `configs/kimi/request_kimi.yaml` - 请求参数配置
- `configs/kimi/response_kimi.yaml` - 响应字段映射

### 字段映射

| CNLLM 参数 | Kimi 字段 | 转换说明 |
|:---|:---|:---|
| `model` | `model` | 直接传递 |
| `messages` | `messages` | 直接传递 |
| `temperature` | `temperature` | 直接传递，kimi-k2.5 固定 0.95 |
| `top_p` | `top_p` | 直接传递，默认 1.0 |
| `max_tokens` | `max_completion_tokens` | 字段名映射 |
| `stream` | `stream` | 直接传递 |
| `stop` | `stop` | 直接传递（字符串或数组） |
| `n` | `n` | 直接传递 |
| `tools` | `tools` | 直接传递 |
| `tool_choice` | `tool_choice` | 直接传递 |
| `user` | `user` | 直接传递 |
| `presence_penalty` | `presence_penalty` | 直接传递 |
| `frequency_penalty` | `frequency_penalty` | 直接传递 |
| `response_format` | `response_format` | 直接传递 |
| `stream_options` | `stream_options` | 直接传递 |
| `thinking` | `thinking` | 格式转换 |

### `thinking` 参数转换

CNLLM 使用布尔值 `thinking=true/false`，Kimi 使用对象格式 `thinking.type`：

```yaml
# configs/kimi/request_kimi.yaml
optional_fields:
  thinking:
    body: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
```

**转换示例：**
- CNLLM: `thinking=true` → Kimi: `{"thinking": {"type": "enabled"}}`
- CNLLM: `thinking=false` → Kimi: `{"thinking": {"type": "disabled"}}`

**注意**：`thinking` 参数仅对 kimi-k2.5 和 kimi-k2-thinking 系列模型有效，其他模型传入此参数会忽略。

### Kimi 特有参数

| 参数 | 说明 |
|------|------|
| `prompt_cache_key` | 缓存键，用于加速相同上下文的请求 |
| `safety_identifier` | 安全标识，控制内容安全策略 |

### `messages` 结构详解

**role 取值**: system, user, assistant

**content 格式**:

```yaml
# 纯文本
messages[].content: string

# 多模态内容（Vision 支持）
messages[].content:
  - type: "text"
    text: string  # 文本内容
  - type: "image_url"
    image_url:
      url: string  # 支持 base64 或 ms://<file_id>
  - type: "video_url"
    video_url:
      url: string  # 支持 base64 或 ms://<file_id>
```

**assistant 消息特殊字段**:
```yaml
messages[].reasoning_content: string  # 思维链内容（kimi-k2 系列）
messages[].tool_calls:  # 工具调用
  - id: string
    type: "function"
    function:
      name: string
      arguments: string  # JSON 格式
```

**tool 消息结构**:
```yaml
messages[].role: "tool"
messages[].content: string
messages[].tool_call_id: string  # 关联 tool_calls 的 id
```

## 响应特性

### 标准响应字段

| Kimi 响应字段 | CNLLM/OpenAI 字段 | 说明 |
|:---|:---|:---|
| `id` | `id` | 消息唯一标识 |
| `created` | `created` | 时间戳 |
| `model` | `model` | 模型名称 |
| `choices[0].message.content` | `content` | 纯净文本 |
| `choices[0].message.reasoning_content` | `_thinking` | 推理过程（kimi-k2 系列） |
| `choices[0].finish_reason` | `finish_reason` | 结束原因 |
| `choices[0].message.tool_calls` | `tool_calls` | 函数调用 |
| `usage.prompt_tokens` | `prompt_tokens` | 输入 token 数 |
| `usage.completion_tokens` | `completion_tokens` | 输出 token 数 |
| `usage.total_tokens` | `total_tokens` | 总 token 数 |

### 响应格式示例

**非流式响应：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "moonshot-v1-8k",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！有什么我可以帮助你的吗？"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

**带 thinking 的非流式响应（kimi-k2 系列）：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "kimi-k2-thinking-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "天是蓝色的，因为瑞利散射...",
      "reasoning_content": "用户问为什么天是蓝色，我需要解释瑞利散射原理..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  }
}
```

### 流式响应

**标准格式：**
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"moonshot-v1-8k","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":8,"completion_tokens":10,"total_tokens":18}}
```

**带 thinking 的流式响应（kimi-k2 系列）：**
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"kimi-k2-thinking-turbo","choices":[{"index":0,"delta":{"reasoning_content":"用户"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"kimi-k2-thinking-turbo","choices":[{"index":0,"delta":{"reasoning_content":"问为什么"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1677652288,"model":"kimi-k2-thinking-turbo","choices":[{"index":0,"delta":{"content":"天是蓝色的..."},"finish_reason":null}]}

...

data: [DONE]
```

### 属性访问

| 属性 | 来源 | 说明 |
|------|------|------|
| `client.chat.still` | `choices[0].message.content` | 纯净文本 |
| `client.chat.think` | `_thinking` 或 `reasoning_content` | 思考过程 |
| `client.chat.tools` | `choices[0].message.tool_calls` | 函数调用 |
| `client.chat.raw` | 原始响应 | 包含 Kimi 特有字段 |

### 流式累积

流式响应中，内容会被自动累积到对应属性：

```python
client = CNLLM(model="kimi-k2-thinking-turbo", api_key="...")

# 流式调用
for chunk in client.chat.create(messages=[...], stream=True, thinking=True):
    pass

# 获取累积内容
print(client.chat.still)    # 纯净文本
print(client.chat.think)    # 思考过程（实时累积）
print(client.chat.tools)    # 工具调用
```

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | CNLLM 异常 |
|--------|------|-----------|
| 200 | 成功 | - |
| 400 | 请求错误 | `InvalidRequestError` |
| 401 | 认证失败 | `AuthenticationError` |
| 403 | 权限不足 | `AuthenticationError` |
| 404 | 未找到 | `InvalidRequestError` |
| 429 | 请求频率超限 | `RateLimitError` |
| 500 | 服务器内部错误 | `ModelAPIError` |
| 503 | 服务不可用 | `ModelAPIError` |

### Kimi 错误码体系

**错误响应格式：**
```json
{
  "error": {
    "message": "Invalid Authentication",
    "type": "invalid_authentication_error",
    "code": "invalid_authentication_error",
    "param": null,
    "status": 401
  }
}
```

**业务错误码 (error.code)：**

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| `invalid_authentication_error` | API Key 无效或已过期 | 检查 API Key 是否正确 |
| `authentication_error` | 认证失败 | 检查认证信息 |
| `rate_limit_error` | 请求频率超限 | 降低请求频率 |
| `invalid_request_error` | 请求参数错误 | 检查请求参数 |
| `server_error` | 服务器内部错误 | 稍后重试 |
| `not_found_error` | 资源未找到 | 检查请求路径 |
| `insufficient_quota_error` | 配额不足 | 检查账户余额 |
| `content_filter` | 内容被过滤 | 修改内容重试 |
| `access_denied_error` | 禁止访问 | 检查权限设置 |

### 错误解析

```python
# cnllm/core/vendor/kimi.py
class KimiVendorError(VendorError):
    VENDOR_NAME = "kimi"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["KimiVendorError"]:
        if not raw_response:
            return None
        error = raw_response.get("error", {})
        code = error.get("code")
        if code is None:
            return None
        message = error.get("message", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)
```

## 实现架构

### 核心类

```
cnllm/core/vendor/kimi.py
├── KimiVendorError    # Kimi 特有错误解析
├── KimiResponder      # Kimi 响应格式转换
└── KimiAdapter        # Kimi 适配器（继承 BaseAdapter）
```

### KimiAdapter 实现要点

1. **OpenAI 兼容**：Kimi API 与 OpenAI 格式高度兼容，大部分字段直接传递
2. **thinking 参数转换**：需要将布尔值转换为 Kimi 的对象格式
3. **错误解析**：通过 `KimiVendorError.from_response()` 解析 Kimi 特有错误

```python
class KimiAdapter(BaseAdapter):
    ADAPTER_NAME = "kimi"
    CONFIG_DIR = "kimi"

    def _build_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model = params.get("model") or self.model
        vendor_model = self.get_vendor_model(model)

        payload = {"model": vendor_model}
        excluded = {"model", "api_key", "base_url", "timeout", "max_retries", "retry_delay", "fallback_models"}
        optional_fields = self._get_config_value("optional_fields", default={})

        for key, value in params.items():
            if key in excluded or value is None:
                continue
            field_config = optional_fields.get(key, key)
            if isinstance(field_config, dict):
                transform = field_config.get("transform")
                if transform and value in transform:
                    value = transform[value]
                mapped_key = field_config.get("body", key)
                if mapped_key == "__skip__":
                    continue
            else:
                mapped_key = field_config if field_config else key
            payload[mapped_key] = value
        return payload
```

## 使用示例

### 基础调用

```python
from cnllm import CNLLM

client = CNLLM(model="moonshot-v1-8k", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response["choices"][0]["message"]["content"])
```

### 启用思考过程（kimi-k2 系列）

```python
client = CNLLM(model="kimi-k2-thinking-turbo", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "解释为什么天是蓝色的"}],
    max_tokens=500,
    thinking=True  # 启用思维链
)

# 获取思考过程
print(client.chat.think)
# 获取最终答案
print(client.chat.still)
```

### 流式调用

```python
client = CNLLM(model="moonshot-v1-8k", api_key="your-api-key")

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
):
    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
```

### 工具调用

```python
client = CNLLM(model="moonshot-v1-32k", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "查一下北京的天气"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名"}
                }
            }
        }
    }]
)
print(client.chat.tools)
```

### 设置 temperature 和 top_p

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "写一个故事"}],
    temperature=0.8,
    top_p=0.9
)
```

### 设置停止序列

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "数到10"}],
    stop=["5", "6"]  # 到达 5 或 6 时停止
)
```
