# DeepSeek 适配

## 支持的模型

| 模型 | 说明 | 支持 thinking |
|------|------|--------------|
| `deepseek-chat` | DeepSeek 对话模型 | ✅ |
| `deepseek-reasoner` | DeepSeek 推理模型 | ✅ |

**两个模型支持的参数完全一致**，唯一区别是 `thinking` 和 `max_tokens` 默认值不同。

## 接口配置

- **API 地址**: `https://api.deepseek.com/v1`
- **端点**: `/chat/completions`
- **认证**: `Authorization: Bearer ${DEEPSEEK_API_KEY}`
- **协议**: OpenAI 兼容 API

## 请求配置

### 配置文件

- `configs/deepseek/request_deepseek.yaml` - 请求参数配置
- `configs/deepseek/response_deepseek.yaml` - 响应字段映射

### 字段映射

| CNLLM 参数 | DeepSeek 字段 | 转换说明 |
|:---|:---|:---|
| `model` | `model` | 直接传递 |
| `messages` | `messages` | 直接传递 |
| `temperature` | `temperature` | 直接传递，默认 0.7 |
| `top_p` | `top_p` | 直接传递，默认 1.0 |
| `max_tokens` | `max_tokens` | 直接传递，默认 chat=4096, reasoner=32768 |
| `stream` | `stream` | 直接传递 |
| `stop` | `stop` | 直接传递（字符串或数组） |
| `n` | `n` | 直接传递 |
| `tools` | `tools` | 直接传递 |
| `tool_choice` | `tool_choice` | 直接传递 |
| `user` | `user` | 直接传递 |
| `presence_penalty` | `presence_penalty` | 直接传递 |
| `frequency_penalty` | `frequency_penalty` | 直接传递 |
| `response_format` | `response_format` | 直接传递 |
| `thinking` | `thinking` | 格式转换 |
| `logit_bias` | `logit_bias` | 直接传递 |

### `thinking` 参数转换

CNLLM 使用布尔值 `thinking=true/false`，DeepSeek 使用对象格式 `thinking.type`：

```yaml
# configs/deepseek/request_deepseek.yaml
optional_fields:
  thinking:
    body: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
```

**转换示例：**
- CNLLM: `thinking=true` → DeepSeek: `{"thinking": {"type": "enabled"}}`
- CNLLM: `thinking=false` → DeepSeek: `{"thinking": {"type": "disabled"}}`

**默认行为**：
- `deepseek-chat`: 默认 `disabled`
- `deepseek-reasoner`: 默认 `enabled`

### 不支持的参数

DeepSeek **不支持**以下 OpenAI 标准参数：
- `logprobs` (请求参数)
- `echo`
- `best_of`
- `seed`
- `service_tier`
- `stream_options`

## 响应特性

### 标准响应字段

| DeepSeek 响应字段 | CNLLM/OpenAI 字段 | 说明 |
|:---|:---|:---|
| `id` | `id` | 消息唯一标识 |
| `created` | `created` | 时间戳 |
| `model` | `model` | 模型名称 |
| `choices[0].message.content` | `content` | 纯净文本 |
| `choices[0].message.reasoning_content` | `_thinking` | 推理过程（reasoner 模型） |
| `choices[0].finish_reason` | `finish_reason` | 结束原因 |
| `choices[0].message.tool_calls` | `tool_calls` | 函数调用 |
| `choices[0].logprobs` | `choices[0].logprobs` | token 对数概率（通常为 null） |
| `system_fingerprint` | `system_fingerprint` | 服务端指纹 |
| `usage.prompt_tokens` | `prompt_tokens` | 输入 token 数 |
| `usage.completion_tokens` | `completion_tokens` | 输出 token 数 |
| `usage.total_tokens` | `total_tokens` | 总 token 数 |

### 响应格式示例

**非流式响应：**
```json
{
  "id": "deepseek-chat-xxx",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-chat",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！有什么我可以帮助你的吗？"
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  },
  "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache"
}
```

**带 thinking 的响应（deepseek-reasoner）：**
```json
{
  "id": "deepseek-reasoner-xxx",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-reasoner",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "天是蓝色的，因为瑞利散射...",
      "reasoning_content": "用户问为什么天是蓝色，我需要解释瑞利散射原理..."
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  },
  "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache"
}
```

### 流式响应

**标准格式：**
```
data: {"id":"deepseek-chat-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"deepseek-chat-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

...

data: {"id":"deepseek-chat-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-chat","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":8,"completion_tokens":10,"total_tokens":18}}
```

**带 thinking 的流式响应（deepseek-reasoner）：**
```
data: {"id":"deepseek-reasoner-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"用户"},"finish_reason":null}]}

data: {"id":"deepseek-reasoner-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"问为什么"},"finish_reason":null}]}

...

data: {"id":"deepseek-reasoner-xxx","object":"chat.completion.chunk","created":1677652288,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"天是蓝色的..."},"finish_reason":null}]}

...

data: [DONE]
```

### 属性访问

| 属性 | 来源 | 说明 |
|------|------|------|
| `client.chat.still` | `choices[0].message.content` | 纯净文本 |
| `client.chat.think` | `_thinking` 或 `reasoning_content` | 思考过程 |
| `client.chat.tools` | `choices[0].message.tool_calls` | 函数调用 |
| `client.chat.raw` | 原始响应 | 包含 DeepSeek 特有字段 |

### 流式累积

流式响应中，内容会被自动累积到对应属性：

```python
client = CNLLM(model="deepseek-reasoner", api_key="...")

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
| 402 | 配额不足 | `RateLimitError` |
| 403 | 权限不足 | `AuthenticationError` |
| 404 | 未找到 | `InvalidRequestError` |
| 429 | 请求频率超限 | `RateLimitError` |
| 500 | 服务器内部错误 | `ModelAPIError` |
| 503 | 服务不可用 | `ModelAPIError` |

### DeepSeek 错误码体系

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

**业务错误码 (error.type)：**

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
# cnllm/core/vendor/deepseek.py
class DeepSeekVendorError(VendorError):
    VENDOR_NAME = "deepseek"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["DeepSeekVendorError"]:
        if not raw_response:
            return None

        error = raw_response.get("error", {})
        error_type = error.get("type")
        if error_type is None:
            return None

        message = error.get("message", "")
        code = error.get("code", error_type)

        return cls(
            code=code,
            message=message,
            vendor=cls.VENDOR_NAME,
            raw_response=raw_response
        )
```

## 实现架构

### 核心类

```
cnllm/core/vendor/deepseek.py
├── DeepSeekVendorError    # DeepSeek 特有错误解析
├── DeepSeekResponder      # DeepSeek 响应格式转换
└── DeepSeekAdapter        # DeepSeek 适配器（继承 BaseAdapter）
```

### DeepSeekAdapter 实现要点

1. **OpenAI 兼容**：DeepSeek API 与 OpenAI 格式高度兼容，大部分字段直接传递
2. **thinking 参数转换**：需要将布尔值转换为 DeepSeek 的对象格式
3. **错误解析**：通过 `DeepSeekVendorError.from_response()` 解析 DeepSeek 特有错误
4. **字段映射**：从 `choices[0].logprobs` 提取到标准响应的 `choices[0].logprobs`
5. **系统指纹**：从 `system_fingerprint` 提取到标准响应的顶层

```python
class DeepSeekAdapter(BaseAdapter):
    ADAPTER_NAME = "deepseek"
    CONFIG_DIR = "deepseek"

    def __init__(self, api_key: str, model: str, ...):
        super().__init__(api_key=api_key, model=model, ...)
        self.responder = DeepSeekResponder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


DeepSeekAdapter._register()
```

## 使用示例

### 基础调用

```python
from cnllm import CNLLM

client = CNLLM(model="deepseek-chat", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response["choices"][0]["message"]["content"])
```

### 启用思考过程（deepseek-reasoner）

```python
client = CNLLM(model="deepseek-reasoner", api_key="your-api-key")

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
client = CNLLM(model="deepseek-chat", api_key="your-api-key")

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
):
    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
```

### 流式 thinking

```python
client = CNLLM(model="deepseek-reasoner", api_key="your-api-key")

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "1+1等于几"}],
    stream=True,
    thinking=True
):
    think_preview = chunk.get("_thinking", "")
    content_preview = chunk.get("_content", "")
    if think_preview:
        print(f"[Think] {think_preview[:50]}...")
    if content_preview:
        print(f"[Content] {content_preview}")
```

### 工具调用

```python
client = CNLLM(model="deepseek-chat", api_key="your-api-key")

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

### JSON 响应格式

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "返回一个JSON对象，包含name和age字段"}],
    max_tokens=100,
    response_format={"type": "json_object"}
)
```

## 测试

### 测试文件

| 测试文件 | 说明 |
|----------|------|
| `test_deepseek_client.py` | 客户端集成测试 |
| `test_deepseek_adapter_config.py` | YAML 配置加载测试 |
| `test_deepseek_adapter_payload.py` | Payload 构建测试 |
| `test_deepseek_responder_format.py` | 响应格式转换测试 |
| `test_deepseek_responder_reasoning.py` | 推理内容处理测试 |
| `test_deepseek_stream.py` | 流式响应测试 |
| `test_deepseek_stream_accumulator.py` | 流式累积器测试 |
| `test_deepseek_yaml_request.py` | 请求 YAML 配置测试 |
| `test_deepseek_yaml_response.py` | 响应 YAML 配置测试 |
| `test_deepseek_langchain_runnable.py` | LangChain 集成测试 |

### 运行测试

```bash
# 运行所有 DeepSeek 测试
pytest test_deepseek_*.py -v

# 运行特定测试
pytest test_deepseek_client.py -v -s
```
