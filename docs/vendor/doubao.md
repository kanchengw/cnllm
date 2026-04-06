# Doubao 豆包 AI 适配

## 支持的模型

| CNLLM 模型 | Doubao 原始模型 | 说明 | 支持 thinking |
|------------|-----------------|------|--------------|
| `doubao-seed-2-0-pro` | `doubao-seed-2-0-pro-260215` | Seed 2.0 Pro 版本 | ✅ |
| `doubao-seed-2-0-mini` | `doubao-seed-2-0-mini-260215` | Seed 2.0 Mini 版本 | ✅ |
| `doubao-seed-2-0-lite` | `doubao-seed-2-0-lite-260215` | Seed 2.0 Lite 版本 | ✅ |
| `doubao-seed-2-0-code` | `doubao-seed-2-0-code-preview-260215` | Seed 2.0 Code 预览版 | ✅ |
| `doubao-seed-1-8` | `doubao-seed-1-8-251228` | Seed 1.8 版本 | ✅ |
| `doubao-seed-1-6` | `doubao-seed-1-6-251015` | Seed 1.6 版本 | ✅ |
| `doubao-seed-1-6-lite` | `doubao-seed-1-6-lite-251015` | Seed 1.6 Lite 版本 | ✅ |
| `doubao-seed-1-6-flash` | `doubao-seed-1-6-flash-250828` | Seed 1.6 Flash 版本 | ✅ |

## 接口配置

- **API 地址**: `https://ark.cn-beijing.volces.com/api/v3`
- **端点**: `/chat/completions`
- **认证**: `Authorization: Bearer ${api_key}`
- **协议**: OpenAI 兼容

## 请求配置

### 配置文件

- `configs/doubao/request_doubao.yaml` - 请求参数配置
- `configs/doubao/response_doubao.yaml` - 响应字段映射

### 字段映射

CNLLM 参数与 Doubao API 字段的对应关系：

| CNLLM 参数 | Doubao 字段 | 转换说明 |
|:---|:---|:---|
| `model` | `model` | 通过 model_mapping 映射 |
| `messages` | `messages` | 直接传递 |
| `temperature` | `temperature` | 直接传递 |
| `top_p` | `top_p` | 直接传递 |
| `max_tokens` | `max_tokens` / `max_completion_tokens` | 直接传递 |
| `stream` | `stream` | 直接传递 |
| `stop` | `stop` | 直接传递（数组） |
| `tools` | `tools` | 直接传递 |
| `tool_choice` | `tool_choice` | 直接传递 |
| `user` | `user` | 直接传递 |
| `presence_penalty` | `presence_penalty` | 直接传递 |
| `frequency_penalty` | `frequency_penalty` | 直接传递 |
| `response_format` | `response_format` | 直接传递 |
| `stream_options` | `stream_options` | 直接传递 |
| `thinking` | `thinking.type` | 格式转换 |
| `reasoning_effort` | `reasoning_effort` | 直接传递 |
| `service_tier` | `service_tier` | 直接传递 |

### `thinking` 参数转换

CNLLM 使用布尔值 `thinking=true/false`，Doubao 使用字符串格式 `thinking.type`：

```yaml
# configs/doubao/request_doubao.yaml
optional_fields:
  thinking:
    body: "thinking.type"
    transform:
      true: "enabled"
      false: "disabled"
      auto: "auto"
```

**转换示例：**
- CNLLM: `thinking=true` → Doubao: `{"thinking": {"type": "enabled"}}`
- CNLLM: `thinking=false` → Doubao: `{"thinking": {"type": "disabled"}}`
- CNLLM: `thinking="auto"` → Doubao: `{"thinking": {"type": "auto"}}`

### 模型名称映射

CNLLM 使用简化模型名，Doubao 使用带版本后缀的原始模型名：

```yaml
model_mapping:
  doubao-seed-2-0-pro: "doubao-seed-2-0-pro-260215"
  doubao-seed-2-0-mini: "doubao-seed-2-0-mini-260215"
  doubao-seed-2-0-lite: "doubao-seed-2-0-lite-260215"
  doubao-seed-2-0-code: "doubao-seed-2-0-code-preview-260215"
  doubao-seed-1-8: "doubao-seed-1-8-251228"
  doubao-seed-1-6: "doubao-seed-1-6-251015"
  doubao-seed-1-6-lite: "doubao-seed-1-6-lite-251015"
  doubao-seed-1-6-flash: "doubao-seed-1-6-flash-250828"
```

## 响应特性

### 标准响应字段

| Doubao 响应字段 | CNLLM/OpenAI 字段 | 说明 |
|:---|:---|:---|
| `id` | `id` | 消息唯一标识 |
| `created` | `created` | 时间戳 |
| `model` | `model` | 模型名称 |
| `choices[0].message.content` | `content` | 纯净文本 |
| `choices[0].message.reasoning_content` | `_thinking` | 推理过程（非标准字段） |
| `choices[0].finish_reason` | `finish_reason` | 结束原因 |
| `choices[0].message.tool_calls` | `tool_calls` | 函数调用 |
| `usage.prompt_tokens` | `prompt_tokens` | 输入 token 数 |
| `usage.completion_tokens` | `completion_tokens` | 输出 token 数 |
| `usage.total_tokens` | `total_tokens` | 总 token 数 |

### 响应格式示例

**非流式响应：**
```json
{
  "id": "02177540717413265b2cf176c77d5b1dd20ddbba02891821aaeda",
  "object": "chat.completion",
  "created": 1775407177,
  "model": "doubao-seed-1-6-251015",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！有什么可以帮助你的吗？",
      "reasoning_content": "用户打招呼，我应该友好回应..."
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

### 属性访问

| 属性 | 来源 | 说明 |
|------|------|------|
| `client.chat.still` | `choices[0].message.content` | 纯净文本 |
| `client.chat.think` | `_thinking` 或 `reasoning_content` | 思考过程 |
| `client.chat.tools` | `choices[0].message.tool_calls` | 函数调用 |
| `client.chat.raw` | 原始响应 | 包含 Doubao 特有字段 |

### 流式累积

流式响应中，内容会被自动累积到对应属性：

```python
client = CNLLM(model="doubao-seed-1-6", api_key="...")

# 流式调用
for chunk in client.chat.create(messages=[...], stream=True, stream_options={}):
    pass

# 获取累积内容
print(client.chat.still)    # 纯净文本
print(client.chat.think)    # 思考过程
print(client.chat.tools)    # 工具调用
```

## 错误处理

### Doubao 错误码体系

**HTTP 状态码：**

| 状态码 | 说明 | CNLLM 异常 |
|--------|------|-----------|
| 200 | 成功 | - |
| 401 | 认证失败 | `AuthenticationError` |
| 403 | 权限不足 | `AuthenticationError` |
| 429 | 请求频率超限 | `RateLimitError` |
| 500 | 服务器内部错误 | `ModelAPIError` |
| 503 | 服务不可用 | `ModelAPIError` |

**业务错误码 (error.code)：**

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| `invalid_request` | 请求格式错误 | 检查请求参数 |
| `InvalidEndpointOrModel.NotFound` | 模型或接入点不存在 | 检查模型名称 |
| `MissingParameter` | 缺少必要参数 | 检查必填参数 |
| `InvalidParameter` | 包含非法参数 | 检查参数合法性 |
| `InvalidEndpoint.ClosedEndpoint` | 接入点已关闭 | 稍后重试 |
| `SensitiveContentDetected` | 内容安全拦截 | 更换内容 |
| `SensitiveContentDetected.SevereViolation` | 严重违规 | 更换内容 |
| `SensitiveContentDetected.Violence` | 激进行为信息 | 更换内容 |
| `InputTextSensitiveContentDetected` | 输入文本敏感 | 更换内容 |
| `InputImageSensitiveContentDetected` | 输入图片敏感 | 更换内容 |

### 错误解析

```python
# cnllm/core/vendor/doubao.py
class DoubaoVendorError(VendorError):
    VENDOR_NAME = "doubao"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["DoubaoVendorError"]:
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
cnllm/core/vendor/doubao.py
├── DoubaoVendorError   # Doubao 特有错误解析
├── DoubaoResponder     # Doubao 响应格式转换
└── DoubaoAdapter      # Doubao 适配器（继承 BaseAdapter）
```

### DoubaoAdapter 实现要点

DoubaoAdapter 继承 BaseAdapter，通过 DoubaoResponder 处理响应格式转换：

```python
class DoubaoAdapter(BaseAdapter):
    ADAPTER_NAME = "doubao"
    CONFIG_DIR = "doubao"

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)
```

流式处理的累积逻辑统一在 BaseAdapter 中处理，确保流式响应正确累积到 `.still`、`.think`、`.tools` 属性。
