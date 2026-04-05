# GLM 智谱 AI 适配

## 支持的模型

| 模型 | 说明 | 支持 thinking |
|------|------|--------------|
| `glm-4.6` | GLM-4.6 基线模型 | ✅ |
| `glm-4.7` | GLM-4.7 标准模型 | ✅ |
| `glm-4.7-flash` | GLM-4.7 快速版本 | ✅ |
| `glm-4.7-flashx` | GLM-4.7 极速版 | ✅ |
| `glm-5` | GLM-5 基线模型 | ✅ |
| `glm-5-turbo` | GLM-5 加速版本 | ✅ |

## 接口配置

- **API 地址**: `https://open.bigmodel.cn/api/paas/v4`
- **端点**: `/chat/completions`
- **认证**: `Authorization: Bearer ${api_key}`
- **协议**: 原生 API（非 OpenAI 兼容）

## 请求配置

### 配置文件

- `configs/glm/request_glm.yaml` - 请求参数配置
- `configs/glm/response_glm.yaml` - 响应字段映射

### 字段映射

| CNLLM 参数 | GLM 字段 | 转换说明 |
|:---|:---|:---|
| `model` | `model` | 直接传递 |
| `messages` | `messages` | 直接传递 |
| `temperature` | `temperature` | 直接传递，默认 0.95 |
| `top_p` | `top_p` | 直接传递，默认 0.7 |
| `max_tokens` | `max_tokens` | 直接传递 |
| `stream` | `stream` | 直接传递 |
| `stop` | `stop` | 直接传递（数组） |
| `tools` | `tools` | 直接传递 |
| `tool_choice` | `tool_choice` | 直接传递 |
| `user` | `user_id` | 字段名映射 |
| `thinking` | `thinking.type` | 格式转换 |
| `do_sample` | `do_sample` | 直接透传 |
| `request_id` | `request_id` | 直接透传 |
| `response_format` | `response_format` | 直接透传 |
| `tool_stream` | `tool_stream` | 直接透传 |

### `thinking` 参数转换

CNLLM 使用布尔值 `thinking=true/false`，GLM 使用对象格式 `thinking.type`：

```yaml
# configs/glm/request_glm.yaml
optional_fields:
  thinking:
    body: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
```

**转换示例：**
- CNLLM: `thinking=true` → GLM: `{"thinking": {"type": "enabled"}}`
- CNLLM: `thinking=false` → GLM: `{"thinking": {"type": "disabled"}}`

### `user` 参数映射

CNLLM 的 `user` 参数映射为 GLM 的 `user_id`：

```yaml
optional_fields:
  user:
    body: "user_id"
```

## 响应特性

### 标准响应字段

| GLM 响应字段 | CNLLM/OpenAI 字段 | 说明 |
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
  "id": "8382267772721320722",
  "request_id": "8382267772721320722",
  "created": 1743849600,
  "model": "glm-4.7",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！有什么我可以帮助你的吗？",
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
| `client.chat.raw` | 原始响应 | 包含 GLM 特有字段 |

### 流式累积

流式响应中，内容会被自动累积到对应属性：

```python
client = CNLLM(model="glm-4.7", api_key="...")

# 流式调用
for chunk in client.chat.create(messages=[...], stream=True):
    pass

# 获取累积内容
print(client.chat.still)    # 纯净文本
print(client.chat.think)    # 思考过程
print(client.chat.tools)    # 工具调用
```

## 错误处理

### GLM 错误码体系

**HTTP 状态码：**

| 状态码 | 说明 | CNLLM 异常 |
|--------|------|-----------|
| 200 | 成功 | - |
| 401 | 认证失败 | `AuthenticationError` |
| 403 | 权限不足 | `AuthenticationError` |
| 429 | 请求频率超限 | `RateLimitError` |
| 500 | 服务器内部错误 | `ModelAPIError` |
| 503 | 服务不可用 | `ModelAPIError` |

**业务错误码 (base_resp.status_code)：**

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| 1214 | 内容安全拦截 | 更换内容或联系厂商 |
| 1240 | Token 过长超限 | 减少 max_tokens |
| 1250 | 余额不足 | 检查账户余额 |
| 1414 | 模型不支持该参数 | 检查参数 |
| 1436 | 触发限流 | 降低请求频率 |

### 错误解析

```python
# cnllm/core/vendor/glm.py
class GLMVendorError(VendorError):
    VENDOR_NAME = "glm"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["GLMVendorError"]:
        if not raw_response:
            return None
        base_resp = raw_response.get("base_resp", {})
        code = base_resp.get("status_code")
        if code is None:
            return None
        message = base_resp.get("status_msg", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)
```

## 实现架构

### 核心类

```
cnllm/core/vendor/glm.py
├── GLMVendorError    # GLM 特有错误解析
├── GLMResponder      # GLM 响应格式转换
└── GLMAdapter        # GLM 适配器（继承 BaseAdapter）
```

### GLMAdapter 实现要点

1. **重写 `_build_payload()` 方法**：处理 `thinking` 参数的嵌套结构和 transform
2. **使用 YAML 配置**：字段映射和默认值通过 `configs/glm/request_glm.yaml` 配置
3. **错误解析**：通过 `GLMVendorError.from_response()` 解析 GLM 特有错误

```python
class GLMAdapter(BaseAdapter):
    ADAPTER_NAME = "glm"
    CONFIG_DIR = "glm"

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
                # 处理 transform 和嵌套字段
                transform = field_config.get("transform")
                if transform and value in transform:
                    value = transform[value]
                mapped_key = field_config.get("body", key)
                if mapped_key == "__skip__":
                    continue
                # 处理嵌套路径如 "choices[0].message.content"
                if "." in mapped_key:
                    parts = mapped_key.split(".")
                    current = payload
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
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

client = CNLLM(model="glm-4.7", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response["choices"][0]["message"]["content"])
```

### 启用思考过程

```python
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

### 自定义请求 ID

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100,
    request_id="my_custom_request_id"
)
print(response["id"])  # 自定义 ID 会被保留
```

### JSON 响应格式

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "返回一个JSON对象，包含name和age字段"}],
    max_tokens=100,
    thinking=False,
    response_format={"type": "json_object"}
)
```

### 函数调用

```python
response = client.chat.create(
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名"}
                }
            }
        }
    }]
)

# 获取函数调用结果
print(client.chat.tools)
```

### 流式输出

```python
client = CNLLM(model="glm-4.7", api_key="your-api-key", stream=True)

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "讲个故事"}],
    max_tokens=500
):
    delta = chunk["choices"][0]["delta"]
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
```

### 流式 + thinking

```python
client = CNLLM(model="glm-4.7", api_key="your-api-key", stream=True)

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "解释量子计算"}],
    max_tokens=500,
    thinking=True
):
    delta = chunk["choices"][0]["delta"]
    if delta.get("reasoning_content"):
        print(f"[思考] {delta['reasoning_content']}")
    if delta.get("content"):
        print(f"[回答] {delta['content']}")
```

## 测试验证

### 完整测试覆盖

所有 GLM 模型均通过以下测试：

| 测试类 | 测试项 | 覆盖模型 |
|--------|--------|----------|
| `TestGLMAllModelsBasic` | 基础对话 | 6 模型 |
| `TestGLMAllModelsBasic` | thinking=true | 6 模型 |
| `TestGLMAllModelsBasic` | 流式输出 | 6 模型 |
| `TestGLMAllModelsBasic` | 流式+thinking | 6 模型 |
| `TestGLMAllModelsNative` | response_format | 6 模型 |
| `TestGLMAllModelsNative` | request_id | 6 模型 |
| `TestGLMAllModelsNative` | do_sample | 6 模型 |
| `TestGLMAllModelsMapped` | user 映射 | 6 模型 |
| `TestGLMAllModelsMapped` | thinking 映射 | 6 模型 |
| `TestGLMAllModelsClient` | .still 属性 | 6 模型 |
| `TestGLMAllModelsClient` | .think 属性 | 6 模型 |
| `TestGLMAllModelsOpenAIFormat` | 基础字段 | 6 模型 |
| `TestGLMAllModelsOpenAIFormat` | usage 字段 | 6 模型 |

**总计**: 78 个测试用例，覆盖全部 6 个 GLM 模型。
