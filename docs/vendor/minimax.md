# MiniMax 适配

## 支持的模型

| CNLLM 模型 | MiniMax 原始模型 | 说明 |
|------------|------------------|------|
| `minimax-m2` | `MiniMax-M2` | MiniMax M2 基线模型 |
| `minimax-m2.1` | `MiniMax-M2.1` | MiniMax M2.1 版本 |
| `minimax-m2.5` | `MiniMax-M2.5` | MiniMax M2.5 版本 |
| `minimax-m2.7` | `MiniMax-M2.7` | MiniMax M2.7 版本 |

## 接口配置

- **API 地址**: `https://api.minimaxi.com/v1`
- **端点**: `/text/chatcompletion_v2`
- **认证**: `Authorization: Bearer ${api_key}`
- **协议**: OpenAI 兼容

## 请求配置

### 配置文件

- `configs/minimax/request_minimax.yaml` - 请求参数配置
- `configs/minimax/response_minimax.yaml` - 响应字段映射

### 字段映射

CNLLM 参数与 MiniMax API 字段的对应关系：

| CNLLM 参数 | MiniMax 字段 | 转换说明 |
|:---|:---|:---|
| `model` | `model` | 通过 model_mapping 映射 |
| `messages` | `messages` | 直接传递 |
| `prompt` | `prompt` | 直接传递（替代 messages） |
| `temperature` | `temperature` | 直接传递 |
| `max_tokens` | `max_completion_tokens` | 字段名映射 |
| `stream` | `stream` | 直接传递 |
| `top_p` | `top_p` | 直接传递 |
| `top_k` | `top_k` | 直接传递 |
| `tools` | `tools` | 直接传递 |
| `tool_choice` | `tool_choice` | 直接传递 |
| `thinking` | `thinking` | 直接传递（格式一致） |
| `presence_penalty` | `presence_penalty` | 直接传递 |
| `frequency_penalty` | `frequency_penalty` | 直接传递 |
| `stop` | `stop` | 直接传递（数组） |
| `user` | `user` | 直接传递 |
| `mask` | `mask` | 直接传递（MiniMax 特有） |
| `organization` | `group_id` | 字段名映射（放在 header） |

### `max_tokens` 字段映射

MiniMax API 使用 `max_completion_tokens` 而非 `max_tokens`：

```yaml
# configs/minimax/request_minimax.yaml
optional_fields:
  max_tokens:
    body: "max_completion_tokens"
```

### `organization` 参数

MiniMax 使用 `group_id` 而非 `organization`，且放在 HTTP header 中：

```yaml
optional_fields:
  organization:
    body: "__skip__"
    head: "group_id"
```

### `thinking` 参数

MiniMax 的 `thinking` 参数与 CNLLM 格式一致，无需转换：

```yaml
optional_fields:
  thinking: ""
```

### 模型名称映射

CNLLM 使用简化模型名，MiniMax 使用原始模型名：

```yaml
model_mapping:
  minimax-m2: "MiniMax-M2"
  minimax-m2.1: "MiniMax-M2.1"
  minimax-m2.5: "MiniMax-M2.5"
  minimax-m2.7: "MiniMax-M2.7"
```

## 响应特性

### 标准响应字段

| MiniMax 响应字段 | CNLLM/OpenAI 字段 | 说明 |
|:---|:---|:---|
| `id` | `id` | 消息唯一标识 |
| `created` | `created` | 时间戳 |
| `model` | `model` | 模型名称 |
| `choices[0].message.content` | `content` | 纯净文本 |
| `choices[0].message.reasoning_content` | `reasoning_content` / `_thinking` | 推理过程 |
| `choices[0].finish_reason` | `finish_reason` | 结束原因 |
| `choices[0].message.tool_calls` | `tool_calls` | 函数调用 |
| `usage.prompt_tokens` | `prompt_tokens` | 输入 token 数 |
| `usage.completion_tokens` | `completion_tokens` | 输出 token 数 |
| `usage.total_tokens` | `total_tokens` | 总 token 数 |
| `usage.completion_tokens_details.reasoning_tokens` | `reasoning_tokens` | 推理 token 数 |

### 响应格式示例

**非流式响应：**
```json
{
  "id": "1234567890@example.com",
  "created": 1743849600,
  "model": "MiniMax-M2",
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
    "total_tokens": 40,
    "completion_tokens_details": {
      "reasoning_tokens": 10
    }
  }
}
```

### 属性访问

| 属性 | 来源 | 说明 |
|------|------|------|
| `client.chat.still` | `choices[0].message.content` | 纯净文本 |
| `client.chat.think` | `reasoning_content` | 思考过程 |
| `client.chat.tools` | `choices[0].message.tool_calls` | 函数调用 |
| `client.chat.raw` | 原始响应 | 包含平台特有字段 |

### 流式累积

流式响应中，内容会被自动累积到对应属性：

```python
client = CNLLM(model="minimax-m2", api_key="...")

# 流式调用
for chunk in client.chat.create(messages=[...], stream=True):
    pass

# 获取累积内容
print(client.chat.still)    # 纯净文本
print(client.chat.think)    # 思考过程
print(client.chat.tools)    # 工具调用
```

### Usage Details 扩展

MiniMax 响应包含详细的 token 使用统计：

```python
response = client.chat.create(messages=[...])

usage = response["usage"]
print(f"推理 token 数: {usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)}")
```

### 内容安全检查

MiniMax 响应包含内容安全检查结果：

```yaml
error_check:
  sensitive_check:
    input_sensitive_type_path: "input_sensitive_type"
    output_sensitive_type_path: "output_sensitive_type"
```

## 错误处理

### MiniMax 错误码体系

**HTTP 状态码：**

| 状态码 | 说明 | CNLLM 异常 |
|--------|------|-----------|
| 200 | 成功 | - |
| 401 | 认证失败 | `AuthenticationError` |
| 403 | 权限不足 | `AuthenticationError` |
| 429 | 请求频率超限 | `RateLimitError` |
| 500 | 服务器内部错误 | `ModelAPIError` |

**业务错误码 (base_resp.status_code)：**

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| 1000 | 通用错误 | 检查请求参数 |
| 1001 | 服务不可用 | 稍后重试 |
| 1002 | 无效参数 | 检查请求格式 |
| 1003 | 认证失败 | 检查 API Key |
| 1004 | 余额不足 | 检查账户余额 |
| 1005 | 限流 | 降低请求频率 |

### 错误解析

```python
# cnllm/core/vendor/minimax.py
class MiniMaxVendorError(VendorError):
    VENDOR_NAME = "minimax"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["MiniMaxVendorError"]:
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
cnllm/core/vendor/minimax.py
├── MiniMaxVendorError    # MiniMax 特有错误解析
├── MiniMaxResponder      # MiniMax 响应格式转换
└── MiniMaxAdapter       # MiniMax 适配器（继承 BaseAdapter）
```

### MiniMaxAdapter 实现要点

1. **模型映射**：通过 `model_mapping` 将 CNLLM 模型名转换为 MiniMax 原始模型名
2. **字段映射**：`max_tokens` → `max_completion_tokens`
3. **header 处理**：`organization` → `group_id`（放在 HTTP header）
4. **响应转换**：通过 `MiniMaxResponder` 统一转换为 OpenAI 格式

```python
class MiniMaxAdapter(BaseAdapter):
    ADAPTER_NAME = "minimax"
    CONFIG_DIR = "minimax"

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        result = self.responder.to_openai_stream_format(raw, model)
        self._collect_stream_result(result)
        return result
```

## 使用示例

### 基础调用

```python
from cnllm import CNLLM

client = CNLLM(model="minimax-m2", api_key="your-api-key")

response = client.chat.create(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response["choices"][0]["message"]["content"])
```

### 指定组织

MiniMax 需要 `organization` 参数（映射为 `group_id`）：

```python
client = CNLLM(
    model="minimax-m2",
    api_key="your-api-key",
    organization="your-group-id"
)
```

### 思考过程

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

# 获取推理 token 统计
usage = response["usage"]
print(f"推理 token: {usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)}")
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
client = CNLLM(model="minimax-m2", api_key="your-api-key", stream=True)

for chunk in client.chat.create(
    messages=[{"role": "user", "content": "讲个故事"}],
    max_tokens=500
):
    delta = chunk["choices"][0]["delta"]
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
```

### 流式 + thinking + token 统计

```python
client = CNLLM(model="minimax-m2", api_key="your-api-key", stream=True)

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

# 获取完整统计
print(f"总 token: {response['usage']['total_tokens']}")
```
