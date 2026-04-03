# Xiaomi MiMo 适配

## 支持的模型

| 模型 | 说明 |
|------|------|
| `mimo-v2-pro` | 小米 MiMo Pro 版本 |
| `mimo-v2-omni` | 小米 MiMo Omni 全能版本 |
| `mimo-v2-flash` | 小米 MiMo Flash 快速版本 |

## 接口配置

- **API 地址**: `https://api.xiaomimimo.com/v1`
- **端点**: `/text/chatcompletion_v2`
- **协议**: OpenAI 兼容

## 请求特性

| 参数 | 类型 | 支持 | 说明 |
|------|------|------|------|
| `thinking.type` | string | ✅ | 启用思维链，可选 `enabled` / `disabled` |
| `tools` | array | ✅ | 函数调用 |
| `tools[].function.strict` | boolean | ✅ | 严格模式遵循 |
| `stream` | boolean | ✅ | 流式输出 |
| `temperature` | float | ✅ | 温度参数 |
| `max_tokens` | int | ✅ | 最大 token 数 |

## 响应特性

### 标准响应

| 字段 | 说明 |
|------|------|
| `id` | 消息唯一标识 |
| `choices[].message.content` | 纯净文本响应 |
| `choices[].message.reasoning_content` | 推理过程（非标准 OpenAI 字段） |
| `usage` | token 使用统计 |

### 属性访问

| 属性 | 来源 | 说明 |
|------|------|------|
| `client.chat.still` | `choices[].message.content` | 纯净文本 |
| `client.chat.think` | `reasoning_content` | 思考过程 |
| `client.chat.tools` | `tool_calls` | 函数调用 |
| `client.chat.raw` | 原始响应 | 包含平台特有字段 |

### 流式累积

流式响应中，`think`、`still`、`tools` 属性会自动累积：

```python
client = CNLLM(model="mimo-v2-flash", api_key="...")
for chunk in client.chat.create(messages=[...], stream=True):
    pass

print(client.chat.think)   # 累积的推理内容
print(client.chat.still)  # 累积的文本内容
print(client.chat.tools)  # 累积的工具调用
```

## 错误处理

| 错误类型 | 说明 |
|----------|------|
| `AuthenticationError` | API Key 无效或过期 |
| `RateLimitError` | 请求频率超限 |
| `ContentFilteredError` | 内容被过滤 |
| `ModelNotSupportedError` | 不支持的模型 |

## 使用示例

```python
from cnllm import CNLLM

client = CNLLM(model="mimo-v2-flash", api_key="your-api-key")

# 基础调用
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello"}]
)
print(response["choices"][0]["message"]["content"])

# 获取推理过程
print(client.chat.think)

# 函数调用
response = client.chat.create(
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {...}}
        }
    }]
)
print(client.chat.tools)
```
