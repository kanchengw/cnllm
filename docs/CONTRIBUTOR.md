# 新厂商适配开发指南

本文档梳理开发新厂商适配的标准流程，基于MiniMax和小米mimo的适配开发经验总结。
项目基础框架已基本完善，详细的系统架构可以参阅[系统架构](docs/ARCHITECTURE.md)。
本文主要关注适配新模型或新厂商的适配开发流程，包括厂商适配器的创建、继承、实现等。
欢迎贡献者参与，一起完善CNLLM的适配器库。

## 两种路线

### 路线1：适配厂商的OpenAI兼容接口

- 优势：适配简单，请求前后字段基本一致。
- 劣势：兼容接口一般功能较少，缺少厂商原生功能。
- 案例：小米mimo系列模型适配采用此路线，因为小米官方只提供OpenAI兼容接口。

### 路线2：适配厂商的原生接口

- 优势：功能完善，支持更多厂商原生功能。
- 劣势：适配复杂，需要详细分析厂商 API 的请求和响应格式，需要在vendor适配器中实现厂商特殊逻辑处理。
- 案例：MiniMax M2系列的模型适配采用此路线，支持更多原生接口的独特能力，如：
  `thinking`深度思考模式
  `top_p`最小概率采样
  `mask`掩码输入
  最后以符合OpenAI API规范的格式返回响应。

---

## 开发流程概览

```
┌─────────────────────────────────────────────────────────┐
│  阶段1: 准备                                             │
│    1.1 确认厂商 API 格式                                 │
│    1.2 分析请求/响应差异                                 │
│    1.3 创建配置文件                                      │
├─────────────────────────────────────────────────────────┤
│  阶段2: 配置                                             │
│    2.1 创建 configs/<vendor>/                           │
│    2.2 编写 request_<vendor>.yaml                       │
│    2.3 编写 response_<vendor>.yaml                      │
├─────────────────────────────────────────────────────────┤
│  阶段3: 厂商适配器开发                                       │
│    3.1 创建 vendor/<vendor>.py                          │
│    3.2 继承 BaseAdapter + Responder + VendorError       │
│    3.3 实现厂商特殊逻辑                                  │
├─────────────────────────────────────────────────────────┤
│  阶段4: 测试验证                                         │
│    基础功能验证                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 阶段1: 准备

### 1.1 确认厂商 API 格式

首先确认厂商 API 的：
- Base URL
- API 路径
- 认证方式
- 关键参数

**示例 - 小米：**
```yaml
base_url: "https://api.xiaomimimo.com/v1"
url: "/chat/completions"
headers:
  Content-Type: "application/json"
  Authorization: "Bearer ${api_key}"
```

### 1.2 分析请求/响应差异

对比 OpenAI 标准格式，找出厂商特殊之处：

| 方面 | OpenAI 标准 | 厂商特殊 |
|------|------------|---------|
| 请求 | `thinking` (boolean) | `thinking.type` (string: enabled/disabled) |
| 响应 | `reasoning_content` 无 | `choices[].message.reasoning_content` |
| 参数 | 无 | `tools[].function.strict` |

### 1.3 记录差异点

在 `docs/<vendor>.md` 中记录：
- 厂商自定义参数（请求端）
- 厂商自定义字段（响应端）
- 暂不支持的功能

---

## 阶段2: 配置

### 2.1 目录结构

```
configs/<vendor>/
├── request_<vendor>.yaml   # 请求配置
└── response_<vendor>.yaml  # 响应配置
```

### 2.2 request_<vendor>.yaml

```yaml
request:
  method: "POST"
  url: "/chat/completions"
  base_url: "https://api.<vendor>.com/v1"
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
  stream: ""
  thinking:
    path: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
  user: ""
  # ...其他支持的参数

model_mapping:
  <model-a>: "<model-a>"
  <model-b>: "<model-b>"

error_check:
  code_path: "base_resp.status_code"
  success_code: 0
  message_path: "base_resp.status_msg"
  auth_code: 1004
  error_codes:
    1000:
      type: "unknown_error"
      message: "未知错误"
      suggestion: "请稍后重试"
    # ...其他错误码映射
```

**关键配置项：**

| 配置项 | 说明 |
|--------|------|
| `method` | HTTP 方法，通常为 POST |
| `base_url` | API 基础地址 |
| `url` | API 路径 |
| `required_fields` | 必填字段 |
| `optional_fields` | 可选字段及其转换规则 |
| `model_mapping` | 模型名称映射 |
| `error_check` | 错误检查配置 |

### 2.3 response_<vendor>.yaml

```yaml
response:
  fields:
    id: "id"
    object: "object"
    created: "created"
    model: "model"
    choices: "choices"
    usage: "usage"

  choices:
    index: "index"
    message: "message"
    finish_reason: "finish_reason"

  message:
    role: "role"
    content: "content"

  usage:
    prompt_tokens: "usage.prompt_tokens"
    completion_tokens: "usage.completion_tokens"
    total_tokens: "usage.total_tokens"

stream:
  delta:
    content: "delta.content"
    role: "delta.role"

special_fields:
  reasoning_content:
    path: "choices[].message.reasoning_content"
    description: "推理内容"
```

---

## 阶段3: 适配器开发

### 3.1 创建适配器文件

```python
# cnllm/core/vendor/<vendor>.py
from . import BaseAdapter

class <Vendor>Adapter(BaseAdapter):
    """<厂商名>厂商适配器"""
    VENDOR_NAME = "<vendor>"
```

### 3.2 继承架构组件

新厂商适配需要继承三类组件：

#### 3.2.1 BaseAdapter (核心适配器)

处理请求构建、发送、响应转换等核心逻辑。

**已实现方法：**
- `validate_model()` - 模型名称验证
- `validate_params()` - 参数验证
- `build_payload()` - 构建请求体
- `create_completion()` - 发起请求
- `_to_openai_format()` - 响应格式转换
- `_to_openai_stream_format()` - 流式响应转换
- `_collect_stream_result()` - 流式结果累积

#### 3.2.2 Responder (响应转换器)

处理厂商特定响应字段到 OpenAI 标准格式的转换。

**职责：**
- 提取 `content`、`reasoning_content`、`tool_calls` 等字段
- 处理 `usage` 信息（prompt_tokens、completion_tokens 等）
- 支持流式响应转换
- 支持敏感内容检测

**配置依赖：**
- `configs/<vendor>/response_<vendor>.yaml` 中的 `fields` 映射

#### 3.2.3 VendorError (厂商错误)

解析厂商错误响应，转换为统一错误类型。

**职责：**
- 解析厂商错误码和错误信息
- 支持敏感内容检测（通过 `error_check.sensitive_check` 配置）

**配置依赖：**
- `configs/<vendor>/response_<vendor>.yaml` 中的 `error_check` 配置

### 3.3 实现厂商适配器

根据阶段1的分析，实现厂商适配器类：

```python
# cnllm/core/vendor/<vendor>.py
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry

class <Vendor>VendorError(VendorError):
    VENDOR_NAME = "<vendor>"

    @classmethod
    def from_response(cls, raw_response: dict):
        """从厂商响应解析错误"""
        if not raw_response:
            return None
        error = raw_response.get("error", {})
        code = error.get("code")
        if code is None:
            return None
        message = error.get("message", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)

VendorErrorRegistry.register("<vendor>", <Vendor>VendorError)


class <Vendor>Responder(Responder):
    """厂商响应转换器"""
    CONFIG_DIR = "<vendor>"


class <Vendor>Adapter(BaseAdapter):
    """<厂商名>厂商适配器"""
    ADAPTER_NAME = "<vendor>"
    CONFIG_DIR = "<vendor>"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.responder = <Vendor>Responder()

    def _get_responder(self):
        """返回响应转换器，供 BaseAdapter 委托使用"""
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将厂商响应转换为 OpenAI 格式（委托给 Responder）"""
        return self.responder.to_openai_format(raw, model)

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将厂商流式响应转换为 OpenAI 格式（委托给 Responder）"""
        return self.responder.to_openai_stream_format(raw, model)


<Vendor>Adapter._register()
```

**关键点：**
1. **VendorError** - 错误解析，通过 `error_check` 配置实现敏感内容检测
2. **Responder** - 响应转换，通过 `response_<vendor>.yaml` 配置字段映射
3. **BaseAdapter** - 核心逻辑，请求发送、流式处理等

### 3.4 常见厂商特殊配置

#### 3.4.1 reasoning_content 配置

在 `response_<vendor>.yaml` 中配置字段映射：

```yaml
fields:
  reasoning_content: "choices[0].message.reasoning_content"
```

流式模式下，`_collect_stream_result` 会自动累积 reasoning_content 到 `_thinking`。

#### 3.4.2 tool_calls 配置

在 `response_<vendor>.yaml` 中配置字段映射：

```yaml
fields:
  tool_calls: "choices[0].message.tool_calls"

stream_fields:
  tool_calls_path: "choices[0].delta.tool_calls"
```

#### 3.4.3 usage 详细配置

```yaml
fields:
  prompt_tokens: "usage.prompt_tokens"
  completion_tokens: "usage.completion_tokens"
  total_tokens: "usage.total_tokens"
  reasoning_tokens: "usage.completion_tokens_details.reasoning_tokens"
  cached_tokens: "usage.prompt_tokens_details.cached_tokens"
```

#### 3.4.4 敏感内容检测配置

在 `response_<vendor>.yaml` 中配置错误检测：

```yaml
error_check:
  sensitive_check:
    input_sensitive_type_path: "input_sensitive_type"
    output_sensitive_type_path: "output_sensitive_type"
```

#### 3.4.5 请求字段转换配置

在 `request_<vendor>.yaml` 中配置请求字段映射：

```yaml
optional_fields:
  thinking:
    path: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
  tools:
    path: "tools"
    keep_array_structure: true
```

## 附录: 小米适配经验总结

### 小米 vs OpenAI 差异点

| 方面 | OpenAI | 小米 |
|------|--------|------|
| thinking 参数 | `thinking: true` | `thinking.type: "enabled"` |
| 响应 reasoning_content | 无 | `choices[].message.reasoning_content` |
| tools.strict | 不支持 | 支持 `tools[].function.strict` |
| 内置工具 | 无 | 自动调用 tool_calls |

---

## 检查清单

新增厂商适配时，确保完成以下检查：

- [ ] `configs/<vendor>/request_<vendor>.yaml` 创建
- [ ] `configs/<vendor>/response_<vendor>.yaml` 创建
- [ ] `cnllm/core/vendor/<vendor>.py` 创建
- [ ] 继承 BaseAdapter + Responder + VendorError 三类组件
- [ ] `_get_responder()` 方法返回 Responder 实例
- [ ] `model_mapping` 配置完整
- [ ] VendorError 注册到 VendorErrorRegistry
- [ ] 基础对话测试通过
- [ ] 流式输出测试通过
- [ ] `.think`、`.still`、`.tools` 属性正常
- [ ] 敏感内容检测正常
- [ ] 标准结构无多余字段
- [ ] 文档 `docs/vendor/<vendor>.md` 更新


## 阶段4: 测试验证

### 4.1 基础对话测试

```python
client = CNLLM(model="<model>", api_key="<key>")
resp = client.chat.create(messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}])
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}])
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}], thinking=True)
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}], stream=True)
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}], stream=True, thinking=True)
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "伊斯坦布尔的天气如何"}], tools=[{"type": "function", "function": {"name": "get_weather", ...} }])
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)
print(f"====="*20)
print(resp.tools)

resp = client.chat.create(messages=[{"role": "user", "content": "伊斯坦布尔的天气如何"}], tools=[{"type": "function", "function": {"name": "get_weather", ...} }], stream=True)
print(resp)
print(f"====="*20) 
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)
print(f"====="*20)
print(resp.tools)
```

**验证点：**
- 确认能够正常工作，不报错
- 观察resp是否符合openai标准结构，不包含其他额外字段（如content不能包含.think（reasoning_content）的思考过程）
- 观察.still是否获取纯净输出
- 观察.think是否获取纯净思考过程
- 观察流式响应中.raw .think .still .tools输出是否实时累积
- 观察.tools是否获取纯净工具调用信息


### 4.2 流式输出测试

```python
client = CNLLM(model="<model>", api_key="<key>", stream=True)
chunks = []
resp = client.chat.create(
    messages=[{"role": "user", "content": "详细解释天空为什么是蓝色的"}],
    thinking=True
)
for i, chunk in enumerate(resp):
    chunks.append(chunk)
    if i < 20:
        print(f"[Chunk {i}] .think: {client.chat.think[:50] if client.chat.think else None}...")
        print(f"[Chunk {i}] .still: {client.chat.still[:50] if client.chat.still else None}...")
        print(f"[Chunk {i}] delta: {chunk.get('choices', [{}])[0].get('delta', {})}")
    elif i == 20:
        print("... (超过20个chunk，不再打印中间过程)")

print(f"\n共 {len(chunks)} 个 chunks")
print(f"====="*20)
print(f".think (完整): {client.chat.think}")
print(f"====="*20)
print(f".still (完整): {client.chat.still}")
print(f"====="*20)
print(f"resp (完整): {chunks[-1] if chunks else None}")
```

**验证点：**
- ✅ 每个 chunk 有 `id`, `object`, `choices`
- ✅ `delta` 包含 `content` 或 `role`
- ✅ `.think` 和 `.still` 可实时累积
- ✅ reasoning_content 不出现在 resp 中

### 4.3 标准结构验证

```python
def check_standard_format(resp):
    """检查响应是否符合 OpenAI 标准格式"""
    required_keys = {"id", "object", "created", "model", "choices", "usage"}
    actual_keys = set(resp.keys())

    extra = actual_keys - required_keys
    if extra:
        print(f"多余字段: {extra}")

    choice = resp.get("choices", [{}])[0]
    choice_keys = set(choice.keys())
    expected_choice_keys = {"index", "message", "finish_reason"}

    message = choice.get("message", {})
    message_keys = set(message.keys())
    expected_message_keys = {"role", "content"}

    print(f"响应结构: {'✅ 标准' if not extra else '❌ 有多余'}")

## scripts 说明

### validate_model_compatible.py

模型兼容性验证脚本，用于测试新模型是否能被现有适配器正确适配。

**功能：**
- 测试已支持模型的兼容性
- 测试潜在兼容模型（如 M2.1 可能兼容 M2.7 系列）
- 测试流式输出
- 测试 Fallback 机制
- 测试 LangChain Runnable 集成

**环境变量：**
- `MINIMAX_API_KEY` - MiniMax API Key
- `XIAOMI_API_KEY` - Xiaomi API Key（可选）
- `CNLLM_SKIP_MODEL_VALIDATION=true` - 跳过模型映射验证（测试未收录模型时）

**使用：**
```bash
python scripts/validate_model_compatible.py
```

### test_e2e_installed.py

端到端测试脚本，模拟用户通过 pip install 安装 cnllm 后的生产环境使用。

**特点：**
- 不引用项目本地模块，使用已安装的 cnllm 包
- 验证安装后包能否正常工作
- 测试基础对话、流式输出、Fallback 等功能

**环境变量：**
- `MINIMAX_API_KEY` - 必需

**使用：**
```bash
python scripts/test_e2e_installed.py
```
