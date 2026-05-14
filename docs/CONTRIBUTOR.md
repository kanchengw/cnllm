# 厂商适配开发指南

本文档梳理为一个新厂商适配 CNLLM 的完整流程，基于 DeepSeek、GLM、KIMI、MiniMax、Doubao、Xiaomi 的适配经验总结。

---

## 架构概述

```
用户调用层 (CNLLM.chat.create / embeddings.create)
     │
     ▼
参数系统 (param_registry.py)
  - PARAM_REGISTRY: 全局参数定义（scope/type/default）
  - validate_for_scope: 按功能域 + 厂商 YAML 验证参数
     │
     ▼
适配器层 (vendor/<vendor>.py)
  - BaseAdapter / BaseEmbeddingAdapter
  - _build_payload: 读取 YAML → 构建 API 请求体
  - _get_request_url: 拼接 API 地址
     │
     ▼
HTTP 层 (http.py)
  - BaseHttpClient: httpx 同步/异步请求
     │
     ▼
响应层 (responder.py)
  - Responder: 读取 response_<vendor>.yaml → 映射为 OpenAI 标准格式
  - StreamAccumulator: 流式 chunks 累积
     │
     ▼
响应封装
  - resp.still / resp.think / resp.tools / resp.raw
```

---

## 文件清单

添加一个新厂商需要涉及以下文件：

| 文件 | 用途 | 必须 |
|------|------|------|
| `configs/<vendor>/request_<vendor>.yaml` | 请求参数映射、模型列表、错误码 | ✅ |
| `configs/<vendor>/response_<vendor>.yaml` | 响应字段路径映射 | ✅ |
| `cnllm/core/vendor/<vendor>.py` | Adapter + Responder + VendorError | ✅ |
| `cnllm/core/vendor/__init__.py` | 注册到厂商列表 | ✅ |
| `tests/key_needed/test_<vendor>_e2e.py` | 端到端测试 | 推荐 |

---

## 阶段一：YAML 请求配置

`configs/<vendor>/request_<vendor>.yaml` 定义了请求格式、参数映射、模型名映射和错误码。

### 基础结构

```yaml
request:
  method: "POST"
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${api_key}"

required_fields:
  api_key:
    skip: true        # 不进请求体，用作 HTTP Header
  model: ~            # ~ 表示直接传递
  input:              # 仅 embedding 需要
    scope: embed

one_of:
  messages_or_prompt:
    messages: ~
    prompt: ~

optional_fields:
  base_url:
    skip: true
    chat:
      default: "https://api.example.com/v1"
      path: "chat/completions"
    embedding:
      default: "https://api.example.com/v1"
      path: "embeddings"
  timeout:
    skip: true
    default: 60.0
  temperature: ~
  top_p: ~
  max_tokens: ~
  stream: ~
  stop: ~
  tools: ~
  tool_choice: ~
  thinking:
    map: "thinking.type"          # 字段名映射
    transform:                    # 值转换
      true: "enabled"
      false: "disabled"
  user:
    map: "user_id"                # 不同字段名映射

model_mapping:
  chat:
    deepseek-chat: "deepseek-chat"
    deepseek-reasoner: "deepseek-reasoner"
  embedding:                      # embedding 模型（可选）
    your-embed-model: "your-embed-model"

error_check:
  code_path: "error.code"
  success_code: null
  message_path: "error.message"
  error_codes:
    InvalidParameter:
      type: "invalid_request_error"
      suggestion: "请求包含非法参数"
    RateLimitExceeded:
      type: "rate_limit_error"
      suggestion: "请求频率超限，请稍后重试"
```

### 字段配置选项

| 配置 | 含义 | 示例 |
|------|------|------|
| `~` (null) | 参数名不变，直接透传 | `temperature: ~` |
| `scope: embed` | 仅 embedding 调用生效 | `input: { scope: embed }` |
| `skip: true` | 跳过请求体（用于 Header 映射、客户端参数） | `api_key: { skip: true }` |
| `map: "xxx"` | 字段名映射 | `thinking: { map: "thinking.type" }` |
| `transform` | 值转换 | `true: "enabled"` |
| `default` | 默认值 | `default: 60.0` |

### 模型映射

`model_mapping.chat` 和 `model_mapping.embedding` 下的 key 是 **CNLLM 用户使用的模型名**，value 是传给 API 的模型名。

```yaml
model_mapping:
  chat:
    glm-4.6: "glm-4.6"           # 相同名称
    doubao-seed-2-0-pro:          # 不同名称 + 视觉标记
      model: "doubao-seed-2-0-pro-260215"
      vision: true
```

---

## 阶段二：YAML 响应配置

`configs/<vendor>/response_<vendor>.yaml` 定义厂商响应 → OpenAI 标准格式的路径映射。

### 非流式响应

```yaml
fields:
  id: "id"
  created: "created"
  model: "model"
  content: "choices[0].message.content"
  tool_calls: "choices[0].message.tool_calls"
  reasoning_content: "choices[0].message.reasoning_content"
  prompt_tokens: "usage.prompt_tokens"
  completion_tokens: "usage.completion_tokens"
  total_tokens: "usage.total_tokens"

defaults:
  object: "chat.completion"
  index: 0
  role: "assistant"
  finish_reason: "stop"

embedding_fields:                 # embedding 响应（可选）
  embedding: "data[0].embedding"
  embedding_object: "data[0].object"

embedding_defaults:
  object: "list"
  embedding_object: "embedding"
```

### 流式响应

```yaml
stream_fields:
  object: "chat.completion.chunk"
  index: 0
  role: "assistant"
  finish_reason: null
  content_path:
    path: "choices[0].delta.content"
    accumulate: true
  tool_calls_path:
    path: "choices[0].delta.tool_calls"
    accumulate: true
  reasoning_content_path:
    path: "choices[0].delta.reasoning_content"
    accumulate: true
```

---

## 阶段三：适配器开发

### 3.1 Chat Adapter

```python
# cnllm/core/vendor/<vendor>.py
import logging
from typing import Dict, Any, Optional
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry

logger = logging.getLogger(__name__)


class <Vendor>VendorError(VendorError):
    VENDOR_NAME = "<vendor>"

    @classmethod
    def from_response(cls, raw_response: dict) -> Optional["<Vendor>VendorError"]:
        error = raw_response.get("error", {})
        code = error.get("code")
        if code is None:
            return None
        message = error.get("message", "")
        return cls(code=code, message=message, vendor=cls.VENDOR_NAME, raw_response=raw_response)


VendorErrorRegistry.register("<vendor>", <Vendor>VendorError)


class <Vendor>Responder(Responder):
    CONFIG_DIR = "<vendor>"


class <Vendor>Adapter(BaseAdapter):
    ADAPTER_NAME = "<vendor>"
    CONFIG_DIR = "<vendor>"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.responder = <Vendor>Responder()

    def _get_responder(self):
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_format(raw, model)

    def _do_to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        return self.responder.to_openai_stream_format(raw, model)


<Vendor>Adapter._register()
```

### 3.2 Embedding Adapter（可选）

如果厂商支持 embedding 模型，需要添加 EmbeddingAdapter：

```python
from ..embedding import BaseEmbeddingAdapter, EmbeddingResponder


class <Vendor>EmbeddingAdapter(BaseEmbeddingAdapter):
    ADAPTER_NAME = "<vendor>"
    CONFIG_DIR = "<vendor>"

    def __init__(self, api_key: str, model: str = None, base_url: str = None, **kwargs):
        super().__init__(
            api_key=api_key, model=model, base_url=base_url,
            config_file=f"request_{self.CONFIG_DIR}.yaml", **kwargs
        )

    @classmethod
    def _load_class_config(cls):
        if cls._class_config is not None:
            return cls._class_config
        import yaml, os
        config_path = os.path.join(
            os.path.dirname(__file__), "../../..", "configs", cls.CONFIG_DIR,
            f"request_{cls.CONFIG_DIR}.yaml"
        )
        try:
            with open(config_path) as f:
                cls._class_config = yaml.safe_load(f) or {}
                mapping = cls._class_config.get("model_mapping", {})
                if isinstance(mapping, dict) and "embedding" in mapping:
                    mapping = mapping["embedding"]
                cls._supported_models = list(mapping.keys()) if isinstance(mapping, dict) else []
                return cls._class_config
        except FileNotFoundError:
            cls._class_config = {}
            cls._supported_models = []
            return {}

    def _get_responder(self) -> EmbeddingResponder:
        return EmbeddingResponder(self.CONFIG_DIR)


<Vendor>EmbeddingAdapter._register()
```

### 3.3 注册到 vendor 包

```python
# cnllm/core/vendor/__init__.py
from .<vendor> import <Vendor>Adapter, <Vendor>Responder

__all__.extend(["<Vendor>Adapter", "<Vendor>Responder"])
```

### 3.4 特殊逻辑处理

如果厂商的请求/响应格式与标准不同，可在 adapter 中覆写对应方法：

| 需要覆写的方法 | 场景 |
|---|---|
| `_build_payload()` | 请求体结构与标准不同 |
| `_get_request_url()` | URL 拼接规则特殊 |
| `create()` | 调用逻辑与标准流程差异大 |
| `_to_openai_format()` | 响应转换需要自定义逻辑 |
| `_handle_stream()` | 流式解析逻辑不同 |
| `check_error()` | 错误检测逻辑不同 |

---

## 阶段四：参数注册（可选）

如果厂商引入了新的标准参数，需要在 `cnllm/core/param_registry.py` 的 `PARAM_REGISTRY` 中声明：

```python
PARAM_REGISTRY = {
    # ...
    "your_new_param": ParamDef(
        types=(str, int),               # 允许的类型
        scope={"chat"},                  # 功能域：chat / embed
        default=None,                    # 默认值
        batch_level=False,               # 是否仅 batch 可用
    ),
}
```

`_SKIP_FIELDS` 中声明的参数不需要在 YAML 中出现即可使用（如 `api_key`、`base_url`）。

---

## 阶段五：测试

### 5.1 单元测试

在 `tests/` 目录添加测试，覆盖：

- YAML 配置加载是否正常
- 参数映射是否正确
- payload 构建是否正确
- 响应格式转换是否正确

### 5.2 E2E 测试

在 `tests/key_needed/` 目录添加端到端测试：

```python
"""
<Vendor> E2E 测试。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("<VENDOR>_API_KEY")
MODEL = "<vendor-model>"


def test_chat_basic():
    if not API_KEY:
        print("SKIP: no API key"); return
    from cnllm import CNLLM
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.create(prompt="1+1=?")
    assert resp.still is not None
    client.close()


if __name__ == "__main__":
    test_chat_basic()
    print("Done.")
```

---

## 阶段六：更新文档

- `README.md` 模型列表添加新厂商
- `README_en.md` 模型列表添加新厂商
- `SKILL.md` 模型列表添加新厂商

---

## 参数系统原理

### PARAM_REGISTRY 与 YAML 的关系

```
用户参数 → validate_for_scope → 过滤后参数 → 各厂商 adapter
              │
              ├─ PARAM_REGISTRY 检查：类型、scope
              ├─ vendor YAML 检查：optional_fields 中是否允许
              └─ drop_params 策略：strict/warn/ignore
```

- 参数如果在 `PARAM_REGISTRY` 中且 scope 匹配 → 直接通过（不走 YAML 验证）
- 参数如果在 `_SKIP_FIELDS` 中 → 跳过所有验证
- 参数不在 registry 但在 vendor YAML 的 `optional_fields` 中 → 通过（厂商特有参数透传）
- 参数都不在 → 按 `drop_params` 策略处理

### 类型判断

| 参数特征 | 处理方式 |
|----------|----------|
| 在 `PARAM_REGISTRY` + scope 匹配 | ✅ 通过 |
| 在 `_SKIP_FIELDS` | ✅ 跳过（不进 payload） |
| 在 vendor YAML `optional_fields` | ✅ 厂商特有参数透传 |
| 都不匹配 | ⚠️ `drop_params` 处理 |

---

## 开发工具

### edit_tool.py

所有 `.py` 代码文件的修改必须通过 `edit_tool.py` 操作：

```python
from edit_tool import edit_file, backup_file, backup_all

edit_file("path/to/file.py", old_text, new_text, description="改了什么")
edit_file("path/to/file.py", old_text, new_text, replace_all=True)
backup_file("path/to/file.py")
backup_all()
```

编辑流程：读取 → 编辑 → 编译验证 → 行数检查 → 备份。

备份存储在 `backups/<文件名>/` 下，每个文件保留最新 10 份。
