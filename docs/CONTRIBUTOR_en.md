# Vendor Adapter Development Guide

This document outlines the complete process for developing a new vendor adapter for CNLLM, based on experience from adapting DeepSeek, GLM, KIMI, MiniMax, Doubao, and Xiaomi.

---

## Architecture Overview

```
User Call Layer (CNLLM.chat.create / embeddings.create)
     │
     ▼
Parameter System (param_registry.py)
  - PARAM_REGISTRY: Global parameter definitions (scope/type/default)
  - validate_for_scope: Validates parameters by capability domain + vendor YAML
     │
     ▼
Adapter Layer (vendor/<vendor>.py)
  - BaseAdapter / BaseEmbeddingAdapter
  - _build_payload: Reads YAML → Builds API request body
  - _get_request_url: Assembles API address
     │
     ▼
HTTP Layer (http.py)
  - BaseHttpClient: httpx sync/async requests
     │
     ▼
Response Layer (responder.py)
  - Responder: Reads response_<vendor>.yaml → Maps to OpenAI standard format
  - StreamAccumulator: Streaming chunks accumulation
     │
     ▼
Response Packaging
  - resp.still / resp.think / resp.tools / resp.raw
```

---

## File Checklist

Adding a new vendor requires the following files:

| File | Purpose | Required |
|------|---------|----------|
| `configs/<vendor>/request_<vendor>.yaml` | Request parameter mapping, model list, error codes | ✅ |
| `configs/<vendor>/response_<vendor>.yaml` | Response field path mapping | ✅ |
| `cnllm/core/vendor/<vendor>.py` | Adapter + Responder + VendorError | ✅ |
| `cnllm/core/vendor/__init__.py` | Register to vendor list | ✅ |
| `tests/key_needed/test_<vendor>_e2e.py` | End-to-end test | Recommended |

---

## Stage 1: YAML Request Configuration

`configs/<vendor>/request_<vendor>.yaml` defines request format, parameter mapping, model name mapping, and error codes.

### Basic Structure

```yaml
request:
  method: "POST"
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${api_key}"

required_fields:
  api_key:
    skip: true        # Not in request body, used as HTTP Header
  model: ~            # ~ means pass through directly
  input:              # Only needed for embedding
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
    map: "thinking.type"          # Field name mapping
    transform:                    # Value transformation
      true: "enabled"
      false: "disabled"
  user:
    map: "user_id"                # Different field name mapping

model_mapping:
  chat:
    deepseek-chat: "deepseek-chat"
    deepseek-reasoner: "deepseek-reasoner"
  embedding:                      # embedding models (optional)
    your-embed-model: "your-embed-model"

error_check:
  code_path: "error.code"
  success_code: null
  message_path: "error.message"
  error_codes:
    InvalidParameter:
      type: "invalid_request_error"
      suggestion: "Request contains invalid parameters"
    RateLimitExceeded:
      type: "rate_limit_error"
      suggestion: "Request rate exceeded, please retry later"
```

### Field Configuration Options

| Config | Meaning | Example |
|--------|---------|---------|
| `~` (null) | Parameter name unchanged, pass through directly | `temperature: ~` |
| `scope: embed` | Only effective for embedding calls | `input: { scope: embed }` |
| `skip: true` | Skip request body (for Header mapping, client parameters) | `api_key: { skip: true }` |
| `map: "xxx"` | Field name mapping | `thinking: { map: "thinking.type" }` |
| `transform` | Value transformation | `true: "enabled"` |
| `default` | Default value | `default: 60.0` |

### Model Mapping

Keys under `model_mapping.chat` and `model_mapping.embedding` are **model names used by CNLLM users**, and values are the model names sent to the API.

```yaml
model_mapping:
  chat:
    glm-4.6: "glm-4.6"           # Same name
    doubao-seed-2-0-pro:          # Different name + vision flag
      model: "doubao-seed-2-0-pro-260215"
      vision: true
```

---

## Stage 2: YAML Response Configuration

`configs/<vendor>/response_<vendor>.yaml` defines vendor response → OpenAI standard format path mapping.

### Non-Streaming Response

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

embedding_fields:                 # embedding response (optional)
  embedding: "data[0].embedding"
  embedding_object: "data[0].object"

embedding_defaults:
  object: "list"
  embedding_object: "embedding"
```

### Streaming Response

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

## Stage 3: Adapter Development

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

### 3.2 Embedding Adapter (Optional)

If the vendor supports embedding models, add an EmbeddingAdapter:

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

### 3.3 Register to Vendor Package

```python
# cnllm/core/vendor/__init__.py
from .<vendor> import <Vendor>Adapter, <Vendor>Responder

__all__.extend(["<Vendor>Adapter", "<Vendor>Responder"])
```

### 3.4 Special Logic Handling

If the vendor's request/response format differs from the standard, override corresponding methods in the adapter:

| Method to Override | Scenario |
|---|---|
| `_build_payload()` | Request body structure differs from standard |
| `_get_request_url()` | URL assembly rules are special |
| `create()` | Call logic differs significantly from standard flow |
| `_to_openai_format()` | Response conversion requires custom logic |
| `_handle_stream()` | Streaming parsing logic differs |
| `check_error()` | Error detection logic differs |

---

## Stage 4: Parameter Registration (Optional)

If the vendor introduces new standard parameters, declare them in `PARAM_REGISTRY` in `cnllm/core/param_registry.py`:

```python
PARAM_REGISTRY = {
    # ...
    "your_new_param": ParamDef(
        types=(str, int),               # Allowed types
        scope={"chat"},                  # Capability domain: chat / embed
        default=None,                    # Default value
        batch_level=False,               # Whether only batch available
    ),
}
```

Parameters declared in `_SKIP_FIELDS` do not need to appear in YAML to be used (e.g., `api_key`, `base_url`).

---

## Stage 5: Testing

### 5.1 Unit Tests

Add tests in the `tests/` directory, covering:

- YAML config loading works correctly
- Parameter mapping is correct
- Payload building is correct
- Response format conversion is correct

### 5.2 E2E Tests

Add end-to-end tests in the `tests/key_needed/` directory:

```python
"""
<Vendor> E2E Test.
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

## Stage 6: Update Documentation

- Add new vendor to model list in `README.md`
- Add new vendor to model list in `README_en.md`
- Add new vendor to model list in `SKILL.md`

---

## Parameter System Principles

### Relationship Between PARAM_REGISTRY and YAML

```
User parameter → validate_for_scope → Filtered parameter → Each vendor adapter
              │
              ├─ PARAM_REGISTRY check: type, scope
              ├─ vendor YAML check: whether allowed in optional_fields
              └─ drop_params strategy: strict/warn/ignore
```

- Parameter in `PARAM_REGISTRY` + scope matches → Pass through directly (skip YAML validation)
- Parameter in `_SKIP_FIELDS` → Skip all validation
- Parameter not in registry but in vendor YAML's `optional_fields` → Pass through (vendor-specific parameter passthrough)
- None match → Handle by `drop_params` strategy

### Type Checking

| Parameter Feature | Handling |
|----------|----------|
| In `PARAM_REGISTRY` + scope matches | ✅ Pass through |
| In `_SKIP_FIELDS` | ✅ Skip (not in payload) |
| In vendor YAML `optional_fields` | ✅ Vendor-specific parameter passthrough |
| None match | ⚠️ `drop_params` handling |

---

## Development Tools

### edit_tool.py

All `.py` code file modifications must be done through `edit_tool.py`:

```python
from edit_tool import edit_file, backup_file, backup_all

edit_file("path/to/file.py", old_text, new_text, description="What changed")
edit_file("path/to/file.py", old_text, new_text, replace_all=True)
backup_file("path/to/file.py")
backup_all()
```

Edit workflow: Read → Edit → Compile verification → Line count check → Backup.

Backups are stored under `backups/<filename>/`, keeping the latest 10 versions for each file.