# New Vendor Adapter Development Guide

This document outlines the standard process for developing a new vendor adapter, based on MiniMax and Xiaomi mimo series adapter experience.

The core framework is now complete. For detailed system architecture, please refer to [System Architecture](docs/ARCHITECTURE_en.md).

This guide focuses on the development process for adapting new models or new vendors, including vendor adapter creation, inheritance, implementation, and more.

Contributors are welcome to participate and help improve the CNLLM adapter library.

## Two Approaches

### Approach 1: Adapting Vendor's OpenAI Compatible Interface

- **Pros**: Simple adaptation, request/response fields are mostly consistent.
- **Cons**: Compatible interfaces usually have fewer features, missing vendor-native capabilities.
- **Example**: Xiaomi mimo series adapter uses this approach, as Xiaomi only provides OpenAI compatible interfaces.

### Approach 2: Adapting Vendor's Native Interface

- **Pros**: Complete feature set, supports more vendor-native capabilities.
- **Cons**: Complex adaptation, requires detailed analysis of vendor API request/response formats, needs special logic handling in the vendor adapter.
- **Example**: MiniMax M2 series adapter uses this approach, supporting native interface capabilities like:
  - `thinking` deep reasoning mode
  - `top_p` minimum probability sampling
  - `mask` input masking
  - Returns responses in OpenAI API compliant format.

---

## Development Process Overview

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Preparation                                  │
│    1.1 Confirm vendor API format                       │
│    1.2 Analyze request/response differences             │
│    1.3 Create configuration files                       │
├─────────────────────────────────────────────────────────┤
│  Phase 2: Configuration                                │
│    2.1 Create configs/<vendor>/                        │
│    2.2 Write request_<vendor>.yaml                    │
│    2.3 Write response_<vendor>.yaml                   │
├─────────────────────────────────────────────────────────┤
│  Phase 3: Adapter Development                          │
│    3.1 Create vendor/<vendor>.py                      │
│    3.2 Inherit BaseAdapter + Responder + VendorError  │
│    3.3 Implement vendor-specific logic                 │
├─────────────────────────────────────────────────────────┤
│  Phase 4: Testing & Validation                          │
│    Basic functionality verification                    │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Preparation

### 1.1 Confirm Vendor API Format

First, confirm the vendor API's:
- Base URL
- API path
- Authentication method
- Key parameters

**Example - Xiaomi:**
```yaml
base_url: "https://api.xiaomimimo.com/v1"
url: "/chat/completions"
headers:
  Content-Type: "application/json"
  Authorization: "Bearer ${api_key}"
```

### 1.2 Analyze Request/Response Differences

Compare with OpenAI standard format to identify vendor-specific aspects:

| Aspect | OpenAI Standard | Vendor Specific |
|--------|-----------------|-----------------|
| Request | `thinking` (boolean) | `thinking.type` (string: enabled/disabled) |
| Response | `reasoning_content` N/A | `choices[].message.reasoning_content` |
| Parameters | N/A | `tools[].function.strict` |

### 1.3 Document Differences

Record in `docs/<vendor>.md`:
- Vendor custom parameters (request side)
- Vendor custom fields (response side)
- Unsupported features

---

## Phase 2: Configuration

### 2.1 Directory Structure

```
configs/<vendor>/
├── request_<vendor>.yaml   # Request configuration
└── response_<vendor>.yaml  # Response configuration
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
  # ...other supported parameters

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
      message: "Unknown error"
      suggestion: "Please try again later"
    # ...other error code mappings
```

**Key Configuration Items:**

| Item | Description |
|------|-------------|
| `method` | HTTP method, usually POST |
| `base_url` | API base address |
| `url` | API path |
| `required_fields` | Required fields |
| `optional_fields` | Optional fields and transformation rules |
| `model_mapping` | Model name mapping |
| `error_check` | Error checking configuration |

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
    description: "Reasoning content"
```

---

## Phase 3: Adapter Development

### 3.1 Create Adapter File

```python
# cnllm/core/vendor/<vendor>.py
from . import BaseAdapter

class <Vendor>Adapter(BaseAdapter):
    """<Vendor> vendor adapter"""
    VENDOR_NAME = "<vendor>"
```

### 3.2 Inherit Architecture Components

New vendor adapters need to inherit three types of components:

#### 3.2.1 BaseAdapter (Core Adapter)

Handles core logic like request building, sending, and response transformation.

**Implemented Methods:**
- `validate_model()` - Model name validation
- `validate_params()` - Parameter validation
- `build_payload()` - Build request body
- `create_completion()` - Send request
- `_to_openai_format()` - Response format conversion
- `_to_openai_stream_format()` - Streaming response conversion
- `_collect_stream_result()` - Stream result accumulation

#### 3.2.2 Responder (Response Transformer)

Handles conversion of vendor-specific response fields to OpenAI standard format.

**Responsibilities:**
- Extract `content`, `reasoning_content`, `tool_calls` fields
- Process `usage` information (prompt_tokens, completion_tokens, etc.)
- Support streaming response conversion
- Support sensitive content detection

**Configuration Dependency:**
- `fields` mapping in `configs/<vendor>/response_<vendor>.yaml`

#### 3.2.3 VendorError (Vendor Errors)

Parses vendor error responses and converts them to unified error types.

**Responsibilities:**
- Parse vendor error codes and messages
- Support sensitive content detection (via `error_check.sensitive_check` config)

**Configuration Dependency:**
- `error_check` config in `configs/<vendor>/response_<vendor>.yaml`

### 3.3 Implement Vendor Adapter

Based on Phase 1 analysis, implement the vendor adapter class:

```python
# cnllm/core/vendor/<vendor>.py
from ..adapter import BaseAdapter
from ..responder import Responder
from ...utils.vendor_error import VendorError, VendorErrorRegistry

class <Vendor>VendorError(VendorError):
    VENDOR_NAME = "<vendor>"

    @classmethod
    def from_response(cls, raw_response: dict):
        """Parse error from vendor response"""
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
    """Vendor response transformer"""
    CONFIG_DIR = "<vendor>"


class <Vendor>Adapter(BaseAdapter):
    """<Vendor> vendor adapter"""
    ADAPTER_NAME = "<vendor>"
    CONFIG_DIR = "<vendor>"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.responder = <Vendor>Responder()

    def _get_responder(self):
        """Return the responder for BaseAdapter delegation"""
        return self.responder

    def _to_openai_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert vendor response to OpenAI format (delegated to Responder)"""
        return self.responder.to_openai_format(raw, model)

    def _to_openai_stream_format(self, raw: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert vendor streaming response to OpenAI format (delegated to Responder)"""
        return self.responder.to_openai_stream_format(raw, model)


<Vendor>Adapter._register()
```

**Key Points:**
1. **VendorError** - Error parsing, sensitive content detection via `error_check` config
2. **Responder** - Response transformation, field mapping via `response_<vendor>.yaml`
3. **BaseAdapter** - Core logic, request sending, streaming handling, etc.

### 3.4 Common Vendor-Specific Configurations

#### 3.4.1 reasoning_content Configuration

Configure field mapping in `response_<vendor>.yaml`:

```yaml
fields:
  reasoning_content: "choices[0].message.reasoning_content"
```

In streaming mode, `_collect_stream_result` automatically accumulates reasoning_content to `_thinking`.

#### 3.4.2 tool_calls Configuration

Configure field mapping in `response_<vendor>.yaml`:

```yaml
fields:
  tool_calls: "choices[0].message.tool_calls"

stream_fields:
  tool_calls_path: "choices[0].delta.tool_calls"
```

#### 3.4.3 usage Detailed Configuration

```yaml
fields:
  prompt_tokens: "usage.prompt_tokens"
  completion_tokens: "usage.completion_tokens"
  total_tokens: "usage.total_tokens"
  reasoning_tokens: "usage.completion_tokens_details.reasoning_tokens"
  cached_tokens: "usage.prompt_tokens_details.cached_tokens"
```

#### 3.4.4 Sensitive Content Detection Configuration

Configure error detection in `response_<vendor>.yaml`:

```yaml
error_check:
  sensitive_check:
    input_sensitive_type_path: "input_sensitive_type"
    output_sensitive_type_path: "output_sensitive_type"
```

#### 3.4.5 Request Field Transformation Configuration

Configure request field mapping in `request_<vendor>.yaml`:

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

## Appendix: Xiaomi Adapter Experience Summary

### Xiaomi vs OpenAI Differences

| Aspect | OpenAI | Xiaomi |
|--------|--------|--------|
| thinking parameter | `thinking: true` | `thinking.type: "enabled"` |
| Response reasoning_content | N/A | `choices[].message.reasoning_content` |
| tools.strict | Not supported | Supports `tools[].function.strict` |
| Built-in tools | None | Auto-calls tool_calls |

---

## Checklist

When adding a new vendor adapter, ensure the following are completed:

- [ ] `configs/<vendor>/request_<vendor>.yaml` created
- [ ] `configs/<vendor>/response_<vendor>.yaml` created
- [ ] `cnllm/core/vendor/<vendor>.py` created
- [ ] Inherited BaseAdapter + Responder + VendorError components
- [ ] `_get_responder()` method returns Responder instance
- [ ] `model_mapping` configuration complete
- [ ] VendorError registered to VendorErrorRegistry
- [ ] Basic chat test passed
- [ ] Streaming output test passed
- [ ] `.think`, `.still`, `.tools` properties work correctly
- [ ] Sensitive content detection works
- [ ] Standard structure has no extra fields
- [ ] Documentation `docs/vendor/<vendor>.md` updated

## Phase 4: Testing & Validation

### 4.1 Basic Chat Test

```python
client = CNLLM(model="<model>", api_key="<key>")
resp = client.chat.create(messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}])
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}])
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}], thinking=True)
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}], stream=True)
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}], stream=True, thinking=True)
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)

resp = client.chat.create(messages=[{"role": "user", "content": "What's the weather in Istanbul"}], tools=[{"type": "function", "function": {"name": "get_weather", ...} }])
print(resp)
print(f"====="*20)
print(resp.raw)
print(f"====="*20)
print(resp.still)
print(f"====="*20)
print(resp.think)
print(f"====="*20)
print(resp.tools)

resp = client.chat.create(messages=[{"role": "user", "content": "What's the weather in Istanbul"}], tools=[{"type": "function", "function": {"name": "get_weather", ...} }], stream=True)
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

**Validation Points:**
- Confirm it works normally without errors
- Verify resp conforms to OpenAI standard structure, no extra fields (e.g., content should not include .think (reasoning_content) reasoning process)
- Verify .still gets clean output
- Verify .think gets clean reasoning process
- Verify .raw .think .still .tools output accumulates in real-time during streaming
- Verify .tools gets clean tool call information

### 4.2 Streaming Output Test

```python
client = CNLLM(model="<model>", api_key="<key>", stream=True)
chunks = []
resp = client.chat.create(
    messages=[{"role": "user", "content": "Explain in detail why the sky is blue"}],
    thinking=True
)
for i, chunk in enumerate(resp):
    chunks.append(chunk)
    if i < 20:
        print(f"[Chunk {i}] .think: {client.chat.think[:50] if client.chat.think else None}...")
        print(f"[Chunk {i}] .still: {client.chat.still[:50] if client.chat.still else None}...")
        print(f"[Chunk {i}] delta: {chunk.get('choices', [{}])[0].get('delta', {})}")
    elif i == 20:
        print("... (more than 20 chunks, stop printing intermediate process)")

print(f"\nTotal {len(chunks)} chunks")
print(f"====="*20)
print(f".think (complete): {client.chat.think}")
print(f"====="*20)
print(f".still (complete): {client.chat.still}")
print(f"====="*20)
print(f"resp (complete): {chunks[-1] if chunks else None}")
```

**Validation Points:**
- ✅ Each chunk has `id`, `object`, `choices`
- ✅ `delta` contains `content` or `role`
- ✅ `.think` and `.still` accumulate in real-time
- ✅ reasoning_content does not appear in resp

### 4.3 Standard Structure Validation

```python
def check_standard_format(resp):
    """Check if response conforms to OpenAI standard format"""
    required_keys = {"id", "object", "created", "model", "choices", "usage"}
    actual_keys = set(resp.keys())

    extra = actual_keys - required_keys
    if extra:
        print(f"Extra fields: {extra}")

    choice = resp.get("choices", [{}])[0]
    choice_keys = set(choice.keys())
    expected_choice_keys = {"index", "message", "finish_reason"}

    message = choice.get("message", {})
    message_keys = set(message.keys())
    expected_message_keys = {"role", "content"}

    print(f"Response structure: {'✅ Standard' if not extra else '❌ Has extras'}")

## Scripts

### validate_model_compatible.py

Model compatibility validation script, used to test whether new models can be correctly adapted by existing adapters.

**Features:**
- Test compatibility of supported models
- Test potential compatible models (e.g., M2.1 may be compatible with M2.7 series)
- Test streaming output
- Test Fallback mechanism
- Test LangChain Runnable integration

**Environment Variables:**
- `MINIMAX_API_KEY` - MiniMax API Key
- `XIAOMI_API_KEY` - Xiaomi API Key (optional)
- `CNLLM_SKIP_MODEL_VALIDATION=true` - Skip model mapping validation (for testing unlisted models)

**Usage:**
```bash
python scripts/validate_model_compatible.py
```

### test_e2e_installed.py

End-to-end test script, simulates production environment usage after user `pip install cnllm`.

**Features:**
- Does not reference project local modules, uses installed cnllm package
- Verifies the installed package works correctly
- Tests basic chat, streaming output, Fallback, etc.

**Environment Variables:**
- `MINIMAX_API_KEY` - Required

**Usage:**
```bash
python scripts/test_e2e_installed.py
```
```
