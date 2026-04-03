# CNLLM Architecture and Design Documentation

## 1. Architecture Design

### 1.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/entry/client.py)                │
├─────────────────────────────────────────────────────────────┤
│  Three Entry Points:                                        │
│  - Simple: client("prompt")                                 │
│  - Standard: client.chat.create(prompt="...")              │
│  - Full: client.chat.create(messages=[...])                │
│                                                             │
│  Response Properties:                                       │
│  - client.chat.still  → Clean text                         │
│  - client.chat.think → Thinking process (reasoning_content)│
│  - client.chat.tools → Tool calls (tool_calls)             │
│  - client.chat.raw   → Raw response (vendor-specific)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseAdapter                              │
│              (cnllm/core/adapter.py)                        │
├─────────────────────────────────────────────────────────────┤
│  - Parameter validation (YAML config driven)                │
│  - Payload construction                                     │
│  - HTTP request sending                                     │
│  - Vendor model mapping                                     │
│  - Delegates to Responder for response transformation       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Responder                                │
│              (cnllm/core/responder.py)                      │
├─────────────────────────────────────────────────────────────┤
│  - Response format transformation (vendor → OpenAI)         │
│  - reasoning_content extraction & accumulation              │
│  - tool_calls extraction & accumulation                    │
│  - usage information processing                            │
│  - Sensitive content detection (input/output)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VendorError                              │
│              (cnllm/utils/vendor_error.py)                 │
├─────────────────────────────────────────────────────────────┤
│  - Vendor error parsing (code → CNLLM Error)               │
│  - Error code mapping (YAML config driven)                  │
│  - Triggers sensitive content detection                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                           │
│                     (cnllm/entry/http.py)                  │
├─────────────────────────────────────────────────────────────┤
│  - HTTP request sending                                     │
│  - Retry mechanism                                         │
│  - Error handling                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       [External API]                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Three-Layer Architecture Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **BaseAdapter** | `core/adapter.py` | Core adapter logic: param validation, payload construction, request sending |
| **Responder** | `core/responder.py` | Response transformation: vendor format → OpenAI standard |
| **VendorError** | `utils/vendor_error.py` | Error handling: vendor error code → CNLLM unified exception |

### 1.3 Layered Architecture Principles

| Layer | Component | Responsibility | Example |
|-------|-----------|---------------|---------|
| **Frontend Entry** | `CNLLM` (entry/client.py) | Unified entry, model validation, param standardization | `CNLLM(model='minimax-m2.7')` |
| **Request Preprocessing** | `BaseAdapter` (core/adapter.py) | Param validation, payload construction, vendor model mapping | `_build_payload()`, `validate_model()` |
| **HTTP Execution** | `BaseHttpClient` (entry/http.py) | Generic HTTP request/retry | `post_stream()`, `post()` |
| **Response Postprocessing** | `Responder` (core/responder.py) | Unified OpenAI format transformation | `to_openai_stream_format()` |
| **Vendor Specific Layer** | Vendor Adapters (core/vendor/) | Vendor-specific response handling | MiniMax's `reasoning_content` |

**Principles**:
- **Frontend Entry Layer** (`CNLLM`): First to receive user parameters, performs lowercase model name conversion and initial validation
- **Preprocessing Layer** (`BaseAdapter`): Param validation, filtering, payload assembly, calls HttpClient
- **HTTP Layer** (`BaseHttpClient`): Generic HTTP logic, no vendor-specific awareness
- **Postprocessing Layer** (`Responder`): Transforms vendor raw response to OpenAI standard format, universal for all vendors
- **Vendor Specific Layer**: Vendor-specific response fields (like MiniMax's `reasoning_content`) must be handled in vendor Adapter layer

***

## 2. Directory Structure

```
cnllm/
├── entry/                    # Entry Layer - Client init and call entry
│   ├── __init__.py
│   ├── client.py             # CNLLM main client class
│   └── http.py               # HTTP request client
├── core/                     # Core Layer - Adapter abstraction and vendor implementation
│   ├── __init__.py
│   ├── adapter.py            # BaseAdapter base adapter
│   ├── responder.py          # Responder response transformation framework
│   ├── framework/
│   │   ├── __init__.py
│   │   └── langchain.py      # LangChain integration
│   └── vendor/               # Vendor implementation
│       ├── __init__.py
│       ├── minimax.py        # MiniMax vendor adapter
│       └── xiaomi.py         # Xiaomi vendor adapter
└── utils/                    # Utility Layer - Common utilities
    ├── __init__.py
    ├── exceptions.py         # Exception definitions
    ├── fallback.py           # Fallback manager
    ├── stream.py             # Streaming utilities
    ├── validator.py          # Parameter validator
    └── vendor_error.py       # Vendor error handling

configs/
├── minimax/
│   ├── request_minimax.yaml  # Request config
│   └── response_minimax.yaml # Response config
└── xiaomi/
    ├── request_xiaomi.yaml   # Request config
    └── response_xiaomi.yaml  # Response config
```

***

## 3. Model Selection Flow

```
chat.create(messages, model, api_key, ...)
        │
        ▼
    model specified?
    ├── Yes → Call adapter directly (skip fallback)
    │
    └── No → Call FallbackManager
                    │
                    ▼
            Primary model available?
            ├── Yes → Primary model succeeds
            │
            └── No → Try fallback_models in order
                        │
                        ├── All fail → FallbackError
                        └── Any succeeds → That model succeeds
```

## 4. YAML Vendor Configuration Files

### 4.1 YAML Feature Integration

| Purpose | Access Point | YAML Path | YAML Filename |
|---------|--------------|-----------|------|
| OpenAI response field mapping | `responder` | `fields` | response_{vendor}.yaml |
| Get default values | `max_retries`, `retry_delay`... | `default_values` | request_{vendor}.yaml |
| Vendor request field mapping | `build_payload` | `body_mapping` (in request) | request_{vendor}.yaml |
| Required param validation | `validate_required_params` | `required_fields` | request_{vendor}.yaml |
| Param support validation | `filter_supported_params` | `optional_fields` | request_{vendor}.yaml |
| Mutual exclusion validation | `validate_one_of` | `one_of` | request_{vendor}.yaml |
| API config | `get_base_url`, `get_api_path` | `request.base_url`, `request.url` | request_{vendor}.yaml |
| Model name mapping | `model_mapping` | `model_mapping` | request_{vendor}.yaml |
| Error check | `error_check` | `error_check` | request_{vendor}.yaml |

### 4.2 request_{vendor}.yaml

```yaml
request:
  method: "POST"
  url: "/chat/completions"
  base_url: "https://api.{vendor}.com/v1"
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
  fallback_models: ""
  stream: ""
  thinking:
    path: "thinking"
    transform:
      true: {"type": "enabled"}
      false: {"type": "disabled"}
  # ... more optional parameters

model_mapping:
  minimax-m2: "MiniMax-M2"
  mimo-v2-flash: "mimo-v2-flash"
  # ... more model mappings

error_check:
  code_path: "base_resp.status_code"
  message_path: "base_resp.status_msg"
  success_code: 0
  error_codes:
    1001: { type: "timeout", message: "Request timeout", suggestion: "Check network connection" }
    # ... more error code mappings
```

### 4.3 response_{vendor}.yaml

```yaml
fields:
  id: "id"
  created: "created"
  model: "model"
  content: "choices[0].message.content"
  reasoning_content: "choices[0].message.reasoning_content"
  tool_calls: "choices[0].message.tool_calls"
  # ...

defaults:
  object: "chat.completion"
  # ...

stream_fields:
  delta:
    content: "delta.content"
    reasoning_content: "delta.reasoning_content"
    role: "delta.role"
    tool_calls: "delta.tool_calls"
  # ...

usage:
  prompt_tokens: "usage.prompt_tokens"
  completion_tokens: "usage.completion_tokens"
  total_tokens: "usage.total_tokens"
  prompt_tokens_details:
    cached_tokens: "usage.prompt_tokens_details.cached_tokens"
  ...
```

***

## 5. Exception Handling System

### 5.1 Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Vendor Error (cnllm/core/vendor/)                │
│  MiniMaxVendorError.from_response()                        │
│  Responsibility: Parse vendor raw response → code, message│
└─────────────────────────┬───────────────────────────────────┘
                          │ Registry.create_vendor_error()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Error Translator (cnllm/utils/vendor_error.py)    │
│  ErrorTranslator.translate()                               │
│  Responsibility: Lookup YAML → type → CNLLM Error          │
└─────────────────────────┬───────────────────────────────────┘
                          │ raise CNLLM Error
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: CNLLM Error (cnllm/utils/exceptions.py)          │
│  RateLimitError, ServerError, AuthenticationError...        │
│  Responsibility: Unified exception type, vendor-agnostic   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: User Code (Application Layer)                    │
│  try: ... except CNLLMError: ...                           │
│  Responsibility: User catches and handles exceptions       │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `VendorError` | `utils/vendor_error.py` | Data class: code, message, vendor, raw_response |
| `VendorErrorRegistry` | `utils/vendor_error.py` | Register vendor error classes, create VendorError instances |
| `ErrorTranslator` | `utils/vendor_error.py` | Lookup YAML to translate to CNLLM Error |
| `MiniMaxVendorError` | `core/vendor/minimax.py` | MiniMax-specific response parsing logic |

***

## 6. Version History

### v0.4.0 (2026-04-03)

- ✨ **mimo Adapter** - Xiaomi mimo model adapter, supports "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"
- ✨ **Architecture Refactoring** - BaseAdapter + Responder + VendorError three-layer architecture separation, clear responsibilities
- ✨ **.think Property** - `client.chat.think` retrieves reasoning_content, not included in resp
- ✨ **.tools Property** - `client.chat.tools` retrieves tool_calls, supports streaming accumulation
- ✨ **Streaming Accumulation** - `.think`, `.still`, `.tools` support real-time accumulation during streaming response

### v0.3.3 (2026-04-02) ✨

- ✨ **Unified Parameters** - Client init parameters unified with call entry parameters, call entry flexibly overrides
- ✨ **Architecture Optimization** - Core logic abstraction, BaseAdapter and Responder handle common logic
- ✨ **Extensibility** - Adding new provider only requires configuring corresponding YAML file, automatically implements request and response field mapping, error code mapping, no need to modify other upper-level components
- ✨ **YAML Function Integration** - Related field mapping, model support validation, required param validation, param support validation, vendor error code mapping logic
- ✨ **MiniMax Support Optimization** - Supports all MiniMax native interface parameters, such as `top_p`, `tools`, `thinking`, etc.

### v0.3.1 (2026-03-29) ✨

- ✨ **Deep LangChain Integration**
  - Runnable adapter as core feature, one function integrates with LangChain chain
  - Runnable streaming output, batch calls, async calls support
- ✨ **chat.create() Streaming Output** - `stream=True` parameter support
- ✨ **Fallback Mechanism** - Automatically switch to backup model when primary fails
- ✨ **Response Entry** - `client.chat.still` easily gets clean chat response, `client.chat.raw` gets full response
- 🔧 **Adapter Refactoring** - Model adapters (Chinese LLMs like MiniMax) + framework adapters (LangChain, etc.) dual-layer architecture
