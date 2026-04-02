# CNLLM Architecture and Design Document

## 1. Architecture Design

### 1.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/core/client.py)                 │
├─────────────────────────────────────────────────────────────┤
│  Three Entry Points:                                        │
│  - Simple: client("prompt")                                │
│  - Standard: client.chat.create(prompt="...")               │
│  - Full: client.chat.create(messages=[...])                │
│                                                             │
│  Response Accessors:                                        │
│  - client.chat.still  → Plain text                         │
│  - client.chat.raw   → Raw response (with platform-specific fields) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Adapter Layer                      │
│              (cnllm/adapters/{vendor}/chat.py)             │
├─────────────────────────────────────────────────────────────┤
│  - Vendor protocol conversion                               │
│  - Parameter validation                                    │
│  - Store raw response in adapter._raw_response             │
│  - Return OpenAI format response                           │
│                                                             │
│  Example: MiniMaxAdapter                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                          │
│                     (cnllm/core/base.py)                   │
├─────────────────────────────────────────────────────────────┤
│  - HTTP request sending                                    │
│  - Retry mechanism                                         │
│  - Error handling                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       [External API]                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Cleaner Layer                     │
│                     (cnllm/utils/cleaner.py)                │
├─────────────────────────────────────────────────────────────┤
│  - Clean Markdown markers                                  │
│  - Extract OpenAI standard fields, convert response to standard format │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Layered Architecture Principles

**Common abstraction layer** contains multiple components, each with its own responsibility:

| Layer | Component | Responsibility | Example |
|------|-----------|----------------|---------|
| **Frontend Entry** | `CNLLM` (entry/client.py) | Unified entry, model validation, parameter standardization | `CNLLM(model='minimax-m2.7')` |
| **Request Preprocessing** | `BaseAdapter` (core/adapter.py) | Parameter validation, Payload building, vendor model mapping | `_build_payload()`, `validate_model()` |
| **HTTP Execution** | `BaseHttpClient` (entry/http.py) | Generic HTTP request/retry | `post_stream()`, `post()` |
| **Response Postprocessing** | `Responder` (core/responder.py) | Unified OpenAI format conversion | `to_openai_stream_format()` |
| **Vendor-Specific Layer** | Vendor Adapters (core/vendor/) | Vendor-specific response handling | MiniMax's `reasoning_content` |

**Principles**:
- **Frontend Entry Layer** (`CNLLM`): First to receive user parameters, unified lowercase conversion and initial validation for model names
- **Preprocessing Layer** (`BaseAdapter`): Parameter validation, filtering, Payload assembly, calling HttpClient
- **HTTP Layer** (`BaseHttpClient`): Generic HTTP logic, vendor-agnostic
- **Postprocessing Layer** (`Responder`): Convert vendor raw response to OpenAI standard format, all vendors use this
- **Vendor-Specific Layer**: Vendor-specific response fields (such as MiniMax's `reasoning_content`) must be handled in vendor Adapter layer

***

## 2. Three Entry Points

| Entry | Calling Method |
| ---------- | ------------------------------------------------------------------- |
| **Simple** | `client("prompt")` |
| **Standard** | `client.chat.create(prompt="prompt")` |
| **Full** | `client.chat.create(messages=[{"role": "user", "content": "..."}])` |

**Call Chain Description**:

- `client("prompt")` calls `chat.create(prompt=prompt)` via `__call__` method

***

## 3. Fallback Flow

### 3.1 Call Decision Flow

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

## 5. Exception System

### 5.1 Exception Types

```
CNLLMError (Base)
├── AuthenticationError      # Auth failure (401)
├── RateLimitError           # Rate limit (429)
├── TimeoutError            # Timeout (408)
├── NetworkError            # Network error
├── ServerError             # Server error (5xx)
├── InvalidRequestError     # Request error (400)
├── ParseError              # Parse error
├── ModelNotSupportedError  # Model not supported
├── MissingParameterError   # Missing parameter
├── ContentFilteredError    # Content filtered (403)
├── TokenLimitError        # Token limit (431)
├── ModelAPIError          # Model API call failed
└── FallbackError          # All models failed
```

### 5.2 Exception Attributes

```python
class CNLLMError(Exception):
    message: str           # Error message
    error_code: ErrorCode  # Error code enum
    status_code: int       # HTTP status code
    provider: str          # Vendor identifier
    details: dict          # Detailed diagnostic info
    suggestion: str        # User suggestion
```

***

## 6. Directory Structure

```
cnllm/
├── entry/                    # Entry Layer - Client init and call entry
│   ├── __init__.py
│   ├── client.py             # CNLLM main client class
│   └── http.py               # HTTP request client
├── core/                     # Core Layer - Adapter abstraction and vendor implementation
│   ├── __init__.py
│   ├── adapter.py            # BaseAdapter base adapter
│   ├── responder.py          # Responder response format conversion framework
│   ├── framework/
│   │   ├── __init__.py
│   │   └── langchain.py      # LangChain integration
│   └── vendor/               # Vendor implementation
│       ├── __init__.py
│       └── minimax.py        # MiniMax vendor adapter
└── utils/                    # Utils Layer - Common utilities
    ├── __init__.py
    ├── exceptions.py         # Exception definitions
    ├── fallback.py           # Fallback manager
    ├── stream.py             # Streaming processing utility
    ├── validator.py          # Parameter validator
    └── vendor_error.py       # Vendor error handling

configs/
└── minimax/
    ├── request_minimax.yaml  # Request config
    └── response_minimax.yaml # Response config
```

***

## 7. YAML Vendor Configuration Files

### 7.1 request_minimax.yaml

```yaml
request:
  method: "POST"
  url: "/text/chatcompletion_v2"
  base_url: "https://api.minimaxi.com/v1"
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
  # ... more optional params

model_mapping:
  minimax-m2: "MiniMax-M2"
  # ... more model mappings

error_check:
  code_path: "base_resp.status_code"
  message_path: "base_resp.status_msg"
  success_code: 0
  error_codes:
    1001: { type: "timeout", message: "Request timeout", suggestion: "Please check network connection" }
    1002: { type: "rate_limit", message: "RPM rate limit triggered", suggestion: "Please reduce request frequency" }
    # ... more error code mappings
```

### 7.2 response_minimax.yaml

```yaml
fields:
  id: "id"
  created: "created"
  model: "model"
  content: "choices[0].message.content"
  # ...

defaults:
  object: "chat.completion"
  # ...

stream_fields:
  # ...
```

### 7.3 YAML Function Integration

| Purpose | Access Point | YAML Path |
|---------|--------------|-----------|
| Get default values | `defaults`, `timeout`, `max_retries`, `retry_delay` | `default_values` |
| Vendor request field mapping | `build_payload` | `body_mapping` (in request) |
| OpenAI response field mapping | `responder` | `fields` |
| Required param validation | `validate_required_params` | `required_fields` |
| Param support validation | `filter_supported_params` | `optional_fields` |
| Mutually exclusive param validation | `validate_one_of` | `one_of` |
| API config | `get_base_url`, `get_api_path` | `request.base_url`, `request.url` |
| Model name mapping | `model_mapping` | `model_mapping` |

***

## 8. Exception Handling System

### 8.1 Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Vendor Error (cnllm/core/vendor/)                │
│  MiniMaxVendorError.from_response()                        │
│  Responsibility: Parse vendor raw response → code, message │
└─────────────────────────┬─────────────────────────────────┘
                          │ Registry.create_vendor_error()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Error Translator (cnllm/utils/vendor_error.py) │
│  ErrorTranslator.translate()                               │
│  Responsibility: Query YAML → type → CNLLM Error          │
└─────────────────────────┬─────────────────────────────────┘
                          │ raise CNLLM Error
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: CNLLM Error (cnllm/utils/exceptions.py)        │
│  RateLimitError, ServerError, AuthenticationError...       │
│  Responsibility: Unified exception type, vendor-agnostic  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: User Code (User Application Layer)             │
│  try: ... except CNLLMError: ...                          │
│  Responsibility: User catches and handles exceptions       │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `VendorError` | `utils/vendor_error.py` | Data class: code, message, vendor, raw_response |
| `VendorErrorRegistry` | `utils/vendor_error.py` | Register vendor error classes, create VendorError instances |
| `ErrorTranslator` | `utils/vendor_error.py` | Query YAML to translate to CNLLM Error |
| `MiniMaxVendorError` | `core/vendor/minimax.py` | MiniMax-specific response parsing logic |

***

## 10. Version Plan

### v0.3.1 ✅ Completed (2026-03-29)

- [x] Structured error system
- [x] Three entry points
- [x] Stream streaming output (`stream=True`)
- [x] Simplified parameter validation (required/supported two categories)
- [x] LangChain Runnable adapter
- [x] Fallback mechanism
- [x] Model compatibility validation tool
- [x] `client.chat.still` / `client.chat.raw` response accessors

### v0.4.0 (Planned)

- [ ] Model adapter development (such as Doubao, Kimi, etc.)
- [ ] Framework adapter validation and deep integration (LlamaIndex, Pydantic, LiteLLM, Instructor)
