# CNLLM Architecture and Design Document

***

## Design Principles

1. **OpenAI Standard Output** - Output format aligned with OpenAI API standard
2. **Developer Friendly** - Unified client entry and call entry
3. **Compatibility Friendly** - Unsupported model parameters warn but don't block

***

## 1. Architecture Design

### 1.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CNLLM Client                         │
│                     (cnllm/core/client.py)                  │
├─────────────────────────────────────────────────────────────┤
│  Three Entry Points:                                       │
│  - Simple: client("prompt")                              │
│  - Standard: client.chat.create(prompt="...")               │
│  - Full: client.chat.create(messages=[...])             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Adapter Layer                      │
│              (cnllm/adapters/{vendor}/chat.py)             │
├─────────────────────────────────────────────────────────────┤
│  Vendor Adapter Layer (Protocol Conversion + Param Validation)│
│  - MiniMaxAdapter                                          │
│  - More model adapters in development                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Framework Adapter Layer                    │
│              (cnllm/adapters/framework/*.py)                │
├─────────────────────────────────────────────────────────────┤
│  Framework Integration (Unified Output Format)              │
│  - LangChainRunnable                                       │
│  - More framework integrations in development               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseHttpClient                           │
│                     (cnllm/core/base.py)                    │
├─────────────────────────────────────────────────────────────┤
│  HTTP Base Layer (Request / Retry / Error Handling)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Cleaner Layer                     │
│                     (cnllm/utils/cleaner.py)                 │
├─────────────────────────────────────────────────────────────┤
│  Unified Output Format (Aligned with OpenAI API Standard)  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Module Responsibilities

| Module           | File                                         | Responsibility                         |
| --------------- | ------------------------------------------- | -------------------------------------- |
| **Client Entry**   | `cnllm/core/client.py`                     | Three entry points, param pass-through, routing |
| **Model Adapter**  | `cnllm/adapters/{vendor}/chat.py`           | Vendor protocol conversion + param validation |
| **Framework Adapter** | `cnllm/adapters/framework/*.py`           | Framework integration (LangChain, etc.) |
| **Model Mapping**  | `cnllm/core/models.py`                     | SUPPORTED\_MODELS + ADAPTER\_MAP |
| **HTTP Base**  | `cnllm/core/base.py`                       | Network requests, retry                          |
| **Exception Def** | `cnllm/utils/exceptions.py`                 | Unified exception system                        |
| **Param Registry** | `cnllm/core/params.py`                    | Param classification config                      |
| **Fallback**   | `cnllm/utils/fallback.py`                   | Fallback mechanism                           |
| **Output Cleaner** | `cnllm/utils/cleaner.py`                  | Unified model output format                     |
| **Model Validation** | `cnllm/utils/validate_model_compatible.py` | Model compatibility validation               |

***

## 2. Three Entry Points

| Entry       | Calling Method                                                             |
| ---------- | ------------------------------------------------------------------------ |
| **Simple** | `client("prompt")`                                                        |
| **Standard** | `client.chat.create(prompt="prompt")`                                |
| **Full**    | `client.chat.create(messages=[{"role": "user", "content": "..."}])` |

**Call Chain Description**:

- `client("prompt")` calls `chat.create(prompt=prompt)` via `__call__`
- `client.chat.create()` is the actual dispatcher, routing to adapter or fallback manager based on parameters
- When `model` parameter is specified, fallback is skipped and adapter is called directly
- When `model` parameter is not specified, enters Fallback Manager (which determines if FB config exists)

***

## 3. Parameter System

### 3.1 Parameter Classification (params.py)

| Classification | Definition   | Handling                 |
| ------------- | ------ | ---------------------- |
| **required**  | Required params | Python signature validation + type check |
| **supported** | Optional params | ✅ Passed to API          |
| **Other**      | Unknown params | ⚠️ Warn + ignore, continue running    |

Unrecognized parameters are warned and ignored, simplifying logic while improving compatibility.

### 3.2 params.py Registry Structure

```python
PROVIDER_PARAMS = {
    "minimax": {
        "init": {
            "required": ["api_key", "model"],
            "supported": ["base_url", "timeout", "max_retries", "retry_delay"],
        },
        "create": {
            "required": [],
            "supported": ["messages", "temperature", "max_tokens", "stream", "tools", "tool_choice", "group_id"],
        }
    }
}
```

***

## 4. Fallback Flow

```
User call (simple/standard/full entry)
        │
        ▼
┌─────────────────────────────────────┐
│ chat.create()                        │
│ 1. Check required fields (messages/prompt) │
│ 2. model parameter specified?          │
│    - Yes → Call adapter directly     │
│    - No → Continue                   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ No fallback_models configured?         │
│    - Yes → Call adapter directly     │
│    - No → Enter Fallback flow        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ FallbackManager.execute_with_fallback │
│ 1. Primary model → _get_adapter_for_model() │
│    - Validate against SUPPORTED_MODELS │
│    - Find corresponding Adapter        │
│ 2. Primary fails → warn → try fallback │
│ 3. FB model → _get_adapter_for_model()│
│    - Validate against SUPPORTED_MODELS │
│    - Find corresponding Adapter        │
│ 4. FB fails → warn → try next FB     │
│ 5. All models fail → FallbackError    │
└─────────────────────────────────────┘
```

### 4.1 Model and Adapter Matching

```
SUPPORTED_MODELS = {
    "minimax-m2.7": "minimax",
    "minimax-m2.5": "minimax",
}

ADAPTER_MAP = {
    "minimax": MiniMaxAdapter,
}

Model Validation Flow:
1. Check if model name is in SUPPORTED_MODELS
2. Get adapter_name via mapping
3. Get Adapter class via adapter_name in ADAPTER_MAP
4. Create Adapter instance
```

***

## 5. Exception System

### 5.1 Exception Types

```
CNLLMError (Base)
├── AuthenticationError      # Auth failure (401)
├── RateLimitError           # Rate limited (429)
├── TimeoutError            # Timeout (408)
├── NetworkError            # Network error
├── ServerError             # Server error (5xx)
├── InvalidRequestError     # Invalid request (400)
├── ParseError              # Parse error
├── ModelNotSupportedError  # Model not supported
├── MissingParameterError   # Missing parameter
├── ContentFilteredError    # Content filtered (403)
├── TokenLimitError        # Token limit exceeded (431)
├── ModelAPIError          # Model API call failed
└── FallbackError          # All models failed
```

### 5.2 Exception Attributes

```python
class CNLLMError(Exception):
    message: str           # Error message
    error_code: ErrorCode  # Error code enum
    status_code: int       # HTTP status code
    provider: str          # Provider identifier
    details: dict          # Detailed diagnostic info
    suggestion: str        # User suggestion
```

***

## 6. Directory Structure

```
cnllm/
├── __init__.py              # Package entry, exports CNLLM and exception classes
├── adapters/
│   ├── minimax/
│   │   └── chat.py         # MiniMax adapter
│   └── framework/
│       └── langchain.py     # LangChain Runnable adapter
├── core/
│   ├── client.py            # Client (three entry points)
│   ├── models.py            # Model mapping (SUPPORTED_MODELS, ADAPTER_MAP)
│   ├── params.py            # Parameter registry
│   └── base.py              # HTTP base layer
└── utils/
    ├── exceptions.py        # Exception definitions
    ├── fallback.py          # Fallback mechanism
    ├── validate_model_compatible.py  # Model compatibility validation
    └── cleaner.py           # Output cleaning
```
```

***

## 7. Version Plan

### v0.3.0 ✅ Completed (2026-03-28)

- [x] Structured error system
- [x] Three entry points
- [x] Stream output (`stream=True`)
- [x] Simplified parameter validation (required/supported)
- [x] LangChain Runnable adapter
- [x] Fallback mechanism
- [x] Model compatibility validation tool

### v0.4.0 (Planned)

- [ ] Model adapter development (e.g., Doubao, Kimi, etc.)
- [ ] Framework adapter validation and deep integration (LlamaIndex, Pydantic, LiteLLM, Instructor)
