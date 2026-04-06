# CNLLM Architecture and Design Documentation

## 1. Architecture Design

### 1.1 Overall Architecture

```mermaid
flowchart TD
    subgraph User Layer
        A[cnllm/entry/client.py<br/>CNLLM Client]
    end

    A --> B[cnllm/core/adapter.py<br/>BaseAdapter]
    B --> C[cnllm/entry/http.py<br/>BaseHttpClient]
    C --> D((External API))

    D --> E[cnllm/core/responder.py<br/>Responder]
    E --> A

    D --> F[cnllm/utils/vendor_error.py<br/>VendorError]
    F --> A

    B -.-> G[/yaml/configs<br/>YAML Config/]
    F -.-> G
    E -.-> G

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style E fill:#e8f5e9
    style F fill:#ffebee
    style G fill:#f5f5f5,stroke:#ccc
```

### 1.2 General Base Class Architecture

| Base Class Component | File | Responsibility | Example |
| --- | --- | --- | --- |
| **Frontend Entry** | `CNLLM` (entry/client.py) | Client initialization, call entry | `CNLLM(model='minimax-m2.7')` |
| **Request Preprocessing** | `BaseAdapter` (core/adapter.py) | Request field mapping, Payload construction | `_build_payload()`, `validate_model()` |
| **HTTP Execution** | `BaseHttpClient` (entry/http.py) | Generic HTTP request, retry mechanism | `post_stream()`, `post()` |
| **Response Postprocessing** | `Responder` (core/responder.py) | Response field mapping, OpenAI standard format construction | `to_openai_stream_format()` |

### 1.2 Vendor Layer Architecture

| Vendor Layer Component | File | Responsibility | Example |
| --- | --- | --- | --- |
| **Vendor Adapter** | `core/vendor/{vendor}.py` | Vendor-specific request handling, Payload construction | `MiniMaxAdapter.create_completion()` |
| **Vendor Response Converter** | `core/vendor/{vendor}.py` | Vendor-specific response conversion logic | `MiniMaxResponder.to_openai_format()` |
| **Vendor Error Parser** | `core/vendor/{vendor}.py` | Vendor-specific error parsing | `MiniMaxVendorError.parse()` |
| **Request Config** | `configs/{vendor}/` | Vendor request field mapping, error code mapping, param validation | `request_{vendor}.yaml` |
| **Response Config** | `configs/{vendor}/` | Vendor response field mapping, stream processing config | `response_{vendor}.yaml` |

### 1.3 Utility Class Architecture

| Utility Class | File | Responsibility | Example |
| --- | --- | --- | --- |
| **Exception System** | `utils/exceptions.py` | CNLLM exception base class, unified exception system | `raise CNLLMError(msg)` |
| **Vendor Error Translator** | `utils/vendor_error.py` | Vendor error translator, translate to CNLLM exception | `translator.to_cnllm_error()` |
| **Fallback Manager** | `utils/fallback.py` | Fallback manager, handle model unavailability fallback logic | `execute_with_fallback()` |
| **Streaming Utility** | `utils/stream.py` | Streaming utility, handle streaming response | `process_stream_chunk()` |
| **Parameter Validator** | `utils/validator.py` | Parameter validator, validate model, field, param range | `validate_model()`, `validate_required()` |

***

## 2. Directory Structure

```
cnllm/
├── entry/                    # Entry Layer - Client initialization and call entry
│   ├── __init__.py
│   ├── client.py             # CNLLM main client class
│   └── http.py               # HTTP request client
├── core/                     # Core Layer - Adapter abstraction and vendor implementation
│   ├── __init__.py
│   ├── adapter.py            # BaseAdapter base adapter
│   ├── responder.py          # Responder response transformation framework
│   ├── framework/
│   │   ├── __init__.py
│   │   └── langchain.py      # LangChain Runnable integration
│   └── vendor/               # Vendor implementation
│       ├── __init__.py
│       ├── minimax.py        # MiniMax vendor adapter
│       └── xiaomi.py         # Xiaomi vendor adapter
└── utils/                    # Utility Layer - Common utilities
    ├── __init__.py
    ├── exceptions.py         # Exception definitions
    ├── fallback.py           # Fallback manager
    ├── stream.py             # Streaming utility
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

## 3. Exception Handling System Architecture

```mermaid
flowchart LR
    subgraph HTTP层["HTTP Layer (cnllm/entry/http.py)"]
        H["Network Layer\nTimeout / 401 / 429 / 500 / 400"]
    end

    subgraph Responder层["Responder Layer (core/responder.py)"]
        R["Business Layer\nContent Filter / Balance / Model Not Supported"]
    end

    subgraph Fallback层["Fallback Layer (utils/fallback.py)"]
        F["FallbackManager\nMulti-model Fallback"]
    end

    subgraph 用户层["User Code"]
        U["try: ... except CNLLMError: ..."]
    end

    HTTP层 -->|"Exception Pass-through"| 用户层
    Responder层 -->|"Exception Pass-through"| 用户层
    Fallback层 -->|"FallbackError"| 用户层

    style HTTP层 fill:#ffebee
    style Responder层 fill:#fff3e0
    style Fallback层 fill:#e8f5e9
    style 用户层 fill:#e3f2fd
```

***

## 4. FallbackManager Flow Design

Only the client initialization entry accepts the `fallback_models` parameter. It is recommended to configure this option for program or application runtime stability.
When the primary model at the client entry is unavailable, it will try models in `fallback_models` in order.
Code example:

```python
client = CNLLM(
    model="minimax-m2.7", api_key="minimax_key",
    fallback_models={"mimo-v2-flash": "xiaomi-key", "minimax-m2.5": None}  # None means use the API_key configured for the primary model
    )
resp = client.chat.create(prompt="What is 2+2?")  # If model is configured again at the call entry, it will override all models configured at the client entry
print(resp)
```

```mermaid
flowchart TD
    A[chat.create Call Entry] --> B{model specified?}
    B -->|Yes| C[Call adapter]
    C -->|Success| J[Entry model success]
    C -->|Failure| K[ModelNotSupportedError]
    B -->|No| D[Call FallbackManager]
    D --> E{Primary model available?}
    E -->|Yes| F[Primary model success]
    E -->|No| G{Try fallback_models in order}
    G -->|All fail| H[FallbackError]
    G -->|Any succeeds| I[That model succeeds]
```

***