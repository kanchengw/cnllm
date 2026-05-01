# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNLLM is a unified adapter layer that translates Chinese LLM vendor APIs (MiniMax, DeepSeek, KIMI, Doubao, GLM, Xiaomi) into OpenAI-compatible request/response formats, enabling seamless integration with LangChain, LlamaIndex, and other OpenAI-compatible frameworks.

## Running Tests

```bash
# Unit tests only (no API keys needed)
pytest tests/test_*.py

# All tests including integration tests (require API keys in env vars)
pytest tests/

# Single test file
pytest tests/test_adapter_config.py -v
```

API-key-dependent tests live in `tests/key_needed/` and are gated by the presence of environment variables (e.g., `MINIMAX_API_KEY`, `XIAOMI_API_KEY`).

## Vision / Multimodal Support

Models with `vision: true` in `model_mapping.chat` support image input via OpenAI-standard content array format:

```python
{"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

Validation is done at the model level — passing images to a text-only model raises `InvalidRequestError` before any API call. See `BaseAdapter._check_image_support()` in `adapter.py`.

Current vision-capable vendors: GLM, Kimi, Doubao, Xiaomi.

## Architecture

```
CNLLM (client) → ChatNamespace / EmbeddingsNamespace → create/batch
  → BaseAdapter._build_payload()  [YAML-driven field mapping]
  → BaseHttpClient (httpx)        [HTTP execution]
  → Responder.to_openai_format()  [vendor response → OpenAI format]
  → Accumulator                   [field accumulation, stream handling]
```

### Three-component vendor pattern

Each vendor in `cnllm/core/vendor/{vendor}.py` implements three classes:

1. **`{Vendor}Adapter(BaseAdapter)`** — builds request payload (`_build_payload`), performs format conversion (`_to_openai_format`, `_do_to_openai_stream_format`), and registers with `_register()`
2. **`{Vendor}Responder(Responder)`** — maps vendor response fields to OpenAI standard fields via `configs/{vendor}/response_{vendor}.yaml`; usually just sets `CONFIG_DIR`
3. **`{Vendor}VendorError(VendorError)`** — parses vendor-specific error responses via `from_response()` and registers with `VendorErrorRegistry.register()`

The vendor module is also where you place any subclass overrides for logic that can't be expressed in YAML (e.g., MiniMax's stream chunk dedup, Xiaomi's `thinking` transform).

### YAML-driven request/response mapping

**Request config** (`configs/{vendor}/request_{vendor}.yaml`) drives:
- `required_fields` — mandatory parameters and validation
- `optional_fields` — optional parameters, including field name `map` (rename), `transform` (value conversion), `skip` (exclude from body, e.g. for headers)
- `model_mapping` — short model alias → vendor model name
- `error_check` — vendor error code → CNLLM exception type mapping

**Response config** (`configs/{vendor}/response_{vendor}.yaml`) drives:
- `fields` — vendor response path → OpenAI field path (e.g. `"content": "choices[0].message.content"`)
- `stream_fields` — same for streaming chunks (`content_path`, `tool_calls_path`, `reasoning_content_path`)
- `defaults` — fallback values when vendor omits fields
- `error_check` — sensitive content detection paths

The parameter processing order is: `validate_required_params` → `filter_supported_params` → `validate_one_of` → `get_default_value` → `validate_base_url` → `get_header_mappings` → `_build_payload` → `get_vendor_model`.

### Field accumulation

Streaming responses accumulate into `adapter._cnllm_extra`:
- `_thinking` — raw reasoning/thinking content
- `_still` — cleaned final response content
- `_tools` — accumulated tool_calls

Accessible via `client.chat.think`, `client.chat.still`, `client.chat.tools`, `client.chat.raw`.

### FallbackManager

`FallbackManager` is only invoked when `chat.create()` is called **without** a `model` argument (or with `model=""`). If the primary model fails, it iterates through `fallback_models` in order. If a model is passed directly to `chat.create()`, no fallback occurs.

### Sync/async relationship

`CNLLM` (sync) holds an internal `AsyncCNLLM` engine and delegates async operations to it. The `LangChainRunnable` integration uses the async engine directly.

## Adding a New Vendor

1. Create `configs/{vendor}/request_{vendor}.yaml` and `configs/{vendor}/response_{vendor}.yaml`
2. Create `cnllm/core/vendor/{vendor}.py` with the three-component pattern; call `{Vendor}Adapter._register()` at the bottom
3. Add model alias → vendor name mappings to the YAML `model_mapping.chat` section
4. Write tests: unit tests in `tests/test_*.py` (no API key), integration tests in `tests/key_needed/` (with key assignment at the top: `MODEL = "..."; API_KEY = os.getenv("...")`)

Full walkthrough: see `docs/CONTRIBUTOR.md`.