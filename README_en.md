# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

![PyPI Version](https://img.shields.io/pypi/v/cnllm)
![Python Versions](https://img.shields.io/pypi/pyversions/cnllm)
![License](https://img.shields.io/github/license/kanchengw/cnllm)
![GitHub Stars](https://img.shields.io/github/stars/kanchengw/cnllm?style=social)
![GitHub Forks](https://img.shields.io/github/forks/kanchengw/cnllm?style=social)

---

A unified adapter library for Chinese large language models (LLMs). It converts API outputs from various Chinese LLMs (such as MiniMax, Doubao, Kimi, etc.) into the unified OpenAI format, enabling zero-cost integration with mainstream AI frameworks like LangChain.

## Changelog

### v0.2.0 (2026-03-27)
- ✨ **New `__call__` method**: Support ultra simple calling style `client("prompt")`
- ✨ **New prompt parameter**: Support direct string input, no need to manually construct messages
- ✨ **Model override mechanism**: Support specifying other models that share the same API at call time
- ✨ **LangChain compatibility tests**: 13 function compatibility verifications
- 📝 **README refactored**: Streamlined structure, added API parameters and calling style examples

## Features

- **OpenAI Compatible** - All outputs fully align with OpenAI API standard format, can directly integrate with LangChain, LlamaIndex and other major frameworks
- **Unified Interface** - One codebase, seamless switching between different LLMs
- **Model Override** - Support specifying other models that share the same API at call time
- **Simple API** - Multiple calling styles, as simple as one line of code
- **Streaming Output** - Support for streaming responses (planned)
- **Retry Mechanism** - Built-in timeout and automatic retry

## Supported Models

- **Verified**: MiniMax-M2.7, MiniMax-M2.5
- **More models and providers in development**

## Installation

```bash
pip install cnllm
```

Or install from source:

```bash
git clone https://github.com/kanchengw/cnllm.git
cd cnllm
pip install -e .
```

## Quick Start

### Three Calling Styles

**1. Ultra Simple `client("prompt")`**

```python
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)
resp = client("Introduce yourself in one sentence")
print(resp["choices"][0]["message"]["content"])
```

**2. Standard `client.chat.create(prompt="prompt")`**

```python
resp = client.chat.create(prompt="Introduce yourself in one sentence")
print(resp["choices"][0]["message"]["content"])
```

**3. Full `client.chat.create(messages=[...])`**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Introduce yourself in one sentence"}
    ]
)
print(resp["choices"][0]["message"]["content"])
```

### Model Override

Support overriding the default model at call time (for multiple models accessible via the same API):

```python
resp = client.chat.create(
    prompt="Introduce yourself",
    model="minimax-m2.5"  # Override the initialized model
)
```

## API Parameters

### CNLLM Client Initialization

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | str | ✅ | - | Model name: `minimax-m2.7`, `minimax-m2.5` |
| `api_key` | str | ✅ | - | API key |
| `timeout` | int | - | 30 | Request timeout (seconds) |
| `max_retries` | int | - | 3 | Max retry attempts |
| `retry_delay` | float | - | 1.0 | Retry delay (seconds) |

### chat.create() Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `messages` | list[dict] | ⚠️ | - | OpenAI format message list (choose one with prompt) |
| `prompt` | str | ⚠️ | - | Shortcut, auto-converted to messages (choose one with messages) |
| `temperature` | float | - | 0.1 | Randomness, 0-2 |
| `stream` | bool | - | False | Streaming response (planned) |
| `model` | str | - | None | Override default model |

### __call__ Shortcut Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | str | ✅ | - | Prompt text |
| `temperature` | float | - | 0.1 | Randomness |
| `model` | str | - | None | Override default model |

## Response Format

All API responses follow OpenAI standard format:

```python
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "minimax-m2.7",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "I am MiniMax-M2.7..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

## LangChain Compatible Functions

| Function/Class | Status | Description |
|----------------|--------|-------------|
| `HumanMessage` | ✅ | Human message type |
| `AIMessage` | ✅ | AI message type |
| `SystemMessage` | ✅ | System message type |
| `BaseMessage` | ✅ | Message base class |
| `ChatPromptTemplate` | ✅ | Chat prompt template |
| `StrOutputParser` | ✅ | String output parser |
| `message_to_dict` | ✅ | Message to dict |
| `messages_to_dict` | ✅ | Batch messages to dict |
| `AIMessageChunk` | ✅ | AI message chunk |
| `ChatMessage` | ✅ | Chat message |
| `FunctionMessage` | ✅ | Function message |
| `ToolMessage` | ✅ | Tool message |

## Use with LangChain

CNLLM returns standard OpenAI format, directly usable with LangChain functions:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import message_to_dict, messages_to_dict
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

# Call CNLLM to get response
resp = client.chat.create(messages=[{"role": "user", "content": "Hello"}])
print(resp["choices"][0]["message"]["content"])

# CNLLM output follows OpenAI standard, directly usable with LangChain
ai_msg = AIMessage(content=resp["choices"][0]["message"]["content"])
print(f"Role: {ai_msg.type}")  # "ai"
print(f"Content: {ai_msg.content}")

# Convert to LangChain dict format
msg_dict = message_to_dict(ai_msg)
print(msg_dict)
# {'type': 'ai', 'data': {'content': '...', ...}}

# Batch convert
msgs = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!")
]
msgs_dict = messages_to_dict(msgs)
print(msgs_dict)
```

## Other Compatible Frameworks

CNLLM output is compatible with all Python libraries using OpenAI format:

- **LangChain** - Message types, chain calling
- **LlamaIndex** - Indexing and querying
- **AutoGen** - Multi-agent collaboration (planned)
- **CrewAI** - Multi-agent workflows
- **Dify** - Platform integration

## License

MIT License - See [LICENSE](LICENSE) file

## Contact

- GitHub Issues: [https://github.com/kanchengw/cnllm/issues](https://github.com/kanchengw/cnllm/issues)
