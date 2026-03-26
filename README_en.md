# CNLLM - Chinese LLM Adapter

[English](README_en.md) | [中文](README.md)

A unified adapter library for Chinese large language models (LLMs). It converts API outputs from various Chinese LLMs (such as MiniMax, Doubao, Kimi, etc.) into the unified OpenAI format, enabling zero-cost integration with mainstream AI frameworks like LangChain and AutoGen.

## Features

- **OpenAI Compatible** - All outputs fully align with OpenAI API standard format
- **LangChain Native** - Directly use LangChain message types and utility functions
- **Unified Interface** - One codebase, seamless switching between different LLMs
- **Streaming Output** - Support for streaming responses (planned)
- **Retry Mechanism** - Built-in timeout and automatic retry
- **Detailed Logging** - Clear error messages and debugging support

## Supported Models

### Verified
- [x] MiniMax-M2.7
- [x] MiniMax-M2.5

### In Development
- [ ] Doubao (ByteDance)
- [ ] Kimi (Moonshot)
- [ ] StepFun (StepFun)
- [ ] ERNIE (Baidu)
- [ ] Qwen (Alibaba)
- [ ] ChatGLM (Zhipu AI)

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

### Basic Usage

```python
from cnllm import CNLLM, MINIMAX_API_KEY

# Initialize client
client = CNLLM(
    model="minimax-m2.7",  # or "minimax-m2.5"
    api_key=MINIMAX_API_KEY
)

# Send message
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Introduce yourself in one sentence"}
    ]
)

# Get response
print(resp["choices"][0]["message"]["content"])
```

### Environment Variables

Create a `.env` file:

```env
MINIMAX_API_KEY=your_api_key_here
```

### Use with LangChain

```python
from langchain_core.messages import HumanMessage, AIMessage
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

# CNLLM output can be directly used by LangChain
resp = client.chat.create(
    messages=[{"role": "user", "content": "Hello"}]
)

# Convert to LangChain message
ai_msg = AIMessage(content=resp["choices"][0]["message"]["content"])
print(ai_msg.content)
```

## API Reference

### CNLLM Client

```python
from cnllm import CNLLM

client = CNLLM(
    model="minimax-m2.7",      # Model name
    api_key="your_api_key",    # API key
    timeout=30,                # Request timeout (seconds)
    max_retries=3,            # Max retry attempts
    retry_delay=1.0            # Retry delay (seconds)
)
```

### chat.create()

```python
resp = client.chat.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.7,           # Temperature parameter
    stream=False,              # Enable streaming
    model="minimax-m2.7"      # Override default model
)
```

### Response Format (OpenAI Standard)

```python
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "minimax-m2.7",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 15,
        "total_tokens": 35
    }
}
```

## Project Structure

```
cnllm/
├── adapters/              # Adapter layer
│   └── minimax/          # MiniMax adapter
│       └── chat.py
├── core/                  # Core components
│   ├── base.py          # HTTP client
│   ├── config.py        # Configuration management
│   ├── exceptions.py    # Exception definitions
│   └── types.py         # Type definitions
├── utils/                # Utilities
│   └── cleaner.py       # Output cleaner
├── client.py             # Unified client entry point
└── __init__.py
```

## Error Handling

```python
from cnllm import CNLLM
from cnllm.core.exceptions import ModelAPIError, ParseError

try:
    client = CNLLM(model="minimax-m2.7", api_key="invalid_key")
    resp = client.chat.create(messages=[{"role": "user", "content": "Hello"}])
except ModelAPIError as e:
    print(f"API Error: {e}")
except ParseError as e:
    print(f"Parse Error: {e}")
except ValueError as e:
    print(f"Parameter Error: {e}")
```

## Development

### Run Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
python test_CNLLM.py
```

### Add New Model Adapter

1. Create a new adapter directory under `adapters/`
2. Implement `create_completion()` method
3. Implement `_to_openai_format()` conversion method
4. Register adapter in `client.py`

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

- GitHub Issues: [https://github.com/kanchengw/cnllm/issues](https://github.com/kanchengw/cnllm/issues)
- Email: wangkancheng1122@163.com
