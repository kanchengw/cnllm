# CNLLM - Chinese LLM Adapter

[English](README_en.md) | 中文

统一的中文大模型适配库，将各种国产大模型（如 MiniMax、字节豆包、Kimi 等）的 API 输出转换为统一的 OpenAI 格式，零成本接入 LangChain、AutoGen 等主流 AI 框架。

## 特性

- **OpenAI 兼容** - 所有输出完全对齐 OpenAI API 标准格式
- **LangChain 原生支持** - 可直接使用 LangChain 的消息类型和工具函数
- **统一接口** - 一套代码，无缝切换不同大模型
- **流式输出** - 支持流式响应（规划中）
- **重试机制** - 内置超时和自动重试
- **详细日志** - 清晰的错误信息和调试支持

## 支持的模型

### 已验证
- [x] MiniMax-M2.7
- [x] MiniMax-M2.5

### 开发中
- [ ] 字节豆包 (Doubao)
- [ ] Kimi (Moonshot)
- [ ] 阶跃星辰 (StepFun)
- [ ] 百度文心一言 (ERNIE)
- [ ] 阿里通义千问 (Qwen)
- [ ] 智谱 GLM (ChatGLM)

## 安装

```bash
pip install cnllm
```

或从源码安装：

```bash
git clone https://github.com/yourusername/cnllm.git
cd cnllm
pip install -e .
```

## 快速开始

### 基础使用

```python
from cnllm import CNLLM, MINIMAX_API_KEY

# 初始化客户端
client = CNLLM(
    model="minimax-m2.7",  # 或 "minimax-m2.5"
    api_key=MINIMAX_API_KEY
)

# 发送消息
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "用一句话介绍自己"}
    ]
)

# 获取回复
print(resp["choices"][0]["message"]["content"])
```

### 环境变量配置

创建 `.env` 文件：

```env
MINIMAX_API_KEY=your_api_key_here
```

### 在 LangChain 中使用

```python
from langchain_core.messages import HumanMessage, AIMessage
from cnllm import CNLLM, MINIMAX_API_KEY

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

# CNLLM 的输出可以直接被 LangChain 使用
resp = client.chat.create(
    messages=[{"role": "user", "content": "你好"}]
)

# 转换为 LangChain 消息
ai_msg = AIMessage(content=resp["choices"][0]["message"]["content"])
print(ai_msg.content)
```

## API 参考

### CNLLM 客户端

```python
from cnllm import CNLLM

client = CNLLM(
    model="minimax-m2.7",      # 模型名称
    api_key="your_api_key",    # API 密钥
    timeout=30,                # 请求超时（秒）
    max_retries=3,             # 最大重试次数
    retry_delay=1.0             # 重试延迟（秒）
)
```

### chat.create()

```python
resp = client.chat.create(
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7,           # 温度参数
    stream=False,              # 是否流式输出
    model="minimax-m2.7"      # 可覆盖默认模型
)
```

### 返回格式（OpenAI 标准）

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
                "content": "你好！有什么可以帮助你的吗？"
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

## 项目结构

```
cnllm/
├── adapters/              # 适配器层
│   └── minimax/          # MiniMax 适配器
│       └── chat.py
├── core/                  # 核心组件
│   ├── base.py          # HTTP 客户端
│   ├── config.py        # 配置管理
│   ├── exceptions.py    # 异常定义
│   └── types.py         # 类型定义
├── utils/                # 工具类
│   └── cleaner.py       # 输出清理
├── client.py             # 统一客户端入口
└── __init__.py
```

## 错误处理

```python
from cnllm import CNLLM
from cnllm.core.exceptions import ModelAPIError, ParseError

try:
    client = CNLLM(model="minimax-m2.7", api_key="invalid_key")
    resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])
except ModelAPIError as e:
    print(f"API 错误: {e}")
except ParseError as e:
    print(f"解析错误: {e}")
except ValueError as e:
    print(f"参数错误: {e}")
```

## 开发

### 运行测试

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行所有测试
python test_CNLLM.py
```

### 添加新的模型适配器

1. 在 `adapters/` 下创建新的适配器目录
2. 实现 `create_completion()` 方法
3. 实现 `_to_openai_format()` 转换方法
4. 在 `client.py` 中注册适配器

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub Issues: [https://github.com/yourusername/cnllm/issues](https://github.com/yourusername/cnllm/issues)
- Email: your.email@example.com
