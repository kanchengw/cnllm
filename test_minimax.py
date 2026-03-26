from cnllm import MiniMaxChat, Message
from cnllm.core.config import MINIMAX_API_KEY

# 初始化模型
llm = MiniMaxChat(
    api_key=MINIMAX_API_KEY,
    model="MiniMax-M2.7"
)

# 构造消息
messages = [
    Message(
        role="user",
        content="请用一句话介绍你自己。输出尽量干净，不要多余内容。"
    )
]

# 调用
print("正在调用 MiniMax API...")
resp = llm.invoke(messages)

# 输出结果
print("\n=== 模型返回内容 ===")
print(resp.content)