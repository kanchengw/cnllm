from cnllm import MiniMaxChat, Message

llm = MiniMaxChat(
    api_key="你的 MINIMAX_API_KEY",
    model="MiniMax-M1"
)

messages = [
    Message(role="user", content="提取实验标题和总结")
]

resp = llm.invoke(messages)
print(resp.content)