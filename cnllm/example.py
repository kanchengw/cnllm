from cnllm import CNLLM

client = CNLLM(
    api_key="your MINIMAX_API_KEY",
    model="minimax-m2.7"
)

resp = client("用一句话介绍自己")
print(resp["choices"][0]["message"]["content"])
