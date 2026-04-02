"""
CNLLM 示例 - 展示核心功能数据流
"""
import os
from dotenv import load_dotenv

load_dotenv()

from cnllm import CNLLM
from cnllm.core.framework import LangChainRunnable

API_KEY = os.getenv("MINIMAX_API_KEY")
if not API_KEY:
    print("请设置 MINIMAX_API_KEY 环境变量")
    exit(1)


def example_1_simple_call():
    print("\n" + "=" * 50)
    print("示例1: 极简调用 - client('prompt')")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)

    response = client("用一句话介绍自己")
    print(f"模型输出: {response['choices'][0]['message']['content']}")


def example_2_chat_create():
    print("\n" + "=" * 50)
    print("示例2: 标准API接口 - client.chat.create(messages=[...])")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是大语言模型？"}
    ]

    response = client.chat.create(messages=messages, temperature=0.7)
    print(f"模型输出: {response['choices'][0]['message']['content']}")


def example_3_with_group_id():
    print("\n" + "=" * 50)
    print("示例3: 厂商特有参数 - group_id")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)

    response = client.chat.create(
        messages=[{"role": "user", "content": "你好"}],
        group_id=os.getenv("MINIMAX_GROUP_ID", "")
    )
    print(f"模型输出: {response['choices'][0]['message']['content']}")


def example_4_streaming():
    print("\n" + "=" * 50)
    print("示例4: 同步流式输出")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    runnable = LangChainRunnable(client)

    print("输出: ", end="", flush=True)
    for chunk in runnable.stream("从1数到5"):
        print(chunk, end="", flush=True)
    print()


def example_5_async_streaming():
    print("\n" + "=" * 50)
    print("示例5: 异步流式输出")
    print("=" * 50)

    import asyncio

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    runnable = LangChainRunnable(client)

    async def run():
        print("异步流式输出 (每个chunk显示):")
        chunk_count = 0
        async for chunk in runnable.astream("讲一个笑话"):
            chunk_count += 1
            print(f"  chunk{chunk_count}: '{chunk}'")
        print(f"  总计: {chunk_count} 个chunk")

    asyncio.run(run())


def example_6_batch():
    print("\n" + "=" * 50)
    print("示例6: 批量调用")
    print("=" * 50)

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    runnable = LangChainRunnable(client)

    inputs = ["你好", "再见", "你是谁"]
    results = runnable.batch(inputs)

    for i, result in enumerate(results):
        print(f"回复{i+1}: {result.content}")


def example_7_langchain_integration():
    print("\n" + "=" * 50)
    print("示例7: LangChain 集成")
    print("=" * 50)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    client = CNLLM(model="minimax-m2.7", api_key=API_KEY)
    runnable = LangChainRunnable(client)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的助手，只用一句话回答"),
        ("user", "{question}")
    ])

    chain = prompt | runnable | StrOutputParser()

    result = chain.invoke({"question": "为什么天空是蓝色的？"})
    print(f"最终输出: {result}")


if __name__ == "__main__":
    example_1_simple_call()
    example_2_chat_create()
    example_3_with_group_id()
    example_4_streaming()
    example_5_async_streaming()
    example_6_batch()
    example_7_langchain_integration()

    print("\n" + "=" * 50)
    print("全部示例完成！")
    print("=" * 50)
