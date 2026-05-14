"""
LangChain 兼容性 E2E 测试。
测试 CNLLM 的响应能否被 LangChain 正确消费：
1. LangChainRunnable.invoke / stream / batch
2. ChatPromptTemplate + LangChainRunnable 链
3. 流式输出与 LCEL 兼容性
"""
import os, sys, asyncio, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM
from cnllm.core.framework.langchain import LangChainRunnable

pytest.importorskip("langchain_core")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"


def test_runnable_invoke():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from langchain_core.messages import AIMessage
    client = CNLLM(model=MODEL, api_key=API_KEY)
    runnable = LangChainRunnable(client)
    resp = runnable.invoke("用一句话介绍自己")
    assert isinstance(resp, AIMessage), f"应为 AIMessage，实际 {type(resp)}"
    assert len(resp.content) > 0
    client.close()


def test_runnable_stream():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    client = CNLLM(model=MODEL, api_key=API_KEY)
    runnable = LangChainRunnable(client)
    chunks = list(runnable.stream("数到5"))
    assert len(chunks) > 0
    client.close()


def test_runnable_batch():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from langchain_core.messages import AIMessage
    client = CNLLM(model=MODEL, api_key=API_KEY)
    runnable = LangChainRunnable(client)
    resps = runnable.batch(["1+1=?", "2+2=?"])
    assert len(resps) == 2
    assert all(isinstance(r, AIMessage) for r in resps)
    client.close()


def test_chat_prompt_template():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from langchain_core.prompts import ChatPromptTemplate
    client = CNLLM(model=MODEL, api_key=API_KEY)
    runnable = LangChainRunnable(client)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是翻译助手，将用户输入翻译成英文"),
        ("human", "{input}"),
    ])
    chain = prompt | runnable
    resp = chain.invoke({"input": "你好世界"})
    assert "hello" in resp.content.lower() or "world" in resp.content.lower(), resp.content
    client.close()


def test_lcel_stream():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from langchain_core.prompts import ChatPromptTemplate
    client = CNLLM(model=MODEL, api_key=API_KEY)
    runnable = LangChainRunnable(client)
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
    chain = prompt | runnable
    chunks = list(chain.stream({"input": "数到3"}))
    assert len(chunks) > 0
    client.close()


def test_runnable_ainvoke():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from langchain_core.messages import AIMessage
    async def run():
        client = CNLLM(model=MODEL, api_key=API_KEY)
        runnable = LangChainRunnable(client)
        resp = await runnable.ainvoke("1+1=?", stream=False)
        return resp
    resp = asyncio.run(run())
    assert "2" in resp.content
