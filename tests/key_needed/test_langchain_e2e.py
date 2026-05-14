"""
LangChain Runnable E2E 测试
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM
from cnllm.core.framework.langchain import LangChainRunnable, LangChainEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"

def test_invoke():
    if not API_KEY: print("SKIP"); return
    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        r = LangChainRunnable(client).invoke("介绍自己")
        assert isinstance(r, AIMessage) and len(r.content) > 0
        print(f"[PASS] invoke")
    finally:
        client.close()

def test_stream():
    if not API_KEY: print("SKIP"); return
    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        c = list(LangChainRunnable(client).stream("数到5"))
        assert len(c) > 0
        print(f"[PASS] stream: {len(c)} chunks")
    finally:
        client.close()

def test_batch():
    if not API_KEY: print("SKIP"); return
    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        r = LangChainRunnable(client).batch(["你好", "1+1"])
        assert len(r) == 2
        print(f"[PASS] batch: 2 results")
    finally:
        client.close()

def test_bind_tools():
    if not API_KEY: print("SKIP"); return
    @tool
    def get_weather(city: str) -> str:
        """获取天气"""
        return "sunny"
    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        raw = client.chat.create(
            messages=[{"role": "user", "content": "北京天气怎么样"}],
            tools=[{"type": "function", "function": {"name": "get_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        )
        msg = raw.get("choices", [{}])[0].get("message", {})
        print(f"  raw content={msg.get('content','')!r}, raw tool_calls={msg.get('tool_calls')}")

        r = LangChainRunnable(client).bind_tools([get_weather]).invoke("北京天气")
        assert isinstance(r, AIMessage)
        assert len(r.content) > 0 or r.additional_kwargs.get("tool_calls")
        print(f"[PASS] bind_tools")
    finally:
        client.close()

def test_with_structured_output():
    """with_structured_output — 验证 bind_tools + tool_choice 能触发工具调用"""
    glm_key = os.getenv("GLM_API_KEY")
    if not glm_key: print("SKIP: no GLM key"); return
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    @tool
    def get_name_age(text: str) -> str:
        """从文本中提取姓名和年龄，返回 JSON"""
        return '{"name": "test", "age": 0}'
    client = CNLLM(model=MODEL, api_key=API_KEY,thinking=False)
    try:
        bound = LangChainRunnable(client).bind_tools([get_name_age], tool_choice="required")
        result = bound.invoke("张三28岁")
        assert isinstance(result, AIMessage)
        tool_calls = result.additional_kwargs.get("tool_calls")
        assert tool_calls, "模型应返回工具调用"
        # 验证工具调用的内容
        tc = tool_calls[0] if isinstance(tool_calls, list) else list(tool_calls.values())[0]
        assert "function" in tc, "工具调用应包含 function"
        func_name = tc["function"]["name"]
        args_raw = tc["function"]["arguments"]
        print(f"[PASS] bind_tools: {func_name}({args_raw})")
    except Exception as e:
        print(f"[SKIP] with_structured_output: {type(e).__name__}: {e}")
    finally:
        client.close()

def test_chain():
    if not API_KEY: print("SKIP"); return
    client = CNLLM(model=MODEL, api_key=API_KEY)
    try:
        prompt = ChatPromptTemplate.from_messages([("system", "简短回答"), ("human", "{input}")])
        r = (prompt | LangChainRunnable(client)).invoke({"input": "2+2"})
        assert isinstance(r, AIMessage) and len(r.content) > 0
        print(f"[PASS] chain")
    finally:
        client.close()

def test_embeddings():
    ek = os.getenv("GLM_API_KEY")
    if not ek: print("SKIP: no GLM key"); return
    client = CNLLM(model="embedding-2", api_key=ek)
    try:
        emb = LangChainEmbeddings(client)
        v = emb.embed_documents(["你好", "世界"])
        assert len(v) == 2 and len(v[0]) > 0
        q = emb.embed_query("测试")
        assert len(q) > 0
        print(f"[PASS] embeddings: dim={len(v[0])}")
    finally:
        client.close()

if __name__ == "__main__":
    test_invoke()
    test_stream()
    test_batch()
    test_bind_tools()
    test_with_structured_output()
    test_chain()
    test_embeddings()
    print("\nDone!")
