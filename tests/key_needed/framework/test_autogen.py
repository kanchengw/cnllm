"""
AutoGen 兼容性 E2E 测试。
CNLLM 作为 AutoGen 的 LLM 后端（通过 OpenAI 兼容接口注册）。
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

pytest.importorskip("autogen_agentchat")
pytest.importorskip("autogen_ext.models.openai")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_autogen_llm_backend():
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://api.deepseek.com/v1",
        model_capabilities={"vision": False, "function_calling": True, "json_output": True, "structured_output": True},
    )
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant. Reply in Chinese.",
    )
    result = await agent.on_messages(
        [TextMessage(content="1+1=?", source="user")],
        cancellation_token=None,
    )
    assert result is not None
    assert result.chat_message.content is not None
