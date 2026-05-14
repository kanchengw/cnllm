"""
DeepEval 兼容性 E2E 测试。
CNLLM 生成的文本交给 DeepEval 评估。
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

pytest.importorskip("deepeval")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"


def test_deepeval_test_case():
    """CNLLM 输出 → DeepEval 评估测试用例"""
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from deepeval.test_case import LLMTestCase
    from cnllm import CNLLM

    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.create(
        messages=[{"role": "user", "content": "1+1=?"}],
        stream=False,
    )
    actual = resp["choices"][0]["message"]["content"]

    test_case = LLMTestCase(
        input="1+1=?",
        actual_output=actual,
        expected_output="2",
    )
    assert test_case.actual_output is not None
    assert len(test_case.actual_output) > 0
    client.close()


def test_deepeval_stream():
    """CNLLM 流式累积 → DeepEval 评估"""
    if not API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from deepeval.test_case import LLMTestCase
    from cnllm import CNLLM

    client = CNLLM(model=MODEL, api_key=API_KEY)
    acc = ""
    for chunk in client.chat.create(prompt="数到3", stream=True):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta:
            acc += delta

    test_case = LLMTestCase(
        input="数到3",
        actual_output=acc,
    )
    assert len(test_case.actual_output) > 0
    client.close()
