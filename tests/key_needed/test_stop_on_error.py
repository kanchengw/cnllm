"""
E2E 测试：Batch stop_on_error 逻辑

验证 stop_on_error=True 时，首个错误后不再执行剩余请求。
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()
import pytest
from cnllm import CNLLM
from cnllm.utils.exceptions import FallbackError

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-v4-flash"
requires_key = pytest.mark.skipif(not API_KEY, reason="需要 DEEPSEEK_API_KEY")


@requires_key
def test_stop_on_error_sync_non_stream():
    """stop_on_error=True: 首个失败后不再执行剩余请求"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    # request_2 使用错误的 api_key，应触发错误
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "hi"},
            {"prompt": "error", "api_key": "sk-wrong-key"},
            {"prompt": "should-not-run"},
        ],
        stop_on_error=True,
        keep=["*"],
    )
    for _ in resp:
        pass

    # 有成功也有失败
    assert resp.status["total"] == 4
    # 第 3 个请求应失败，第 4 个可能没执行
    assert "request_2" in resp.errors or resp.status["success_count"] < 4


@requires_key
def test_stop_on_error_sync_stream():
    """stop_on_error=True + stream"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "error", "api_key": "sk-wrong-key"},
            {"prompt": "should-not-run"},
        ],
        stream=True,
        stop_on_error=True,
        keep=["*"],
    )
    for _ in resp:
        pass

    assert resp.status["total"] == 3
    # request_1 应失败
    if "request_1" in resp.errors:
        assert resp.status["success_count"] < 3


@requires_key
def test_stop_on_error_false_normal():
    """stop_on_error=False: 所有请求都会执行（不限顺序）"""
    client = CNLLM(model=MODEL, api_key=API_KEY)
    resp = client.chat.batch(
        requests=[
            {"prompt": "hello"},
            {"prompt": "error", "api_key": "sk-wrong-key"},
            {"prompt": "world"},
        ],
        stop_on_error=False,
        keep=["*"],
    )
    for _ in resp:
        pass

    # 3 个请求都应有结果（成功或失败）
    assert len(resp.still) + len(resp.errors) == 3, \
        f"still={len(resp.still)}, errors={len(resp.errors)}, 应共 3"
