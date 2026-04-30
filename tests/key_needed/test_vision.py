"""
CNLLM Vision E2E Test - 多模态图片理解 + 纯文本拒绝

测试目标：
1. 多模态模型接收 base64 图片 → 返回有意义的图片描述
2. 多模态模型流式处理图片
3. 纯文本模型传入图片 → 抛出 InvalidRequestError
4. .still / .raw 属性访问
5. 批量调用中纯文本模型拒图
"""
import os
import sys
import time
import pytest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from cnllm import CNLLM
from cnllm.utils.exceptions import InvalidRequestError

# ==============================
# 厂商配置（按需修改）
# ==============================
VISION_MODEL = "glm-4.6v-flash"
TEXT_MODEL = "glm-4.6"
API_KEY_ENV = "GLM_API_KEY"

API_KEY = os.getenv(API_KEY_ENV)

requires_api_key = pytest.mark.skipif(
    not API_KEY,
    reason=f"需要 {API_KEY_ENV}"
)

# 32x32 彩虹渐变 PNG（有效 PNG）
TEST_IMAGE_BASE64 = (
    "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"
    "j/rRPt6P7+MH8ynYjLin6ysAAAAASUVORK5CYII="
)
_IMG = lambda: "data:image/png;base64," + TEST_IMAGE_BASE64


@pytest.fixture(autouse=True)
def _wait_after_vision_test():
    """每个测试后等待，避免触发 API 限流"""
    yield
    time.sleep(3)


class TestVisionNonStream:
    """多模态非流式测试"""

    @requires_api_key
    def test_vision_model_accepts_image(self):
        """多模态模型接收 base64 图片 → 返回图片描述"""
        client = CNLLM(model=VISION_MODEL, api_key=API_KEY)
        resp = client.chat.create(messages=[{"role": "user", "content": [
            {"type": "text", "text": "这张图片展示的是什么内容？请简要描述"},
            {"type": "image_url", "image_url": {"url": _IMG()}}
        ]}])
        still = client.chat.still
        assert still, "still 不应为空"
        print(f"\n[PASS] still: {still[:200]}...")

    @requires_api_key
    def test_vision_model_returns_raw(self):
        """验证 .raw 保留完整原生响应"""
        client = CNLLM(model=VISION_MODEL, api_key=API_KEY)
        resp = client.chat.create(messages=[{"role": "user", "content": [
            {"type": "text", "text": "用一句话描述这张图片"},
            {"type": "image_url", "image_url": {"url": _IMG()}}
        ]}])
        raw = client.chat.raw
        assert raw is not None
        assert "choices" in raw
        print(f"\n[PASS] .raw keys: {list(raw.keys())[:6]}")


class TestVisionStream:
    """多模态流式测试"""

    @requires_api_key
    def test_vision_stream(self):
        """多模态模型流式处理图片 → chunks 正常累积"""
        client = CNLLM(model=VISION_MODEL, api_key=API_KEY)
        acc = ""
        for chunk in client.chat.create(messages=[{"role": "user", "content": [
            {"type": "text", "text": "用一句话描述这张图片"},
            {"type": "image_url", "image_url": {"url": _IMG()}}
        ]}], stream=True):
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                acc += choices[0]["delta"]["content"]
        assert acc, "流式不应为空"
        print(f"\n[PASS] stream: {len(acc)} chars")


class TestTextRejectImage:
    """纯文本模型拒绝图片输入"""

    @requires_api_key
    def test_text_model_rejects_image(self):
        """纯文本模型传入图片 → 抛出 InvalidRequestError"""
        client = CNLLM(model=TEXT_MODEL, api_key=API_KEY)
        with pytest.raises(InvalidRequestError) as e:
            client.chat.create(messages=[{"role": "user", "content": [
                {"type": "text", "text": "有什么？"},
                {"type": "image_url", "image_url": {"url": _IMG()}}
            ]}])
        assert TEXT_MODEL in str(e.value)
        assert "不支持图片输入" in str(e.value)
        print(f"\n[PASS] reject: {str(e.value)[:80]}...")

    @requires_api_key
    def test_text_model_works_with_text(self):
        """纯文本模型传入纯文本 → 正常工作"""
        client = CNLLM(model=TEXT_MODEL, api_key=API_KEY)
        resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])
        assert client.chat.still
        print(f"\n[PASS] text ok")

    @requires_api_key
    def test_text_model_rejects_image_in_batch(self):
        """批量调用中纯文本模型传入图片 → 报错"""
        client = CNLLM(model=TEXT_MODEL, api_key=API_KEY)
        with pytest.raises((InvalidRequestError, Exception)):
            client.chat.batch(requests=[{"messages": [{"role": "user", "content": [
                {"type": "text", "text": "什么？"},
                {"type": "image_url", "image_url": {"url": _IMG()}}
            ]}]}])
        print(f"\n[PASS] batch reject")


class TestVisionWithText:
    """多模态模型传入纯文本 → 正常工作"""

    @requires_api_key
    def test_vision_model_with_text_only(self):
        """多模态模型传纯文本 → 退化为文本模型"""
        client = CNLLM(model=VISION_MODEL, api_key=API_KEY)
        resp = client.chat.create(messages=[{"role": "user", "content": "你好"}])
        assert client.chat.still
        print(f"\n[PASS] vision+text ok")
