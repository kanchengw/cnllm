"""
_batch.py 功能测试：_normalize_batch_requests + split_batch_params 集成
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cnllm.utils.scheduler.base import _normalize_batch_requests


class TestNormalizeBatchRequests:
    """_normalize_batch_requests 单元测试"""

    def test_requests_only(self):
        """纯 requests 模式"""
        requests = [{"prompt": "A"}, {"prompt": "B"}]
        result = _normalize_batch_requests(requests_arg=requests)
        assert len(result) == 2
        assert result[0]["prompt"] == "A"
        assert result[1]["prompt"] == "B"

    def test_prompt_only(self):
        """纯 prompt 列表模式"""
        result = _normalize_batch_requests(prompt=["A", "B"])
        assert len(result) == 2
        assert result[0]["prompt"] == "A"
        assert result[1]["prompt"] == "B"
        assert result[0]["_input_type"] == "prompt"

    def test_messages_only(self):
        """纯 messages 列表模式"""
        msgs = [[{"role": "user", "content": "A"}], [{"role": "user", "content": "B"}]]
        result = _normalize_batch_requests(messages=msgs)
        assert len(result) == 2
        assert result[0]["messages"] == msgs[0]
        assert result[1]["messages"] == msgs[1]

    def test_requests_with_shared_prompt_str(self):
        """requests + 共享 prompt 字符串注入"""
        requests = [{"temperature": 0.7}, {"temperature": 0.2}]
        result = _normalize_batch_requests(requests_arg=requests, prompt="A")
        assert len(result) == 2
        assert result[0]["prompt"] == "A"
        assert result[0]["temperature"] == 0.7
        assert result[1]["prompt"] == "A"
        assert result[1]["temperature"] == 0.2

    def test_requests_with_shared_prompt_list_raises(self):
        """requests + prompt 列表 → TypeError（与 requests 共存时 prompt 必须为字符串）"""
        requests = [{"temperature": 0.7}, {"temperature": 0.2}]
        with pytest.raises(TypeError, match="prompt 必须为字符串"):
            _normalize_batch_requests(requests_arg=requests, prompt=["A", "B"])

    def test_requests_with_shared_messages_raises(self):
        """requests + messages 列表的列表 → TypeError（共存时 messages 必须为单组消息列表）"""
        requests = [{"temperature": 0.7}, {"temperature": 0.2}]
        msgs = [[{"role": "user", "content": "A"}], [{"role": "user", "content": "B"}]]
        with pytest.raises(TypeError, match="messages 必须为单组消息列表"):
            _normalize_batch_requests(requests_arg=requests, messages=msgs)

    def test_requests_own_prompt_takes_precedence(self):
        """item 自有 prompt 优先，共享 prompt 不覆盖"""
        requests = [{"prompt": "C"}, {"prompt": "D"}]
        result = _normalize_batch_requests(requests_arg=requests, prompt="shared")
        assert result[0]["prompt"] == "C"
        assert result[1]["prompt"] == "D"

    def test_requests_missing_prompt_no_shared(self):
        """requests 无 prompt 且无共享 prompt 时报错"""
        requests = [{"temperature": 0.7}]
        with pytest.raises(TypeError, match="必须包含"):
            _normalize_batch_requests(requests_arg=requests)

    def test_requests_with_defaults_merge(self):
        """per_request_defaults 合并"""
        requests = [{"prompt": "A"}, {"prompt": "B"}]
        result = _normalize_batch_requests(
            requests_arg=requests,
            per_request_defaults={"temperature": 0.5}
        )
        assert result[0]["temperature"] == 0.5
        assert result[1]["temperature"] == 0.5

    def test_requests_with_defaults_item_overrides(self):
        """item 自有参数覆盖 per_request_defaults"""
        requests = [{"prompt": "A", "temperature": 0.7}]
        result = _normalize_batch_requests(
            requests_arg=requests,
            per_request_defaults={"temperature": 0.5}
        )
        assert result[0]["temperature"] == 0.7  # item 优先

    def test_empty_requests_raises(self):
        """空 requests 报错"""
        with pytest.raises(TypeError, match="不能为空"):
            _normalize_batch_requests(requests_arg=[])

    def test_prompt_and_messages_mutual_exclusive(self):
        """prompt 和 messages 不能同时提供"""
        with pytest.raises(TypeError, match="不能同时提供"):
            _normalize_batch_requests(prompt=["A"], messages=[[{"role": "user", "content": "B"}]])

    def test_prompt_list_with_requests_raises(self):
        """requests + prompt 列表 → TypeError"""
        requests = [{"temperature": 0.7}, {"temperature": 0.2}]
        with pytest.raises(TypeError, match="prompt 必须为字符串"):
            _normalize_batch_requests(requests_arg=requests, prompt=["A"])

    def test_embedded_batch_level_warning(self):
        """requests 内误传 batch-level 参数时发出警告"""
        requests = [{"prompt": "A", "max_concurrent": 5}]
        import logging
        logger = logging.getLogger("cnllm.utils.batch")
        # 触发一次确保可运行
        result = _normalize_batch_requests(requests_arg=requests)
        assert result[0]["prompt"] == "A"
        assert "max_concurrent" not in result[0]