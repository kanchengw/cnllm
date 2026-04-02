import pytest
from unittest.mock import patch, MagicMock
from cnllm import CNLLM
from cnllm.core.vendor.minimax import MiniMaxAdapter
from cnllm.utils.exceptions import ContentFilteredError


class TestSensitiveContentDetection:
    """敏感内容检测"""

    def test_normal_response_no_sensitive_raises_nothing(self):
        """正常响应（无敏感内容）不抛出异常"""
        print("\n[TEST] Normal response -> no exception")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": None,
            "output_sensitive_type": None
        }

        try:
            adapter._check_sensitive(raw_resp)
            print("  PASS: No exception raised for clean response")
        except ContentFilteredError as e:
            pytest.fail(f"Should not raise ContentFilteredError: {e}")

    def test_input_sensitive_raises_content_filtered_error(self):
        """输入内容敏感时抛出 ContentFilteredError"""
        print("\n[TEST] Input sensitive detected -> ContentFilteredError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": "politics",
            "output_sensitive_type": None
        }

        with pytest.raises(ContentFilteredError) as exc_info:
            adapter._check_sensitive(raw_resp)

        print(f"  Exception message: {exc_info.value.message}")
        assert "输入内容敏感" in exc_info.value.message
        assert "politics" in exc_info.value.message
        print("  PASS: Raises ContentFilteredError for input sensitive")

    def test_output_sensitive_raises_content_filtered_error(self):
        """输出内容敏感时抛出 ContentFilteredError"""
        print("\n[TEST] Output sensitive detected -> ContentFilteredError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": None,
            "output_sensitive_type": "violence"
        }

        with pytest.raises(ContentFilteredError) as exc_info:
            adapter._check_sensitive(raw_resp)

        print(f"  Exception message: {exc_info.value.message}")
        assert "输出内容敏感" in exc_info.value.message
        assert "violence" in exc_info.value.message
        print("  PASS: Raises ContentFilteredError for output sensitive")

    def test_both_sensitive_input_checked_first(self):
        """输入和输出都敏感时，先报输入敏感错误"""
        print("\n[TEST] Both sensitive -> input error raised first")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": "politics",
            "output_sensitive_type": "violence"
        }

        with pytest.raises(ContentFilteredError) as exc_info:
            adapter._check_sensitive(raw_resp)

        print(f"  Exception message: {exc_info.value.message}")
        assert "输入内容敏感" in exc_info.value.message
        print("  PASS: Input sensitive error raised first")

    def test_null_sensitive_values_are_ignored(self):
        """null 值不触发敏感检测"""
        print("\n[TEST] Null sensitive values -> ignored")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": None,
            "output_sensitive_type": "null"
        }

        try:
            adapter._check_sensitive(raw_resp)
            print("  PASS: Null values are ignored")
        except ContentFilteredError as e:
            pytest.fail(f"Should not raise ContentFilteredError for null: {e}")

    def test_zero_sensitive_values_are_ignored(self):
        """0 值不触发敏感检测（MiniMax API 0 = 正常）"""
        print("\n[TEST] Zero sensitive values -> ignored (0 = normal for MiniMax)")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": 0,
            "output_sensitive_type": 0
        }

        try:
            adapter._check_sensitive(raw_resp)
            print("  PASS: Zero values are ignored")
        except ContentFilteredError as e:
            pytest.fail(f"Should not raise ContentFilteredError for 0: {e}")

    def test_empty_string_sensitive_values_are_ignored(self):
        """空字符串不触发敏感检测"""
        print("\n[TEST] Empty string sensitive values -> ignored")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": "",
            "output_sensitive_type": ""
        }

        try:
            adapter._check_sensitive(raw_resp)
            print("  PASS: Empty string values are ignored")
        except ContentFilteredError as e:
            pytest.fail(f"Should not raise ContentFilteredError for empty string: {e}")


class TestSensitiveCheckIntegration:
    """敏感检测集成测试（端到端）"""

    def test_create_completion_with_sensitive_input_raises_error(self):
        """create_completion 输入敏感时抛出 ContentFilteredError"""
        print("\n[TEST] create_completion with sensitive input -> ContentFilteredError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": "politics",
            "output_sensitive_type": None
        }

        with pytest.raises(ContentFilteredError) as exc_info:
            with patch.object(adapter, '_check_error') as mock_check:
                mock_check.side_effect = ContentFilteredError("test")
                adapter._check_sensitive(raw_resp)

        print(f"  Exception raised: {type(exc_info.value).__name__}")
        print("  PASS: ContentFilteredError raised for sensitive input")

    def test_create_completion_with_sensitive_output_raises_error(self):
        """create_completion 输出敏感时抛出 ContentFilteredError"""
        print("\n[TEST] create_completion with sensitive output -> ContentFilteredError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        raw_resp = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "input_sensitive_type": None,
            "output_sensitive_type": "violence"
        }

        with pytest.raises(ContentFilteredError) as exc_info:
            adapter._check_sensitive(raw_resp)

        print(f"  Exception message: {exc_info.value.message}")
        print("  PASS: ContentFilteredError raised for sensitive output")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
