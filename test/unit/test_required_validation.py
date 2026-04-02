import pytest
from unittest.mock import patch, MagicMock
from cnllm import CNLLM
from cnllm.core.vendor.minimax import MiniMaxAdapter
from cnllm.utils.exceptions import MissingParameterError, ModelNotSupportedError


class TestApiKeyValidation:
    """api_key 必填验证"""

    def test_create_completion_without_api_key_raises_error(self):
        """create_completion 不传 api_key 时应报错"""
        print("\n[TEST] create_completion without api_key -> MissingParameterError")

        adapter = MiniMaxAdapter(api_key=None, model='minimax-m2.7')

        with pytest.raises(MissingParameterError) as exc_info:
            adapter.create_completion(prompt='hello')

        print(f"  Exception message: {exc_info.value.message}")
        assert "api_key" in exc_info.value.message
        print("  PASS: Correctly raises MissingParameterError")

    def test_create_completion_with_empty_api_key_raises_error(self):
        """create_completion 传空 api_key 时应报错"""
        print("\n[TEST] create_completion with empty api_key -> MissingParameterError")

        adapter = MiniMaxAdapter(api_key='', model='minimax-m2.7')

        with pytest.raises(MissingParameterError) as exc_info:
            adapter.create_completion(prompt='hello')

        print(f"  Exception message: {exc_info.value.message}")
        print("  PASS: Correctly raises MissingParameterError")

    def test_create_completion_with_valid_api_key_works(self):
        """create_completion 传有效 api_key 时正常工作"""
        print("\n[TEST] create_completion with valid api_key -> works normally")

        adapter = MiniMaxAdapter(api_key='test-key', model='minimax-m2.7')

        with patch.object(adapter, '_check_error'):
            with patch.object(adapter, '_to_openai_format', return_value={'choices': [{'message': {'content': 'ok'}}]}):
                with patch('cnllm.entry.http.BaseHttpClient') as mock_http:
                    mock_client = MagicMock()
                    mock_client.post.return_value = {}
                    mock_http.return_value = mock_client

                    result = adapter.create_completion(prompt='hello')

                    print(f"  Result: {result}")
                    assert result['choices'][0]['message']['content'] == 'ok'
                    print("  PASS: Returns correct result")


class TestModelValidation:
    """model 必填验证"""

    def test_create_completion_without_model_uses_adapter_model(self):
        """create_completion 不传 model 时使用 adapter 的 model"""
        print("\n[TEST] create_completion without model -> uses adapter.model, no validation")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with patch.object(adapter._validator, 'validate_model') as mock_validate:
            with patch.object(adapter, '_check_error'):
                with patch.object(adapter, '_to_openai_format', return_value={'choices': [{'message': {'content': 'ok'}}]}):
                    with patch('cnllm.entry.http.BaseHttpClient') as mock_http:
                        mock_client = MagicMock()
                        mock_client.post.return_value = {}
                        mock_http.return_value = mock_client

                        result = adapter.create_completion(prompt='hello')

                        print(f"  validate_model called: {mock_validate.called}")
                        assert mock_validate.called == False, "validate_model should NOT be called when model is None"
                        print("  PASS: No validation when model is None")

    def test_create_completion_with_model_overrides_adapter_model(self):
        """create_completion 传 model 时覆盖 adapter 的 model"""
        print("\n[TEST] create_completion with model='minimax-m2.5' -> overrides adapter.model, validates")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with patch.object(adapter._validator, 'validate_model') as mock_validate:
            with patch.object(adapter, '_check_error'):
                with patch.object(adapter, '_to_openai_format', return_value={'choices': [{'message': {'content': 'ok'}}]}):
                    with patch('cnllm.entry.http.BaseHttpClient') as mock_http:
                        mock_client = MagicMock()
                        mock_client.post.return_value = {}
                        mock_http.return_value = mock_client

                        result = adapter.create_completion(prompt='hello', model='minimax-m2.5')

                        print(f"  validate_model called: {mock_validate.called}, args: {mock_validate.call_args}")
                        assert mock_validate.called == True
                        mock_validate.assert_called_with('minimax-m2.5')
                        print("  PASS: validate_model called with correct model")

    def test_create_completion_with_invalid_model_raises_error(self):
        """create_completion 传无效 model 时报错"""
        print("\n[TEST] create_completion with invalid model -> ModelNotSupportedError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with pytest.raises(ModelNotSupportedError) as exc_info:
            adapter.create_completion(prompt='hello', model='invalid-model')

        print(f"  Exception type: {type(exc_info.value).__name__}")
        assert "invalid-model" in str(exc_info.value.message)
        print("  PASS: Correctly raises ModelNotSupportedError")

    def test_create_completion_with_empty_model_raises_error(self):
        """create_completion 传空 model 时报错"""
        print("\n[TEST] create_completion with empty model -> MissingParameterError")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with pytest.raises(MissingParameterError) as exc_info:
            adapter.create_completion(prompt='hello', model='')

        print(f"  Exception message: {exc_info.value.message}")
        print("  PASS: Correctly raises MissingParameterError")


class TestMessagesValidation:
    """messages/prompt 必填验证（通过 YAML one_of 配置）"""

    def test_create_completion_without_messages_nor_prompt_raises_error(self):
        """create_completion 既不传 messages 也不传 prompt 时报错"""
        print("\n[TEST] create_completion without messages AND prompt -> MissingParameterError (one_of)")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with pytest.raises(MissingParameterError) as exc_info:
            adapter.create_completion()

        print(f"  Exception message: {exc_info.value.message}")
        assert "one of" in str(exc_info.value.message)
        print("  PASS: Correctly raises MissingParameterError via one_of validation")

    def test_create_completion_with_messages_only_works(self):
        """create_completion 只传 messages 时正常"""
        print("\n[TEST] create_completion with messages only -> works normally")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with patch.object(adapter, '_check_error'):
            with patch.object(adapter, '_to_openai_format', return_value={'choices': [{'message': {'content': 'ok'}}]}):
                with patch('cnllm.entry.http.BaseHttpClient') as mock_http:
                    mock_client = MagicMock()
                    mock_client.post.return_value = {}
                    mock_http.return_value = mock_client

                    messages = [{"role": "user", "content": "hello"}]
                    result = adapter.create_completion(messages=messages)

                    print(f"  Result: {result}")
                    assert result['choices'][0]['message']['content'] == 'ok'
                    print("  PASS: Works with messages only")

    def test_create_completion_with_prompt_only_works(self):
        """create_completion 只传 prompt 时正常（prompt 转 messages）"""
        print("\n[TEST] create_completion with prompt only -> works normally (prompt converted to messages)")

        adapter = MiniMaxAdapter(api_key='test', model='minimax-m2.7')

        with patch.object(adapter, '_check_error'):
            with patch.object(adapter, '_to_openai_format', return_value={'choices': [{'message': {'content': 'ok'}}]}):
                with patch('cnllm.entry.http.BaseHttpClient') as mock_http:
                    mock_client = MagicMock()
                    mock_client.post.return_value = {}
                    mock_http.return_value = mock_client

                    result = adapter.create_completion(prompt='hello')

                    print(f"  Result: {result}")
                    assert result['choices'][0]['message']['content'] == 'ok'
                    print("  PASS: Works with prompt only")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
