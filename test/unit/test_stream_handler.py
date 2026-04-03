import pytest
from unittest.mock import Mock, MagicMock
from cnllm.utils.stream import StreamHandler, SSEDecoder


class TestSSEDecoder:
    def test_decode_sse_basic(self):
        raw_lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " World"}}]}',
            b'data: [DONE]'
        ]
        result = list(SSEDecoder.decode_stream(iter(raw_lines)))

        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "Hello"
        assert result[1]["choices"][0]["delta"]["content"] == " World"

    def test_decode_sse_empty(self):
        raw_lines = []
        result = list(SSEDecoder.decode_stream(iter(raw_lines)))
        assert len(result) == 0

    def test_decode_sse_with_tool_calls(self):
        raw_lines = [
            b'data: {"choices": [{"delta": {"content": ""}, "finish_reason": "tool_calls"}]}',
            b'data: {"choices": [{"delta": {"tool_calls": [{"function": {"name": "test"}}]}}]}',
            b'data: [DONE]'
        ]
        result = list(SSEDecoder.decode_stream(iter(raw_lines)))

        assert len(result) == 2
        assert result[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "test"


class TestStreamHandler:
    def test_handle_stream_basic(self):
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " World"}}]}',
        ]

        mock_client = Mock()
        mock_client.post_stream.return_value = iter(sse_lines)

        def format_func(raw, model):
            return raw

        result = list(StreamHandler.handle_stream(
            mock_client, "/chat", {"model": "test"}, format_func
        ))

        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "Hello"
        assert result[1]["choices"][0]["delta"]["content"] == " World"

    def test_handle_stream_empty_response(self):
        mock_client = Mock()
        mock_client.post_stream.return_value = iter([])

        def format_func(raw, model):
            return raw

        result = list(StreamHandler.handle_stream(
            mock_client, "/chat", {"model": "test"}, format_func
        ))

        assert len(result) == 0

    def test_handle_stream_with_tool_calls(self):
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": ""}, "finish_reason": "tool_calls"}]}',
            b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call_1", "function": {"name": "test"}}]}}]}',
        ]

        mock_client = Mock()
        mock_client.post_stream.return_value = iter(sse_lines)

        def format_func(raw, model):
            return raw

        result = list(StreamHandler.handle_stream(
            mock_client, "/chat", {"model": "test"}, format_func
        ))

        assert len(result) == 2
        assert result[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "test"

    def test_handle_stream_with_reasoning_content(self):
        sse_lines = [
            b'data: {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}',
            b'data: {"choices": [{"delta": {"content": "final answer"}}]}',
        ]

        mock_client = Mock()
        mock_client.post_stream.return_value = iter(sse_lines)

        def format_func(raw, model):
            delta = raw.get("choices", [{}])[0].get("delta", {})
            content = delta.get("reasoning_content") or delta.get("content", "")
            return {"choices": [{"delta": {"content": content}}]}

        result = list(StreamHandler.handle_stream(
            mock_client, "/chat", {"model": "test"}, format_func
        ))

        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "thinking..."
        assert result[1]["choices"][0]["delta"]["content"] == "final answer"

    def test_handle_stream_passes_model_from_payload(self):
        sse_lines = [b'data: {"choices": [{"delta": {"content": "test"}}]}']

        mock_client = Mock()
        mock_client.post_stream.return_value = iter(sse_lines)

        captured_model = []
        def format_func(raw, model):
            captured_model.append(model)
            return raw

        list(StreamHandler.handle_stream(
            mock_client, "/chat", {"model": "minimax-m2.7"}, format_func
        ))

        assert captured_model == ["minimax-m2.7"]

    def test_handle_stream_error_propagation(self):
        mock_client = Mock()
        mock_client.post_stream.side_effect = Exception("Network error")

        def format_func(raw, model):
            return raw

        with pytest.raises(Exception, match="Network error"):
            list(StreamHandler.handle_stream(
                mock_client, "/chat", {"model": "test"}, format_func
            ))


class TestStreamHandlerIntegration:
    def test_stream_handler_with_mock_adapter(self):
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        MiniMaxAdapter._class_config = None
        MiniMaxAdapter._supported_models = []

        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="abab6.5-chat"
        )

        mock_raw_chunk = {
            "choices": [{
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": "stop"
            }]
        }

        result = adapter._to_openai_stream_format(mock_raw_chunk, "abab6.5-chat")

        assert result["choices"][0]["delta"]["content"] == "Hello"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_stream_handler_reasoning_content(self):
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        mock_raw_chunk = {
            "choices": [{
                "delta": {
                    "reasoning_content": "Let me think..."
                }
            }]
        }

        adapter = MiniMaxAdapter(api_key="test", model="abab6.5-chat")
        result = MiniMaxAdapter._to_openai_stream_format(
            adapter,
            mock_raw_chunk,
            "abab6.5-chat"
        )

        assert result["choices"][0]["delta"].get("content") == "", "reasoning_content 不应在 delta.content 中"
        assert adapter._raw_response.get("_thinking") == "Let me think...", "reasoning_content 应存入 _thinking"

    def test_stream_handler_tool_calls_in_delta(self):
        from cnllm.core.vendor.minimax import MiniMaxAdapter

        mock_raw_chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Boston"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        adapter = MiniMaxAdapter(api_key="test", model="abab6.5-chat")
        result = adapter._to_openai_stream_format(mock_raw_chunk, "abab6.5-chat")

        assert result["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamHandlerEndToEnd:
    def test_full_stream_pipeline(self):
        from cnllm.core.adapter import BaseAdapter

        class TestAdapter(BaseAdapter):
            ADAPTER_NAME = "test"
            CONFIG_DIR = ""

            def _to_openai_format(self, raw, model):
                return raw

            def _to_openai_stream_format(self, raw, model):
                delta = raw.get("choices", [{}])[0].get("delta", {})
                return {
                    "choices": [{
                        "delta": delta,
                        "finish_reason": raw.get("choices", [{}])[0].get("finish_reason")
                    }]
                }

        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "Part 1"}, "finish_reason": null}]}',
            b'data: {"choices": [{"delta": {"content": "Part 2"}, "finish_reason": null}]}',
            b'data: {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}',
        ]

        mock_client = Mock()
        mock_client.post_stream.return_value = iter(sse_lines)

        adapter = TestAdapter(api_key="test", model="test-model")
        handler = StreamHandler()

        results = list(handler.handle_stream(
            mock_client, "/test", {"model": "test-model"}, adapter._to_openai_stream_format
        ))

        assert len(results) == 3
        assert results[0]["choices"][0]["delta"]["content"] == "Part 1"
        assert results[1]["choices"][0]["delta"]["content"] == "Part 2"
        assert results[2]["choices"][0]["finish_reason"] == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
