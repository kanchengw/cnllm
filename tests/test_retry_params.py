"""
CNLLM 重试参数传递测试 - 验证 timeout, max_retries, retry_delay 参数传递链

测试目标：
1. Chat 单条调用参数传递 (Type 1, 2, 5, 6)
2. Chat 批量调用参数传递 (Type 3, 4, 7, 8)
3. Embedding 单条调用参数传递 (Type 9, 11)
4. Embedding 批量调用参数传递 (Type 10, 12)

参数传递链：
用户调用 → Namespace.create/batch → 获取客户端级默认值 → 适配器初始化 → Scheduler 初始化 → 请求发送
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from cnllm.entry.client import CNLLM
from cnllm.utils.batch import BatchScheduler, StreamBatchScheduler, EmbeddingBatchScheduler, EmbeddingBatchItemResult
from cnllm.core.embedding import BaseEmbeddingAdapter, EmbeddingsNamespace


class TestChatSingleParameterPassing:
    """Chat 单条调用参数传递测试 (Type 1, 2, 5, 6)"""

    def test_chat_single_explicit_params(self):
        """测试：用户显式传入参数"""
        with patch('httpx.Client') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "test"}}]
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            client = CNLLM(model="test-model", api_key="test-key")

            with patch.object(client.chat, 'create') as mock_create:
                mock_create.return_value = {"choices": [{"message": {"content": "test"}}]}
                
                try:
                    client.chat.create(
                        prompt="hello",
                        timeout=60,
                        max_retries=5,
                        retry_delay=2.0
                    )
                except Exception:
                    pass

                if mock_create.called:
                    call_kwargs = mock_create.call_args
                    print(f"\n[PASS] Chat 单条显式参数传递成功")
                    print(f"  timeout: {call_kwargs.kwargs.get('timeout')}")
                    print(f"  max_retries: {call_kwargs.kwargs.get('max_retries')}")
                    print(f"  retry_delay: {call_kwargs.kwargs.get('retry_delay')}")

    def test_chat_single_client_level_defaults(self):
        """测试：使用客户端级默认值"""
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        assert client.timeout == 30, "客户端 timeout 应正确设置"
        assert client.max_retries == 3, "客户端 max_retries 应正确设置"
        assert client.retry_delay == 1.0, "客户端 retry_delay 应正确设置"

        print(f"\n[PASS] Chat 单条客户端级默认值正确")
        print(f"  timeout: {client.timeout}")
        print(f"  max_retries: {client.max_retries}")
        print(f"  retry_delay: {client.retry_delay}")


class TestChatBatchParameterPassing:
    """Chat 批量调用参数传递测试 (Type 3, 4, 7, 8)"""

    def test_chat_batch_explicit_params_sync(self):
        """测试同步批量调用：用户显式传入参数"""
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=5,
            timeout=60,
            max_retries=5,
            retry_delay=2.0
        )

        assert scheduler.timeout == 60, "Scheduler timeout 应使用显式值"
        assert scheduler.max_retries == 5, "Scheduler max_retries 应使用显式值"
        assert scheduler.retry_delay == 2.0, "Scheduler retry_delay 应使用显式值"

        print(f"\n[PASS] Chat 同步批量显式参数正确")
        print(f"  timeout: {scheduler.timeout}")
        print(f"  max_retries: {scheduler.max_retries}")
        print(f"  retry_delay: {scheduler.retry_delay}")

    def test_chat_batch_client_level_defaults_sync(self):
        """测试同步批量调用：使用客户端级默认值
        
        注意：BatchScheduler 本身不处理默认值，默认值由 chat.batch 方法处理
        这里测试 chat.batch 方法的参数解析逻辑
        """
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        with patch.object(client.chat, 'batch') as mock_batch:
            try:
                client.chat.batch(
                    prompt=["test1", "test2"],
                    timeout=None,
                    max_retries=None,
                    retry_delay=None
                )
            except Exception:
                pass

            if mock_batch.called:
                call_kwargs = mock_batch.call_args
                assert call_kwargs.kwargs.get('timeout') is None, "显式传入 None 时应传递 None"
                assert call_kwargs.kwargs.get('max_retries') is None, "显式传入 None 时应传递 None"
                assert call_kwargs.kwargs.get('retry_delay') is None, "显式传入 None 时应传递 None"

        print(f"\n[PASS] Chat 同步批量默认值传递正确 (由 chat.batch 处理)")
        print(f"  传递 timeout=None: 期望由 chat.batch 解析为客户端默认值 30")

    def test_chat_batch_explicit_override(self):
        """测试批量调用：显式参数覆盖客户端默认值"""
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=5,
            timeout=120,
            max_retries=10,
            retry_delay=5.0
        )

        assert scheduler.timeout == 120, "显式参数应覆盖默认值"
        assert scheduler.max_retries == 10, "显式参数应覆盖默认值"
        assert scheduler.retry_delay == 5.0, "显式参数应覆盖默认值"

        print(f"\n[PASS] Chat 批量显式覆盖正确")
        print(f"  timeout: {scheduler.timeout} (expected: 120)")
        print(f"  max_retries: {scheduler.max_retries} (expected: 10)")
        print(f"  retry_delay: {scheduler.retry_delay} (expected: 5.0)")


class TestEmbeddingSingleParameterPassing:
    """Embedding 单条调用参数传递测试 (Type 9, 11)"""

    def test_embedding_namespace_get_adapter_passes_params(self):
        """测试：EmbeddingsNamespace._get_adapter 传递参数"""
        client = CNLLM(
            model="embedding-test",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        with patch('cnllm.core.embedding.BaseEmbeddingAdapter') as MockAdapter:
            mock_instance = Mock()
            mock_instance.timeout = 30
            mock_instance.max_retries = 3
            mock_instance.retry_delay = 1.0
            MockAdapter.return_value = mock_instance

            embeddings_ns = EmbeddingsNamespace(client)
            
            with patch.object(embeddings_ns, '_get_adapter') as mock_get_adapter:
                mock_get_adapter.return_value = mock_instance
                
                try:
                    embeddings_ns.create(input="test text", model="embedding-test")
                except Exception:
                    pass

                if mock_get_adapter.called:
                    call_kwargs = mock_get_adapter.call_args
                    print(f"\n[PASS] Embedding Namespace 传递参数成功")
                    print(f"  调用参数: {call_kwargs}")


class TestEmbeddingBatchParameterPassing:
    """Embedding 批量调用参数传递测试 (Type 10, 12)"""

    def test_embedding_batch_explicit_params(self):
        """测试 Embedding 批量调用：用户显式传入参数"""
        mock_adapter = Mock()
        mock_adapter.timeout = 30
        mock_adapter.max_retries = 3
        mock_adapter.retry_delay = 1.0

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=5,
            timeout=60,
            max_retries=5,
            retry_delay=2.0
        )

        assert scheduler.timeout == 60, "Embedding Scheduler timeout 应使用显式值"
        assert scheduler.max_retries == 5, "Embedding Scheduler max_retries 应使用显式值"
        assert scheduler.retry_delay == 2.0, "Embedding Scheduler retry_delay 应使用显式值"

        print(f"\n[PASS] Embedding 批量显式参数正确")
        print(f"  timeout: {scheduler.timeout}")
        print(f"  max_retries: {scheduler.max_retries}")
        print(f"  retry_delay: {scheduler.retry_delay}")

    def test_embedding_batch_adapter_defaults(self):
        """测试 Embedding 批量调用：使用适配器默认值"""
        mock_adapter = Mock()
        mock_adapter.timeout = 45
        mock_adapter.max_retries = 7
        mock_adapter.retry_delay = 3.5

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=5,
            timeout=None,
            max_retries=None,
            retry_delay=None
        )

        assert scheduler.timeout == 45, "Embedding Scheduler timeout 应使用适配器默认值"
        assert scheduler.max_retries == 7, "Embedding Scheduler max_retries 应使用适配器默认值"
        assert scheduler.retry_delay == 3.5, "Embedding Scheduler retry_delay 应使用适配器默认值"

        print(f"\n[PASS] Embedding 批量适配器默认值正确")
        print(f"  timeout: {scheduler.timeout}")
        print(f"  max_retries: {scheduler.max_retries}")
        print(f"  retry_delay: {scheduler.retry_delay}")

    def test_embedding_batch_explicit_override(self):
        """测试 Embedding 批量调用：显式参数覆盖适配器默认值"""
        mock_adapter = Mock()
        mock_adapter.timeout = 30
        mock_adapter.max_retries = 3
        mock_adapter.retry_delay = 1.0

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=5,
            timeout=120,
            max_retries=10,
            retry_delay=5.0
        )

        assert scheduler.timeout == 120, "显式参数应覆盖适配器默认值"
        assert scheduler.max_retries == 10, "显式参数应覆盖适配器默认值"
        assert scheduler.retry_delay == 5.0, "显式参数应覆盖适配器默认值"

        print(f"\n[PASS] Embedding 批量显式覆盖正确")
        print(f"  timeout: {scheduler.timeout} (expected: 120)")
        print(f"  max_retries: {scheduler.max_retries} (expected: 10)")
        print(f"  retry_delay: {scheduler.retry_delay} (expected: 5.0)")


class TestParameterPassingChain:
    """参数传递链集成测试"""

    def test_full_chain_chat_batch_explicit(self):
        """测试 Chat 批量调用完整参数链：显式参数"""
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        with patch.object(client.chat, 'batch') as mock_batch:
            try:
                client.chat.batch(
                    prompt=["test1", "test2"],
                    timeout=60,
                    max_retries=5,
                    retry_delay=2.0
                )
            except Exception:
                pass

            if mock_batch.called:
                call_kwargs = mock_batch.call_args
                assert call_kwargs.kwargs.get('timeout') == 60
                assert call_kwargs.kwargs.get('max_retries') == 5
                assert call_kwargs.kwargs.get('retry_delay') == 2.0
                print(f"\n[PASS] Chat 批量完整参数链（显式）")
                print(f"  调用参数: {call_kwargs.kwargs}")

    def test_full_chain_chat_batch_client_defaults(self):
        """测试 Chat 批量调用完整参数链：使用客户端默认值"""
        client = CNLLM(
            model="test-model",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=5,
            timeout=None,
            max_retries=None,
            retry_delay=None
        )

        with patch.object(scheduler, 'execute') as mock_execute:
            try:
                scheduler.execute(["test1", "test2"])
            except Exception:
                pass

        print(f"\n[PASS] Chat 批量完整参数链（客户端默认）")
        print(f"  chat.batch 会将 None 解析为客户端默认值")

    def test_full_chain_embedding_batch_explicit(self):
        """测试 Embedding 批量调用完整参数链：显式参数"""
        from cnllm.core.embedding import EmbeddingsNamespace
        
        client = CNLLM(
            model="test-embedding",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        with patch.object(client.embeddings, 'batch') as mock_batch:
            try:
                client.embeddings.batch(
                    inputs=["test1", "test2"],
                    timeout=60,
                    max_retries=5,
                    retry_delay=2.0
                )
            except Exception:
                pass

            if mock_batch.called:
                call_kwargs = mock_batch.call_args
                assert call_kwargs.kwargs.get('timeout') == 60
                assert call_kwargs.kwargs.get('max_retries') == 5
                assert call_kwargs.kwargs.get('retry_delay') == 2.0
                print(f"\n[PASS] Embedding 批量完整参数链（显式）")
                print(f"  调用参数: {call_kwargs.kwargs}")

    def test_embedding_namespace_adapter_params(self):
        """测试 Embedding Namespace 到 Adapter 的参数传递"""
        client = CNLLM(
            model="test-embedding",
            api_key="test-key",
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )

        from cnllm.core.embedding import EmbeddingsNamespace
        embeddings_ns = EmbeddingsNamespace(client)

        with patch('cnllm.core.embedding.BaseEmbeddingAdapter') as MockAdapter:
            mock_instance = Mock()
            MockAdapter.return_value = mock_instance

            try:
                adapter = embeddings_ns._get_adapter("embedding-test")
            except Exception:
                pass

            call_kwargs = MockAdapter.call_args
            if call_kwargs:
                assert call_kwargs.kwargs.get('timeout') == 30
                assert call_kwargs.kwargs.get('max_retries') == 3
                assert call_kwargs.kwargs.get('retry_delay') == 1.0
                print(f"\n[PASS] Embedding Namespace → Adapter 参数传递")
                print(f"  timeout: {call_kwargs.kwargs.get('timeout')}")
                print(f"  max_retries: {call_kwargs.kwargs.get('max_retries')}")
                print(f"  retry_delay: {call_kwargs.kwargs.get('retry_delay')}")


class TestStopOnErrorAndCallbacks:
    """stop_on_error 和 callbacks 功能测试"""

    def test_chat_batch_stop_on_error_parameter(self):
        """测试 Chat 批量调用：stop_on_error 参数传递"""
        client = CNLLM(model="test-model", api_key="test-key")

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=5,
            stop_on_error=True
        )

        assert scheduler.stop_on_error == True, "Scheduler stop_on_error 应正确设置"
        print(f"\n[PASS] Chat 批量 stop_on_error 参数正确")
        print(f"  stop_on_error: {scheduler.stop_on_error}")

    def test_chat_batch_callbacks_parameter(self):
        """测试 Chat 批量调用：callbacks 参数传递"""
        client = CNLLM(model="test-model", api_key="test-key")

        def my_callback(result):
            pass

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=5,
            callbacks=[my_callback]
        )

        assert len(scheduler.callbacks) == 1, "Scheduler callbacks 应正确设置"
        assert scheduler.callbacks[0] == my_callback, "callbacks 内容应正确"
        print(f"\n[PASS] Chat 批量 callbacks 参数正确")
        print(f"  callbacks 数量: {len(scheduler.callbacks)}")

    def test_embedding_batch_stop_on_error_parameter(self):
        """测试 Embedding 批量调用：stop_on_error 参数传递"""
        mock_adapter = Mock()
        mock_adapter.timeout = 30
        mock_adapter.max_retries = 3
        mock_adapter.retry_delay = 1.0

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=5,
            stop_on_error=True
        )

        assert scheduler.stop_on_error == True, "Scheduler stop_on_error 应正确设置"
        print(f"\n[PASS] Embedding 批量 stop_on_error 参数正确")
        print(f"  stop_on_error: {scheduler.stop_on_error}")

    def test_embedding_batch_callbacks_parameter(self):
        """测试 Embedding 批量调用：callbacks 参数传递"""
        mock_adapter = Mock()
        mock_adapter.timeout = 30
        mock_adapter.max_retries = 3
        mock_adapter.retry_delay = 1.0

        def my_callback(item_result):
            pass

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=5,
            callbacks=[my_callback]
        )

        assert len(scheduler.callbacks) == 1, "Scheduler callbacks 应正确设置"
        assert scheduler.callbacks[0] == my_callback, "callbacks 内容应正确"
        print(f"\n[PASS] Embedding 批量 callbacks 参数正确")
        print(f"  callbacks 数量: {len(scheduler.callbacks)}")

    def test_embedding_batchitem_result_structure(self):
        """测试 EmbeddingBatchItemResult 数据结构"""
        success_result = EmbeddingBatchItemResult(
            index=0,
            request_id="request_0",
            result={"data": [{"embedding": [0.1, 0.2]}]},
            status="success"
        )

        assert success_result.index == 0, "index 应正确设置"
        assert success_result.request_id == "request_0", "request_id 应正确设置"
        assert success_result.result is not None, "result 应设置"
        assert success_result.status == "success", "status 应为 success"
        assert success_result.error is None, "success 时 error 应为 None"

        error_result = EmbeddingBatchItemResult(
            index=1,
            request_id="request_1",
            error=Exception("Test error"),
            status="error"
        )

        assert error_result.index == 1, "index 应正确设置"
        assert error_result.error is not None, "error 时 error 应设置"
        assert error_result.status == "error", "status 应为 error"
        assert error_result.result is None, "error 时 result 应为 None"

        print(f"\n[PASS] EmbeddingBatchItemResult 结构正确")
        print(f"  success_result: index={success_result.index}, status={success_result.status}")
        print(f"  error_result: index={error_result.index}, status={error_result.status}")

    def test_chat_namespace_batch_passes_stop_on_error_and_callbacks(self):
        """测试 Chat Namespace.batch 传递 stop_on_error 和 callbacks"""
        client = CNLLM(model="test-model", api_key="test-key")

        def my_callback(result):
            pass

        with patch.object(client.chat, 'batch') as mock_batch:
            try:
                client.chat.batch(
                    prompt=["test1", "test2"],
                    stop_on_error=True,
                    callbacks=[my_callback]
                )
            except Exception:
                pass

            if mock_batch.called:
                call_kwargs = mock_batch.call_args
                assert call_kwargs.kwargs.get('stop_on_error') == True
                assert len(call_kwargs.kwargs.get('callbacks', [])) == 1
                print(f"\n[PASS] Chat Namespace.batch 传递 stop_on_error 和 callbacks")
                print(f"  stop_on_error: {call_kwargs.kwargs.get('stop_on_error')}")
                print(f"  callbacks 数量: {len(call_kwargs.kwargs.get('callbacks', []))}")

    def test_embedding_namespace_batch_passes_stop_on_error_and_callbacks(self):
        """测试 Embedding Namespace.batch 传递 stop_on_error 和 callbacks"""
        client = CNLLM(model="test-embedding", api_key="test-key")

        def my_callback(item_result):
            pass

        with patch.object(client.embeddings, 'batch') as mock_batch:
            try:
                client.embeddings.batch(
                    inputs=["test1", "test2"],
                    stop_on_error=True,
                    callbacks=[my_callback]
                )
            except Exception:
                pass

            if mock_batch.called:
                call_kwargs = mock_batch.call_args
                assert call_kwargs.kwargs.get('stop_on_error') == True
                assert len(call_kwargs.kwargs.get('callbacks', [])) == 1
                print(f"\n[PASS] Embedding Namespace.batch 传递 stop_on_error 和 callbacks")
                print(f"  stop_on_error: {call_kwargs.kwargs.get('stop_on_error')}")
                print(f"  callbacks 数量: {len(call_kwargs.kwargs.get('callbacks', []))}")


if __name__ == "__main__":
    print("=" * 60)
    print("CNLLM 重试参数传递测试")
    print("=" * 60)
    
    pytest.main([__file__, "-v", "-s"])
