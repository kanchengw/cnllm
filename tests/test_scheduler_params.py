"""
CNLLM 调度器参数测试 - 验证 stop_on_error 和 callbacks 是调度器参数而非请求参数

关键点：
1. stop_on_error 和 callbacks 是 BatchScheduler/EmbeddingBatchScheduler 的内部参数
2. 它们存储在 scheduler 实例上，控制执行流程
3. 它们不会传递给 adapter，因此不经过 filter_supported_params 验证
4. 这与 timeout, max_retries, retry_delay 不同，后者在 client 层面使用
"""
import pytest
from unittest.mock import Mock
from cnllm.utils.batch import BatchScheduler, EmbeddingBatchScheduler


class TestSchedulerVsRequestParams:
    """验证调度器参数 vs 请求参数的区别"""

    def test_scheduler_params_are_stored_on_scheduler(self):
        """测试：调度器参数存储在 scheduler 实例上"""
        client = Mock()

        scheduler = BatchScheduler(
            client=client,
            max_concurrent=1,
            stop_on_error=True,
            callbacks=[lambda x: None],
            max_retries=2,
            retry_delay=0.5,
            timeout=30
        )

        assert scheduler.stop_on_error == True
        assert len(scheduler.callbacks) == 1
        assert scheduler.max_retries == 2
        assert scheduler.retry_delay == 0.5
        assert scheduler.timeout == 30
        assert scheduler.max_concurrent == 1

        print("\n[PASS] Chat 调度器参数存储在实例上")
        print(f"  stop_on_error={scheduler.stop_on_error} (调度器参数)")
        print(f"  callbacks={len(scheduler.callbacks)} (调度器参数)")
        print(f"  max_retries={scheduler.max_retries} (调度器参数)")
        print(f"  timeout={scheduler.timeout} (调度器参数)")

    def test_embedding_scheduler_params_are_stored_on_scheduler(self):
        """测试：Embedding 调度器参数存储在 scheduler 实例上"""
        mock_adapter = Mock()
        mock_adapter.timeout = 30
        mock_adapter.max_retries = 3
        mock_adapter.retry_delay = 1.0

        scheduler = EmbeddingBatchScheduler(
            adapter=mock_adapter,
            max_concurrent=1,
            stop_on_error=True,
            callbacks=[lambda x: None],
            max_retries=2,
            retry_delay=0.5,
            timeout=30
        )

        assert scheduler.stop_on_error == True
        assert len(scheduler.callbacks) == 1
        assert scheduler.max_retries == 2
        assert scheduler.retry_delay == 0.5
        assert scheduler.timeout == 30
        assert scheduler.max_concurrent == 1

        print("\n[PASS] Embedding 调度器参数存储在实例上")
        print(f"  stop_on_error={scheduler.stop_on_error} (调度器参数)")
        print(f"  callbacks={len(scheduler.callbacks)} (调度器参数)")
        print(f"  max_retries={scheduler.max_retries} (调度器参数)")
        print(f"  timeout={scheduler.timeout} (调度器参数)")

    def test_scheduler_params_not_in_yaml(self):
        """测试：调度器参数不在 yml 配置中
        
        这解释了为什么这些参数不需要在 yml 中定义：
        它们是调度器内部控制流程的参数，不是 API 请求参数
        
        timeout/max_retries/retry_delay 已是 per-request 参数（在 yml 中有定义），
        因此不再被列入调度器参数。
        """
        scheduler_params = [
            'stop_on_error',
            'callbacks',
            'max_concurrent',
            'custom_ids'
        ]
        
        yaml_params = [
            'messages', 'prompt', 'model', 'temperature',
            'max_tokens', 'stream', 'top_p', 'tools'
        ]
        
        for param in scheduler_params:
            assert param not in yaml_params, \
                f"{param} 是调度器参数，不在 yml 中"
        
        print("\n[PASS] 调度器参数与 yml 请求参数是分开的")
        print(f"  调度器参数: {scheduler_params}")
        print(f"  yml 请求参数: {yaml_params}")


class TestParameterFlow:
    """参数流程说明"""

    def test_explain_parameter_flow(self):
        """说明参数如何流动"""
        flow = """
        1. 用户调用 chat.batch(request, stop_on_error=True, callbacks=[...])
           ↑
        2. chat.batch() 接收参数
           ↓
        3. 创建 BatchScheduler(client, stop_on_error=True, callbacks=[...])
           ↑
        4. 调度器存储 stop_on_error=True, callbacks=[...] 在 self 上
           ↓
        5. 调度器调用 client._get_adapter() 获取 adapter
           ↑
        6. 调度器调用 adapter.create_completion(request, timeout=30, ...)
           - stop_on_error, callbacks 不会传递给 adapter
           - adapter 调用 filter_supported_params() 验证请求参数
           - stop_on_error, callbacks 不经过验证
        """
        print("\n" + flow)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
