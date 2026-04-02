import pytest
import os
from unittest.mock import patch, MagicMock


class TestParameterMergingAndOverride:
    """测试参数合并和覆盖机制"""

    def test_client_init_with_request_params_stored_as_defaults(self):
        """客户端入口可以传请求参数作为默认值"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            temperature=0.7,
            max_tokens=100,
            stream=False
        )

        assert client.temperature == 0.7
        assert client.max_tokens == 100
        assert client.stream == False

    def test_invoke_params_override_client_defaults(self):
        """调用入口参数覆盖客户端默认值"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            temperature=0.7,
            max_tokens=100
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', temperature=0.9, max_tokens=200)

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['temperature'] == 0.9
            assert call_kwargs['max_tokens'] == 200

    def test_invoke_without_params_uses_client_defaults(self):
        """调用入口不传参数时使用客户端默认值"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            temperature=0.7,
            max_tokens=100
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['max_tokens'] == 100

    def test_kwargs_merge_and_override(self):
        """额外参数合并和覆盖"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            top_p=0.9,
            organization='org-client'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', top_p=0.95, organization='org-invoke')

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['top_p'] == 0.95
            assert call_kwargs['organization'] == 'org-invoke'

    def test_client_kwargs_defaults(self):
        """客户端入口 kwargs 作为默认参数"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            top_p=0.9
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['top_p'] == 0.9


class TestConfigParamsOverride:
    """测试配置参数覆盖"""

    def test_timeout_override(self):
        """timeout 参数覆盖"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            timeout=30
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', timeout=60)

            call_kwargs = mock_get_adapter.call_args.kwargs
            assert call_kwargs['timeout'] == 60

    def test_timeout_defaults_to_client_value(self):
        """不传 timeout 时使用客户端值"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            timeout=30
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            call_kwargs = mock_get_adapter.call_args.kwargs
            assert call_kwargs['timeout'] == 30


class TestApiKeyOverride:
    """测试 API Key 覆盖"""

    def test_invoke_api_key_overrides_client(self):
        """调用入口 api_key 覆盖客户端"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='client-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', api_key='invoke-key')

            call_args = mock_get_adapter.call_args
            assert call_args[0][1] == 'invoke-key'

    def test_invoke_without_api_key_uses_client(self):
        """调用入口不传 api_key 时使用客户端"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='client-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            call_args = mock_get_adapter.call_args
            assert call_args[0][1] == 'client-key'


class TestModelOverride:
    """测试模型覆盖"""

    def test_invoke_model_overrides_client_model(self):
        """调用入口 model 覆盖客户端主模型"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', model='minimax-m2.5')

            mock_get_adapter.assert_called_once()
            call_args = mock_get_adapter.call_args
            assert call_args[0][0] == 'minimax-m2.5'


class TestFallbackMechanism:
    """测试 Fallback 机制"""

    def test_fallback_triggered_on_primary_failure(self):
        """主模型失败时触发 fallback 并成功返回"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            fallback_models={'minimax-m2.5': None}
        )

        adapters_created = []

        def capture_adapter(model, api_key, **kwargs):
            mock_adapter = MagicMock()
            if len(adapters_created) == 0:
                mock_adapter.create_completion.side_effect = Exception("Primary failed")
            else:
                mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'fallback success'}}]}
            adapters_created.append((model, api_key))
            return mock_adapter

        with patch.object(client.chat.parent, '_get_adapter', side_effect=capture_adapter):
            resp = client.chat.create(prompt='hello')

            assert len(adapters_created) == 2
            assert adapters_created[0][0] == 'minimax-m2.7'
            assert adapters_created[1][0] == 'minimax-m2.5'
            assert resp['choices'][0]['message']['content'] == 'fallback success'

    def test_fallback_uses_primary_api_key_when_fallback_key_is_none(self):
        """Fallback 使用主模型的 api_key"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='primary-key',
            fallback_models={'minimax-m2.5': None}
        )

        adapters_created = []

        def capture_adapter(model, api_key, **kwargs):
            mock_adapter = MagicMock()
            if model == 'minimax-m2.7':
                mock_adapter.create_completion.side_effect = Exception("Primary failed")
            else:
                mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'ok'}}]}
            adapters_created.append((model, api_key))
            return mock_adapter

        with patch.object(client.chat.parent, '_get_adapter', side_effect=capture_adapter):
            with patch.object(client.chat.parent, '_on_fallback'):
                client.chat.create(prompt='hello')

        assert adapters_created[0][1] == 'primary-key'
        assert adapters_created[1][1] == 'primary-key'

    def test_fallback_with_custom_api_key(self):
        """Fallback 使用指定的 api_key"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='primary-key',
            fallback_models={'minimax-m2.5': 'fallback-key'}
        )

        adapters_created = []

        def capture_adapter(model, api_key, **kwargs):
            mock_adapter = MagicMock()
            if model == 'minimax-m2.7':
                mock_adapter.create_completion.side_effect = Exception("Primary failed")
            else:
                mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'ok'}}]}
            adapters_created.append((model, api_key))
            return mock_adapter

        with patch.object(client.chat.parent, '_get_adapter', side_effect=capture_adapter):
            with patch.object(client.chat.parent, '_on_fallback'):
                client.chat.create(prompt='hello')

        assert adapters_created[0][1] == 'primary-key'
        assert adapters_created[1][1] == 'fallback-key'

    def test_fallback_params_from_invoke(self):
        """Fallback 时调用入口参数仍然生效"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            fallback_models={'minimax-m2.5': None}
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter_fail = MagicMock()
            mock_adapter_fail.create_completion.side_effect = Exception("Primary failed")

            mock_adapter_success = MagicMock()
            mock_adapter_success.create_completion.return_value = {'choices': [{'message': {'content': 'ok'}}]}

            mock_get_adapter.side_effect = [mock_adapter_fail, mock_adapter_success]

            with patch.object(client.chat.parent, '_on_fallback'):
                client.chat.create(prompt='hello', temperature=0.8, max_tokens=150)

            success_call = mock_adapter_success.create_completion
            assert success_call.call_args.kwargs['temperature'] == 0.8
            assert success_call.call_args.kwargs['max_tokens'] == 150

    def test_fallback_config_not_overrideable(self):
        """fallback_models 只能在客户端入口设置，调用入口无法覆盖"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            fallback_models={'minimax-m2.5': None}
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            with pytest.raises(TypeError, match="chat.create 不接受 fallback_models 参数"):
                client.chat.create(prompt='hello', fallback_models={'minimax-m2': 'key'})

            assert client.fallback_models == {'minimax-m2.5': None}


class TestNoFallbackMode:
    """测试非 fallback 模式"""

    def test_no_fallback_when_model_provided(self):
        """调用入口传 model 时不走 fallback"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            fallback_models={'minimax-m2.5': None}
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', model='minimax-m2.5')

            assert mock_get_adapter.call_count == 1


class TestPromptConversion:
    """测试 prompt 到 messages 的转换"""

    def test_prompt_converted_to_messages(self):
        """prompt 参数被转换为 messages"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello world')

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['messages'] == [{'role': 'user', 'content': 'hello world'}]

    def test_messages_passed_directly(self):
        """messages 参数直接传递"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            messages = [{'role': 'user', 'content': 'hi'}]
            client.chat.create(messages=messages)

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['messages'] == messages


class TestStreamParameter:
    """测试 stream 参数"""

    def test_stream_false_by_default(self):
        """stream 默认为 False"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key'
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['stream'] == False

    def test_stream_override_on_invoke(self):
        """调用入口覆盖 stream"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            stream=False
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', stream=True)

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['stream'] == True


class TestRequestParamsFlow:
    """测试请求参数流向"""

    def test_request_params_flow_to_create_completion(self):
        """请求参数正确流向 create_completion"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            temperature=0.5,
            max_tokens=50
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(
                prompt='hello',
                temperature=0.9,
                max_tokens=200,
                top_p=0.95,
                organization='my-org'
            )

            create_call = mock_adapter.create_completion.call_args.kwargs
            assert create_call['temperature'] == 0.9
            assert create_call['max_tokens'] == 200
            assert create_call['top_p'] == 0.95
            assert create_call['organization'] == 'my-org'


class TestInternalParamsExclusion:
    """测试内部参数被正确排除"""

    def test_fallback_models_not_in_adapter_call(self):
        """fallback_models 不会传给 adapter"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            fallback_models={'minimax-m2.5': None}
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello')

            create_call = mock_adapter.create_completion.call_args.kwargs
            assert 'fallback_models' not in create_call


class TestEdgeCases:
    """边界情况测试"""

    def test_client_kwargs_merged_with_invoke_kwargs(self):
        """客户端 kwargs 和调用入口 kwargs 合并"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            top_p=0.8,
            presence_penalty=0.5
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', top_p=0.9)

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['top_p'] == 0.9
            assert call_kwargs['presence_penalty'] == 0.5

    def test_invoke_kwargs_override_client_kwargs(self):
        """调用入口 kwargs 覆盖客户端 kwargs"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='test-key',
            top_p=0.8,
            presence_penalty=0.5
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(prompt='hello', top_p=0.9, presence_penalty=0.6)

            call_kwargs = mock_adapter.create_completion.call_args.kwargs
            assert call_kwargs['top_p'] == 0.9
            assert call_kwargs['presence_penalty'] == 0.6


class TestComprehensiveScenarios:
    """综合场景测试"""

    def test_full_parameter_override_scenario(self):
        """完整参数覆盖场景"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='client-key',
            temperature=0.3,
            max_tokens=100,
            timeout=60,
            top_p=0.8
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.create_completion.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_get_adapter.return_value = mock_adapter

            client.chat.create(
                prompt='hello',
                api_key='invoke-key',
                temperature=0.9,
                max_tokens=500,
                timeout=120,
                top_p=0.95,
                presence_penalty=0.5
            )

            get_adapter_call = mock_get_adapter.call_args
            assert get_adapter_call[0][1] == 'invoke-key'
            assert get_adapter_call[1]['timeout'] == 120

            create_call = mock_adapter.create_completion.call_args.kwargs
            assert create_call['temperature'] == 0.9
            assert create_call['max_tokens'] == 500
            assert create_call['top_p'] == 0.95
            assert create_call['presence_penalty'] == 0.5

    def test_fallback_with_all_override_params(self):
        """带完整参数覆盖的 Fallback 场景"""
        from cnllm import CNLLM

        client = CNLLM(
            model='minimax-m2.7',
            api_key='primary-key',
            fallback_models={'minimax-m2.5': 'fallback-key'},
            temperature=0.5,
            max_tokens=200
        )

        with patch.object(client.chat.parent, '_get_adapter') as mock_get_adapter:
            mock_adapter_fail = MagicMock()
            mock_adapter_fail.create_completion.side_effect = Exception("Primary failed")

            mock_adapter_success = MagicMock()
            mock_adapter_success.create_completion.return_value = {'choices': [{'message': {'content': 'success'}}]}

            mock_get_adapter.side_effect = [mock_adapter_fail, mock_adapter_success]

            with patch.object(client.chat.parent, '_on_fallback'):
                resp = client.chat.create(
                    prompt='hello',
                    temperature=0.8,
                    max_tokens=300
                )

            assert resp['choices'][0]['message']['content'] == 'success'

            success_call = mock_adapter_success.create_completion
            assert success_call.call_args.kwargs['temperature'] == 0.8
            assert success_call.call_args.kwargs['max_tokens'] == 300