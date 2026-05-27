"""
LiveDict 单元测试（使用 unittest，Mock Rich）
"""
import unittest
from unittest.mock import MagicMock, patch


class TestLiveDict(unittest.TestCase):
    """LiveDict 上下文管理器和刷新行为"""

    def setUp(self):
        self.mock_live = MagicMock()
        self.mock_live.__enter__ = MagicMock(return_value=self.mock_live)
        self.mock_live.__exit__ = MagicMock(return_value=None)

        self.rich_patches = [
            patch('rich.live.Live', return_value=self.mock_live),
            patch('rich.text.Text'),
            patch('rich.pretty.Pretty', return_value="Pretty(...)"),
        ]
        for p in self.rich_patches:
            p.start()

        from cnllm.core.accumulators.live import LiveDict
        self.mock_acc = MagicMock()
        self.ld = LiveDict(self.mock_acc)

    def tearDown(self):
        for p in self.rich_patches:
            p.stop()

    def test_enter_creates_live(self):
        result = self.ld.__enter__()
        self.assertIs(result, self.ld)
        self.assertIs(self.ld._live, self.mock_live)

    def test_exit_closes_live(self):
        self.ld._live = self.mock_live
        self.ld.__exit__(None, None, None)
        self.mock_live.__exit__.assert_called_once()

    def test_exit_no_live_does_not_raise(self):
        self.ld._live = None
        self.ld.__exit__(None, None, None)

    def test_refresh_updates_live(self):
        self.ld._live = self.mock_live
        self.mock_acc._accumulate.return_value = {"choices": [{"delta": {"content": "测试"}}]}
        self.ld.refresh()
        self.mock_acc._accumulate.assert_called_once()
        self.mock_live.update.assert_called_once()

    def test_refresh_empty_chunks(self):
        self.ld._live = self.mock_live
        self.mock_acc._accumulate.return_value = {}
        self.ld.refresh()
        self.mock_live.update.assert_called_once()

    def test_with_statement(self):
        with self.ld:
            self.ld.refresh()
        self.mock_live.__enter__.assert_called_once()
        self.mock_live.__exit__.assert_called_once()
        self.mock_live.update.assert_called_once()


class TestReprPropertyOnAccumulator(unittest.TestCase):
    """StreamBaseAccumulator.repr property"""

    def setUp(self):
        self.rich_patches = [
            patch('rich.live.Live'),
            patch('rich.text.Text'),
            patch('rich.pretty.Pretty'),
        ]
        for p in self.rich_patches:
            p.start()

    def tearDown(self):
        for p in self.rich_patches:
            p.stop()

    def test_repr_property_returns_livedict(self):
        from cnllm.core.accumulators.base import StreamBaseAccumulator
        from cnllm.core.accumulators.live import LiveDict

        mock_adapter = MagicMock()
        acc = StreamBaseAccumulator(mock_adapter)
        view_obj = acc.repr
        self.assertIsInstance(view_obj, LiveDict)
        self.assertIs(view_obj._acc, acc)

    def test_from_chunks_repr_property(self):
        from cnllm.core.accumulators.single_accumulator import StreamAccumulator
        from cnllm.core.accumulators.live import LiveDict

        acc = StreamAccumulator.from_chunks([])
        view_obj = acc.repr
        self.assertIsInstance(view_obj, LiveDict)

    def test_multiple_calls_return_new_livedict(self):
        from cnllm.core.accumulators.base import StreamBaseAccumulator

        mock_adapter = MagicMock()
        acc = StreamBaseAccumulator(mock_adapter)
        v1 = acc.repr
        v2 = acc.repr
        self.assertIsNot(v1, v2)


if __name__ == "__main__":
    unittest.main()
