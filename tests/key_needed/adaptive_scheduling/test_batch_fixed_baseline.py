"""
MiniMax 固定策略 (rps=2, mc=3) 基线测试
30 条请求，不涉及控制器探针
"""
import os, sys, time, unittest
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
API_KEY = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MODEL = "minimax-m2.1"

@unittest.skipUnless(API_KEY, "need API Key")
class TestFixedBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_fixed_30(self):
        """固定 rps=2, mc=3 跑 30 条"""
        from cnllm.utils.scheduler.base import BatchScheduler
        from cnllm.utils.scheduler.controller import AdaptiveController

        total = 30
        prompts = [f"r {i}" for i in range(total)]

        # 创建 scheduler，不传 controllers（空 dict）
        scheduler = BatchScheduler(
            client=self.client,
            max_concurrent=3,
            rps=2,
            timeout=None,
            max_retries=None,
            retry_delay=None,
            stop_on_error=False,
            controllers={},
        )

        # 获取 controller key 并用固定 mc=3 的控制器覆盖
        ck = scheduler._ctrl_key()
        if ck:
            ctrl = AdaptiveController()
            ctrl.mc = 3
            scheduler.controllers[ck] = ctrl

        from cnllm.core.accumulators.batch_accumulator import BatchResponse
        batch_response = BatchResponse()
        batch_response._total = total
        batch_response._start_time = time.time()
        scheduler._execute_batch_response = batch_response

        from cnllm.utils.scheduler.base import _normalize_batch_requests
        requests = _normalize_batch_requests(prompt=prompts)

        t0 = time.time()
        scheduler.execute(requests)
        elapsed = time.time() - t0

        ctrl = next(iter(scheduler.controllers.values())) if scheduler.controllers else None

        print(f"\n=== Fixed Baseline (mc=3, rps=2) ===")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Status: {batch_response.status}")
        print(f"  Controller mc: {ctrl.mc if ctrl else 'N/A'}")
        print(f"  Controller trace: {ctrl.trace if ctrl else 'N/A'}")
        print(f"  Errors: {batch_response.errors}")

        self.assertGreaterEqual(batch_response.status.get("success_count", 0), 1)

    def test_fixed_via_batch_api(self):
        """通过 batch() 接口，显式传 rps=2 max_concurrent=3"""
        total = 30
        prompts = [f"r {i}" for i in range(total)]

        t0 = time.time()
        resp = self.client.chat.batch(prompt=prompts, rps=2, max_concurrent=3)
        for _ in resp:
            pass
        elapsed = time.time() - t0

        print(f"\n=== Batch API (rps=2, mc=3) ===")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Status: {resp.status}")
        if hasattr(self.client, '_last_controllers') and self.client._last_controllers:
            for k, ctrl in self.client._last_controllers.items():
                print(f"  Controller: mc={ctrl.mc} last_safe={ctrl._last_safe} trace={ctrl.trace}")
        print(f"  Errors: {resp.errors}")

        self.assertGreaterEqual(resp.status.get("success_count", 0), 1)


    def test_rps1(self):
        """纯 RPS=1 限速，不触发 429，看是否 ~60s 跑完"""
        from cnllm.utils.scheduler.base import BatchScheduler
        from cnllm.utils.scheduler.controller import AdaptiveController

        total = 30
        prompts = [f"r {i}" for i in range(total)]

        scheduler = BatchScheduler(
            client=self.client,
            max_concurrent=1,
            rps=1,
            controllers={},
        )
        ck = scheduler._ctrl_key()
        if ck:
            ctrl = AdaptiveController()
            ctrl.mc = 1
            scheduler.controllers[ck] = ctrl

        from cnllm.core.accumulators.batch_accumulator import BatchResponse
        batch_response = BatchResponse()
        batch_response._total = total
        batch_response._start_time = time.time()
        scheduler._execute_batch_response = batch_response

        from cnllm.utils.scheduler.base import _normalize_batch_requests
        requests = _normalize_batch_requests(prompt=prompts)

        t0 = time.time()
        scheduler.execute(requests)
        elapsed = time.time() - t0

        print(f"\n=== RPS=1 Test ===")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Status: {batch_response.status}")
        print(f"  Errors: {batch_response.errors}")


if __name__ == "__main__":
    unittest.main()
