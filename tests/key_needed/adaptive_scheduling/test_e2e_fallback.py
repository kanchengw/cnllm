"""E2E tests: normal/heterogeneous batch x single ctrl/ctrl+fallback"""
import os, sys, time, unittest
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

PRI_KEY = os.environ.get("DEEPSEEK_API_KEY")
SEC_KEY = os.environ.get("MINIMAX_API_KEY")
PRI_MODEL = "deepseek-v4-flash"
SEC_MODEL = "minimax-m2.1"
FALLBACK_CFG = {SEC_MODEL: {"api_key": SEC_KEY}}
BATCH_SIZE = 50
PROMPTS = [f"Reply in {i} words: hello world" for i in range(1, BATCH_SIZE + 1)]


def _print_ctrl(client, label):
    ctrls = getattr(client, '_last_controllers', None) or {}
    if not ctrls:
        print(f"  [{label}] NO CONTROLLERS")
        return
    for key, ctrl in ctrls.items():
        lat = ctrl._lat_ewma if ctrl._lat_ewma_init else 0
        print(f"  [{label}] key={key} mc={ctrl.mc} rps={ctrl.rps} "
              f"rpm={ctrl._rpm_limit} "
              f"n429={ctrl._n_429} "
              f"ok={ctrl._total_ok} lat={lat:.2f}s")


@unittest.skipUnless(PRI_KEY, "need DEEPSEEK_API_KEY")
class TestNormalBatchSingleCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL)

    def test_normal(self):
        resp = self.client.chat.batch(prompt=PROMPTS, keep=["errors"])
        for _ in resp:
            pass
        s = resp.status
        e = resp.errors
        print(f"\n=== 普通batch/单ctrl ===")
        print(f"  Status: {s}  Errors: {len(e)}")
        for rid, err in list(e.items())[:3]:
            print(f"    {rid}: {err}")
        _print_ctrl(self.client, "ctrl")
        self.assertGreater(s.get("success_count", 0), 0)


@unittest.skipUnless(PRI_KEY and SEC_KEY, "need both keys")
class TestNormalBatchWithFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL,
                           fallback_models=FALLBACK_CFG)

    def test_normal(self):
        resp = self.client.chat.batch(prompt=PROMPTS, keep=["errors"])
        for _ in resp:
            pass
        s = resp.status
        e = resp.errors
        print(f"\n=== 普通batch+fb ===")
        print(f"  Status: {s}  Errors: {len(e)}")
        for rid, err in list(e.items())[:3]:
            print(f"    {rid}: {err}")
        _print_ctrl(self.client, "all")
        self.assertGreater(s.get("success_count", 0), 0)


@unittest.skipUnless(PRI_KEY, "need DEEPSEEK_API_KEY")
class TestHeterogeneousBatchSingleCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL)

    def test_hetero(self):
        requests = [{"prompt": p, "model": PRI_MODEL, "api_key": PRI_KEY} for p in PROMPTS]
        resp = self.client.chat.batch(requests=requests, keep=["errors"])
        for _ in resp:
            pass
        s = resp.status
        e = resp.errors
        print(f"\n=== 异构batch/单ctrl ===")
        print(f"  Status: {s}  Errors: {len(e)}")
        for rid, err in list(e.items())[:3]:
            print(f"    {rid}: {err}")
        _print_ctrl(self.client, "ctrl")
        self.assertGreater(s.get("success_count", 0), 0)


@unittest.skipUnless(PRI_KEY and SEC_KEY, "need both keys")
class TestHeterogeneousBatchWithFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL)

    def test_hetero(self):
        # 50 个请求全有效主模型，高并发测试 fallback 分流
        N = 50
        prompts = [f"Reply in {i} words: hello world" for i in range(1, N + 1)]
        requests = [{"prompt": p, "model": PRI_MODEL, "api_key": PRI_KEY,
                     "fallback_models": {SEC_MODEL: {"api_key": SEC_KEY}}}
                    for p in prompts]
        resp = self.client.chat.batch(requests=requests, keep=["errors"])
        for _ in resp:
            pass
        s = resp.status
        e = resp.errors
        print(f"\n=== 异构batch+fb ===")
        print(f"  Status: {s}  Errors: {len(e)}")
        for rid, err in list(e.items())[:3]:
            print(f"    {rid}: {err}")
        _print_ctrl(self.client, "all")
        self.assertGreater(s.get("success_count", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
