import os, sys, time, threading, unittest
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
API_KEY = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MODEL = "minimax-m2.1"

@unittest.skipUnless(API_KEY, "need API Key")
class TestAdaptiveControllerSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_sync_adaptive(self):
        from cnllm.utils.scheduler.controller import AdaptiveController
        ctrl = AdaptiveController()
        lock = threading.Lock()
        active = 0
        results = []

        def req(prompt):
            nonlocal active
            while True:
                with lock:
                    if not ctrl.rate_limited and active < ctrl.mc:
                        active += 1
                        break
                time.sleep(0.02)
            t0 = time.time()
            try:
                self.client.chat.create(messages=[{"role":"user","content":prompt}])
                el = time.time() - t0
                with lock:
                    active -= 1
                    ctrl.on_complete(el, 200)
                    results.append(("ok", el, ctrl.mc))
            except Exception as e:
                el = time.time() - t0
                with lock:
                    active -= 1
                    ctrl.on_complete(el, 429)
                    results.append(("429", el, ctrl.mc))

        total = 50
        prompts = [f"r {i}" for i in range(total)]
        threads = []
        for p in prompts:
            t = threading.Thread(target=req, args=(p,))
            threads.append(t)
            t.start()
            time.sleep(1.0 / max(ctrl.rps, 1))
        for t in threads:
            t.join()
        print(f"\n=== Sync Adaptive ===")
        print(f"mc={ctrl.mc} rps={ctrl.rps} ")
        if ctrl.trace:
            print(f"Trace: {ctrl.trace}")
        succ = sum(1 for r in results if r[0]=="ok")
        print(f"Success: {succ}/{total}")
        self.assertGreater(succ, 0)
        self.assertGreaterEqual(ctrl.mc, 2)

    def test_sync_batch_full_chain(self):
        """同步 client.chat.batch() → BatchScheduler → controller 完整链路"""
        import time
        total = 50
        prompts = [f"r {i}" for i in range(total)]
        t0 = time.time()
        resp = self.client.chat.batch(prompt=prompts)
        for _ in resp:
            pass
        elapsed = time.time() - t0
        print(f"\n=== Sync Batch Full Chain ===")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Status: {resp.status}")
        print(f"  Errors: {resp.errors}")
        if hasattr(self.client, '_last_controllers') and self.client._last_controllers:
            for k, ctrl in self.client._last_controllers.items():
                print(f"  Controller: mc={ctrl.mc} rps={ctrl.rps}")
                if ctrl.trace:
                    print(f"  Trace: {ctrl.trace}")
        else:
            print("  NO CONTROLLER DATA")
        print(f"  Errors: {resp.errors}")
        if hasattr(self.client, '_last_controllers') and self.client._last_controllers:
            for k, ctrl in self.client._last_controllers.items():
                lat = ctrl._lat_ewma if ctrl._lat_ewma_init else 0
                if lats:
                    print(f"  Latency: ewma={lat:.2f}s")
                    print(f"  Backoff: {ctrl._n_backoff} times, 429: {ctrl._n_429} times")
        self.assertGreaterEqual(resp.status.get("success_count", 0), 1)

