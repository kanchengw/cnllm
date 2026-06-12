"""
自适应控制器集成测试 — 用 minimax-m2.1 验证 MC/RPS 动态变化

通过活跃请求计数器真正限制并发，429 不重试，冻结期重试。
需要有效 API Key。
"""
import os
import sys
import time
import threading
import unittest
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

API_KEY = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MODEL = "minimax-m2.1"


@unittest.skipUnless(API_KEY, "need API Key")
class TestAdaptiveController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

    def test_adaptive_control_curve(self):
        from cnllm.utils.scheduler.controller import AdaptiveController

        controller = AdaptiveController()
        lock = threading.Lock()
        active = 0
        trace = []
        t0 = time.time()

        def log(event, status):
            nonlocal active
            trace.append({
                "t": round(time.time() - t0, 2),
                "mc": controller.mc,
                "rps": controller.rps,
                "rate_limited": controller.rate_limited,
                "active": active,
                "event": event,
                "status": status,
            })

        def send_request(prompt):
            nonlocal active
            while True:
                with lock:
                    if active < controller.mc:
                        active += 1
                        break
                time.sleep(0.02)

            while True:
                start = time.time()
                try:
                    self.client.chat.create(messages=[
                        {"role": "user", "content": prompt}
                    ])
                    elapsed = time.time() - start
                    with lock:
                        active -= 1
                        controller.on_complete(elapsed, 200)
                        log("ok", 200)
                    break
                except Exception as e:
                    elapsed = time.time() - start
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        with lock:
                            controller.on_complete(elapsed, 429)
                            log("429", 429)
                        while controller.rate_limited:
                            time.sleep(0.1)
                    else:
                        with lock:
                            active -= 1
                            log("err", 500)
                        break

        total = 30
        prompts = [f"Reply in {i} words: hello" for i in range(1, total + 1)]
        threads = []

        for p in prompts:
            t = threading.Thread(target=send_request, args=(p,))
            threads.append(t)
            t.start()
            time.sleep(1.0 / max(controller.rps, 1))

        for t in threads:
            t.join()

        print("\n=== MC/RPS Change Curve ===")
        print(f"{'t(s)':>5} {'event':>6} {'mc':>4} {'rps':>4} {'safe':>4} {'frozen':>6} {'act':>3}")
        print("-" * 55)
        prev = None
        for t in trace:
            key = (t["mc"], t["rps"], t["frozen"])
            if prev != key:
                print(f"{t['t']:>5.1f} {t['event']:>6} {t['mc']:>4} {t['rps']:>4} {t['last_safe']:>4} {str(t['frozen']):>6} {t['active']:>3}")
                prev = key

        print("\n=== CSV ===")
        print("t,event,mc,rps,last_safe,frozen,active")
        for t in trace:
            print(f"{t['t']},{t['event']},{t['mc']},{t['rps']},{t['last_safe']},{t['frozen']},{t['active']}")

        print(f"\n  Total: {len(trace)} events, Final: mc={controller.mc}, rps={controller.rps}, ")
        self.assertGreaterEqual(controller.mc, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
