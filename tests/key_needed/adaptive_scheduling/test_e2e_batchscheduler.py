"""
BatchScheduler 综合 E2E 测试
测试无 fb / 有 fb 主优先 / 有 fb 池化 三种模式
v2: 新增 adaptive_scheduler_v2_results.md 记录完整学习状态
"""
import os, sys, time, unittest, datetime
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

PRI_KEY = os.environ.get("DEEPSEEK_API_KEY")
SEC_KEY = os.environ.get("MINIMAX_API_KEY")
THIRD_KEY = os.environ.get("BAIDU_API_KEY")
KIMI_KEY = os.environ.get("KIMI_API_KEY")
BAIDU_KEY = os.environ.get("BAIDU_API_KEY")
PRI_MODEL = "deepseek-v4-flash"
SEC_MODEL = "minimax-m2.1"
THIRD_MODEL = "ernie-lite-pro-128k"
KIMI_MODEL = "moonshot-v1-8k"
FALLBACK_CFG = {SEC_MODEL: {"api_key": SEC_KEY}, THIRD_MODEL: {"api_key": THIRD_KEY}}

BATCH_SIZE = 200
PROMPTS = ["hi" for _ in range(BATCH_SIZE)]

_RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'files', 'test_results.md')
_V2_RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'files', 'adaptive_scheduler_v2_results.md')


def _save_result(test_name, config, ok, total, elapsed, ctrls):
    """追加测试结果到记录文件（旧格式，保持兼容）"""
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    pri_ctrl = fb_ctrl = None
    for key, ctrl in ctrls.items():
        if key[0] == PRI_KEY or key[0] == KIMI_KEY:
            pri_ctrl = ctrl
        else:
            fb_ctrl = ctrl

    def _trace_str(c):
        if not c or not c.trace: return "-"
        parts = []
        t0 = c.trace[0].get('t', 0)
        for tr in c.trace:
            dt = tr.get('t', t0) - t0
            lat = tr.get('lat', 0)
            rpm_v = tr.get('rpm', 0)
            rpm_lim = tr.get('rpm_limit', '?')
            reason = tr.get('reason', '')
            parts.append(f"{dt:.0f}s mc={tr['mc']} rps={tr['rps']:.0f} lat={lat:.1f} rpm={rpm_v} lim={rpm_lim} {reason}")
        return " | ".join(parts)

    def _summary(c):
        if not c: return "-"
        a = c._lat_ewma if c._lat_ewma_init else 0
        return f"mc={c.mc} rps={c.rps} lat={a:.2f} n429={c._n_429} b={c._n_backoff} ok={c._total_ok} rpm_limit={c._rpm_limit}"

    pri_trace_line = _trace_str(pri_ctrl) if pri_ctrl else "-"
    fb_trace_line = _trace_str(fb_ctrl) if fb_ctrl else "-"
    pri_s = _summary(pri_ctrl)
    fb_s = _summary(fb_ctrl) if fb_ctrl else "-"

    line = f"{now} | {test_name} | {PRI_MODEL} | {config} | {ok}/{total} | {elapsed:.0f}s\n"
    line += f"  primary: {pri_s}\n"
    if fb_ctrl:
        line += f"  fallback: {fb_s}\n"
    line += f"  primary trace: {pri_trace_line}\n"
    if fb_ctrl:
        line += f"  fallback trace: {fb_trace_line}\n"
    try:
        with open(_RESULTS_FILE, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass


def _save_result_v2(test_name, config, ok, total, elapsed, ctrls, batch_start):
    """写入 comprehensive v2 测试结果到专门文件"""
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    lines = []
    lines.append(f"## {now} | {test_name} | {config}\n")
    lines.append(f"- **Result**: {ok}/{total} success, {elapsed:.0f}s, {ok/elapsed:.2f} req/s\n")
    for key, ctrl in ctrls.items():
        lat = ctrl._lat_ewma if ctrl._lat_ewma_init else 0
        clean_sz = len(ctrl._clean_window) if hasattr(ctrl, '_clean_window') else 'N/A'
        learned = ctrl._limit_learned if hasattr(ctrl, '_limit_learned') else 'N/A'
        consec = ctrl._consecutive_429 if hasattr(ctrl, '_consecutive_429') else 'N/A'
        if hasattr(ctrl, '_limit_learned'):
            lines.append(f"  **{key[1]}** (v2): mc={ctrl.mc} rpm_limit={ctrl._rpm_limit} "
                        f"learned={learned} clean_sz={clean_sz} consec={consec} "
                        f"n429={ctrl._n_429} ok={ctrl._total_ok} lat={lat:.2f}s\n")
        else:
            lines.append(f"  **{key[1]}** (legacy): mc={ctrl.mc} rpm_limit={ctrl._rpm_limit} "
                        f"n429={ctrl._n_429} ok={ctrl._total_ok} lat={lat:.2f}s\n")
        if ctrl.trace:
            lines.append("  ```\n")
            lines.append(f"  {'t(s)':>5} {'mc':>3} {'rps':>4} {'lat':>5} {'thr(rmp)':>9} {'lim':>5} {'event':<14}\n")
            lines.append(f"  {'-'*50}\n")
            t0 = ctrl.trace[0].get('t', batch_start)
            for tr in ctrl.trace:
                dt = tr.get('t', t0) - t0
                lat_v = tr.get('lat', 0)
                rpm_v = tr.get('rpm', 0)
                rpm_lim = tr.get('rpm_limit', '?')
                reason = tr.get('reason', '')
                lines.append(f"  {dt:>5.0f} {tr['mc']:>3} {tr['rps']:>4.0f} {lat_v:>5.1f} {rpm_v:>9} {str(rpm_lim):>5} {reason:<14}\n")
            lines.append("  ```\n")
    lines.append("\n")
    try:
        with open(_V2_RESULTS_FILE, 'a', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"  [WARN] v2 results write failed: {e}")


def _print_curve(client, label):
    """打印 controller 的完整曲线数据（v2 增强版）"""
    ctrls = getattr(client, '_last_controllers', None) or {}
    if not ctrls:
        print(f"  [{label}] NO CONTROLLERS")
        return
    for key, ctrl in ctrls.items():
        lat_ewma = ctrl._lat_ewma if ctrl._lat_ewma_init else 0
        n = 0
        avg_lat = lat_ewma
        tp = ctrl._rpm_limit / 60 if (hasattr(ctrl, '_limit_learned') and ctrl._limit_learned) else (ctrl.mc / avg_lat if avg_lat > 0 else 0)
        trace = list(ctrl.trace)
        print(f"\n  [{label}] key={key[1]} ({key[0][:8]}...)")
        mc_final = ctrl.mc
        print(f"    mc_final={mc_final} rps={ctrl.rps}")
        print(f"    rpm_limit={ctrl._rpm_limit} n429={ctrl._n_429} n_backoff={ctrl._n_backoff}")
        # v2 状态输出
        if hasattr(ctrl, '_limit_learned'):
            clean_sz = len(ctrl._clean_window)
            print(f"    ** v2 state: learned={ctrl._limit_learned} clean_window={clean_sz} consec_429={ctrl._consecutive_429}")
            if ctrl._limit_learned:
                print(f"    ** v2 locked: revalidate_counter={ctrl._revalidate_counter}")
        print(f"    total_ok={ctrl._total_ok} avg_lat={avg_lat:.2f}s throughput={tp:.2f} req/s")
        # 时间线表格（增加 rpm_limit 列）
        t0 = trace[0].get('t', 0) if trace else 0
        print(f"    {'t(s)':>5} {'mc':>3} {'rps':>4} {'lat':>5} {'thrpm':>6} {'cw':>4} {'limit':>5} {'learn':>5} {'event':<16}")
        print(f"    {'-'*61}")
        for tr in trace:
            dt = tr.get('t', t0) - t0
            lat_v = tr.get('lat', 0)
            rpm_v = tr.get('rpm', 0)
            cw_v = tr.get('cw', '')
            rpm_lim = tr.get('rpm_limit', '?')
            learned = tr.get('learned', '')
            reason = tr.get('reason', '')
            print(f"    {dt:>5.0f} {tr['mc']:>3} {tr['rps']:>4.0f} {lat_v:>5.1f} {rpm_v:>6} {str(cw_v):>4} {str(rpm_lim):>5} {str(learned):>5} {reason:<16}")

        print(f"    {'-'*56}")
        print(f"    total_ok={ctrl._total_ok} avg_lat={avg_lat:.2f}s rpm_limit={ctrl._rpm_limit} n429={ctrl._n_429}")


@unittest.skipUnless(PRI_KEY, "need DEEPSEEK_API_KEY")
class TestNoFallback(unittest.TestCase):
    """单模型，无 fallback"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=SEC_KEY, model=SEC_MODEL)

    def test_single_model(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS[:50], keep=["errors"])
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  无 fallback | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "single")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("no-fb", "single", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("no-fb", "single", ok, s.get('total', 0), elapsed, ctrls, t0)
        self.assertGreater(ok, 0)


@unittest.skipUnless(KIMI_KEY, "need KIMI_API_KEY")
class TestNoFallbackMoonshot(unittest.TestCase):
    """单模型 moonshot-v1-8k，无 fallback"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=KIMI_KEY, model=KIMI_MODEL)

    def test_single_model_moonshot(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS[:50], keep=["errors"])
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  moonshot 无 fb | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "moonshot")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("no-fb-moonshot", "single-moonshot", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("no-fb-moonshot", "single-moonshot", ok, s.get('total', 0), elapsed, ctrls, t0)
        self.assertGreater(ok, 0)


@unittest.skipUnless(PRI_KEY, "need DEEPSEEK_API_KEY")
class TestNoFallbackDeepseek(unittest.TestCase):
    """单模型 deepseek-v4-flash"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL)

    def test_single_model_deepseek(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS[:50], keep=["errors"])
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  deepseek 无 fb | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "deepseek")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("no-fb-deepseek", "single-deepseek", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("no-fb-deepseek", "single-deepseek", ok, s.get('total', 0), elapsed, ctrls, t0)
        self.assertGreater(ok, 0)


@unittest.skipUnless(BAIDU_KEY, "need BAIDU_API_KEY")
class TestNoFallbackErnie(unittest.TestCase):
    """单模型 ernie-lite-pro-128k"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=BAIDU_KEY, model="ernie-lite-pro-128k")

    def test_single_model_ernie(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS[:50], keep=["errors"])
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  ernie 无 fb | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "ernie")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("no-fb-ernie", "single-ernie", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("no-fb-ernie", "single-ernie", ok, s.get('total', 0), elapsed, ctrls, t0)
        self.assertGreater(ok, 0)


@unittest.skipUnless(PRI_KEY and SEC_KEY, "need both keys")
class TestFallbackPriority(unittest.TestCase):
    """有 fallback + performance=False（主模型优先）"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL,
                           fallback_models=FALLBACK_CFG)

    def test_priority_mode(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS, keep=["errors"],
                                       performance=False)
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  主模型优先 | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "all")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("priority", "fb+priority", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("priority", "fb+priority", ok, s.get('total', 0), elapsed, ctrls, t0)
        self.assertGreater(ok, 0)


@unittest.skipUnless(PRI_KEY and SEC_KEY, "need both keys")
class TestFallbackPooled(unittest.TestCase):
    """有 fallback + performance=True（池化分发）"""

    @classmethod
    def setUpClass(cls):
        from cnllm import CNLLM
        cls.client = CNLLM(api_key=PRI_KEY, model=PRI_MODEL,
                           fallback_models=FALLBACK_CFG)

    def test_pooled_mode(self):
        t0 = time.time()
        resp = self.client.chat.batch(prompt=PROMPTS, keep=["errors"],
                                       performance=True)
        for _ in resp:
            pass
        elapsed = time.time() - t0
        s = resp.status
        ok = s.get("success_count", 0)
        print(f"\n{'='*60}")
        print(f"  池化分发   | {ok}/{s.get('total')} 成功 "
              f"| {elapsed:.0f}s | 吞吐: {ok/elapsed:.2f} req/s")
        print(f"{'='*60}")
        _print_curve(self.client, "all")
        ctrls = getattr(self.client, '_last_controllers', None) or {}
        _save_result("pooled", "fb+pooled", ok, s.get('total', 0), elapsed, ctrls)
        _save_result_v2("pooled", "fb+pooled", ok, s.get('total', 0), elapsed, ctrls, t0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
 