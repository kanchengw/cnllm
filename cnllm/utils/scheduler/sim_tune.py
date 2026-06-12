"""Pooled dispatch simulator with virtual clock — supports rpm estimation,
growth strategy, probe cleanup, and cooldown reset optimizations"""
import random
from collections import deque, defaultdict

SIM_TIME = 0.0
def now(): global SIM_TIME; return SIM_TIME
def adv(dt): global SIM_TIME; SIM_TIME += dt

class SimAPI:
    """模拟 API：10s 滑动窗口 + burst 容差"""
    def __init__(self, rpm_limit, burst_tol=1.3):
        self.rpm_limit = rpm_limit
        self.burst_tol = burst_tol
        self._recent = deque()
    def call(self, name, lat):
        n = now()
        cutoff = n - 10
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        rc = len(self._recent)
        rpm = rc / max(self._recent[-1][0] - self._recent[0][0], 0.001) * 60 if rc >= 2 else 0
        if rpm > self.rpm_limit * self.burst_tol and rc > 2:
            el = 0.1 + random.random() * 0.1
            adv(el); return False, el
        el = lat * (0.8 + random.random() * 0.4)
        adv(el)
        self._recent.append((now(), name))
        return True, max(0.01, el)


# ============ 配置参数 ============
CFG = {
    # ---- 增速 ----
    "rpm_grow_mode": "multiplicative",   # multiplicative(×1.5) / additive(+N)
    "rpm_grow_mul": 1.5,                 # 乘性因子
    "rpm_grow_add": 5,                   # 加性增量
    "batch_threshold": 200,              # <= 此值算小批量
    # ---- 429 学习 ----
    "rpm_est_mode": "halt",              # half: _rpm_smooth*=0.5; current: 直接用 cur_rpm
    "rpm_init_est": 60,                  # 无测量时默认 60
    "rpm_floor": 10,                     # 限流估计下限
    # ---- 探针 ----
    "probe_no_reset": True,             # True: 不重置 _next_at
    "probe_reset_cooldown": True,       # True: 探针成功重置 cooldown=2.0
    "probe_decay_n429": True,           # True: 探针成功 _n_429 = max(0, _n_429 - 1)
    # ---- mc 增长 ----
    "mc_grow_ok": 10, "mc_grow_factor": 5,
    # ---- AIMD ----
    "aimd_bn": 1, "aimd_bd": 2,
    # ---- 退避 ----
    "cooldown_init": 2.0, "cooldown_max": 32.0,
    # ---- 初始 ----
    "mc_init": 1,
    # ---- 其他 ----
    "hard_gate": True, "dead_fb": True,
    "weight_floor": 0.3, "init_weight": 1.0,
}

MODELS = [{"n": "moonshot", "l": 0.70}, {"n": "mimo", "l": 0.55}, {"n": "minimax", "l": 3.0}]
BATCH = 200
BURST = 1.3


class SimCtrl:
    def __init__(self, name, cfg):
        self.name = name; self.cfg = cfg
        self.mc = cfg["mc_init"]
        self._tp_ewma = 0.0; self._tp_init = False
        self._rpm_limit: int | None = None
        self._next_at = 0.0
        self._frozen = False; self._n_429 = 0
        self._cooldown = cfg["cooldown_init"]
        self._ok_since_grow = 0
        self._ct = deque()
        self._rpm_smooth = 0.0; self._total_ok = 0
        self._probe = False; self._next_probe_at = 0.0
        self._lat_ewma = 0.0; self._lat_init = False
        self.trace = [{"mc": self.mc, "t": 0}]; self._tmc = self.mc

    @property
    def can_accept(self):
        if not self.cfg["hard_gate"]: return not self._frozen
        return not self._frozen and now() >= self._next_at

    @property
    def cur_rpm(self):
        n = now(); cutoff = n - 60
        while self._ct and self._ct[0] < cutoff: self._ct.popleft()
        if len(self._ct) < 2: return 0
        s = self._ct[-1] - self._ct[0]
        return int(len(self._ct) / s * 60) if s > 0 else 0

    def _st(self, reason=""):
        if self.mc != self._tmc:
            self.trace.append({"mc": self.mc, "t": now(), "r": reason})
            self._tmc = self.mc

    def to_probe(self):
        return self._frozen and not self._probe and now() >= self._next_probe_at

    def start_probe(self): self._probe = True

    # ---------- rpm_limit 上探 ----------
    def _grow_rpm_limit(self):
        mode = self.cfg["rpm_grow_mode"]
        cur = self._rpm_limit or 30
        if mode == "additive":
            return cur + self.cfg["rpm_grow_add"]
        else:  # multiplicative
            return max(int(cur * self.cfg["rpm_grow_mul"]), cur + 10)

    # ---------- rpm_limit 学习（429 时） ----------
    def _estimate_rpm_limit(self):
        mode = self.cfg["rpm_est_mode"]
        if mode == "current":
            raw = self.cur_rpm
        else:  # "half"
            self._rpm_smooth *= 0.5
            raw = self._rpm_smooth if self._rpm_smooth > 0 else self.cur_rpm
        return max(self.cfg["rpm_floor"], int(raw)) if raw > 0 else self.cfg["rpm_init_est"]

    # ---------- 自节奏 _next_at ----------
    def _pace_next_at(self, elapsed):
        r = self._rpm_limit
        if r is not None and r < 100000:
            self._next_at = now() + max(0.01, 60.0 / r * self.mc - elapsed)
        else:
            self._next_at = now() + 0.01

    # ---------- on_complete ----------
    def on_complete(self, elapsed, sc, ra=0.0):
        n = now()

        # --- 探针路径 ---
        if self._probe:
            self._probe = False
            if sc == 429:
                self._next_probe_at = n + self._cooldown
                self._cooldown = min(self.cfg["cooldown_max"], self._cooldown * 2)
                return "STILL_LIMITED"
            self._frozen = False
            self.mc = max(1, self.mc * self.cfg["aimd_bn"] // self.cfg["aimd_bd"])
            self._ok_since_grow = 0
            # 不重置 _next_at（staging retry 已设好）
            if self.cfg["probe_reset_cooldown"]:
                self._cooldown = self.cfg["cooldown_init"]
            if self.cfg["probe_decay_n429"]:
                self._n_429 = max(0, self._n_429 - 1)
            if sc == 200:
                self._total_ok += 1
            self._ct.append(n)
            self._st("thaw")
            return "PROBE_OK"

        # --- 冻结状态 ---
        if self._frozen:
            if sc == 429:
                return "STILL_LIMITED"
            self._ct.append(n)
            self._total_ok += 1
            self._pace_next_at(elapsed)  # 必须更新闸门
            return None

        # --- 429 处理 ---
        if sc == 429:
            self._n_429 += 1
            est = self._estimate_rpm_limit()
            self._rpm_limit = est if self._rpm_limit is None else min(self._rpm_limit, est)
            self._frozen = True
            delay = ra if ra > 0 else self._cooldown
            self._next_at = n + delay
            self._cooldown = min(self.cfg["cooldown_max"], self._cooldown * 2)
            self._next_probe_at = self._next_at
            self._st("429")
            return "RATE_LIMITED"

        # --- 200 ---
        self._ct.append(n)
        self._total_ok += 1
        self._ok_since_grow += 1

        if not self._lat_init:
            self._lat_ewma = elapsed; self._lat_init = True
        else:
            self._lat_ewma = 0.2 * elapsed + 0.8 * self._lat_ewma

        tp = 1.0 / max(self._lat_ewma, 0.001)
        if not self._tp_init:
            self._tp_ewma = tp; self._tp_init = True
        else:
            self._tp_ewma = 0.3 * tp + 0.7 * self._tp_ewma

        raw = self.cur_rpm
        if self._rpm_smooth == 0:
            self._rpm_smooth = float(raw) if raw > 0 else 60.0
        else:
            self._rpm_smooth = 0.3 * raw + 0.7 * self._rpm_smooth

        self._pace_next_at(elapsed)

        # mc 增长：无 429 或衰减恢复后可增长
        can_grow = self._n_429 == 0
        if can_grow:
            th = max(self.cfg["mc_grow_ok"], self.mc * self.cfg["mc_grow_factor"])
            if self._ok_since_grow >= th:
                self.mc += 1
                self._ok_since_grow = 0
                self._st("grow")

        # rpm_limit 上探
        can_up = self._n_429 == 0 and self._total_ok > 0 and self._total_ok % 5 == 0
        if can_up:
            self._rpm_limit = self._grow_rpm_limit()
            self._st("rpm_up")

        return None

    def weight(self):
        if not self._tp_init:
            return self.cfg["init_weight"]
        return max(self.cfg["weight_floor"], self._tp_ewma)


class SimSched:
    def __init__(self, ctrls, cfg, apis):
        self.ctrls = ctrls; self.cfg = cfg; self.apis = apis
        self._inf = defaultdict(int); self._sel = defaultdict(int)

    def _pick(self):
        n = now(); cand = []
        for c in self.ctrls:
            if not c.can_accept:
                continue
            if self._inf[c.name] >= c.mc:
                continue
            cand.append((c, c.weight()))
        if cand:
            total = sum(w for _, w in cand)
            r = random.random() * total
            cum = 0
            for c, w in cand:
                cum += w
                if r <= cum:
                    self._sel[c.name] += 1
                    self._inf[c.name] += 1
                    return c
            c = cand[-1][0]
            self._sel[c.name] += 1
            self._inf[c.name] += 1
            return c
        if self.cfg["dead_fb"]:
            soon, soon_at = None, float("inf")
            for c in self.ctrls:
                if c._frozen: continue
                if self._inf[c.name] >= c.mc: continue
                if c._next_at < soon_at:
                    soon_at = c._next_at
                    soon = c
            if soon is not None:
                w = soon_at - n
                if w > 0: adv(w)
                self._sel[soon.name] += 1
                self._inf[soon.name] += 1
                return soon
        return None

    def _exec(self):
        sel = self._pick()
        if sel is None:
            return None, "all_frozen"
        idx = [c.name for c in self.ctrls].index(sel.name)
        ok, el = self.apis[sel.name].call(sel.name, MODELS[idx]["l"])
        if ok:
            sel.on_complete(el, 200)
            self._inf[sel.name] = max(0, self._inf[sel.name] - 1)
            return sel.name, "ok"
        sel.on_complete(el, 429)
        self._inf[sel.name] = max(0, self._inf[sel.name] - 1)
        for c in self.ctrls:
            if c is sel or c._frozen: continue
            if self._inf[c.name] >= c.mc: continue
            fidx = [x.name for x in self.ctrls].index(c.name)
            self._inf[c.name] += 1
            ok2, el2 = self.apis[c.name].call(c.name, MODELS[fidx]["l"])
            self._inf[c.name] = max(0, self._inf[c.name] - 1)
            if ok2:
                c.on_complete(el2, 200)
                return c.name, "fb_ok"
            else:
                c.on_complete(el2, 429)
        return sel.name, "all_fail"

    def run(self, n):
        res = []
        sub = 0
        while sub < n:
            for c in self.ctrls:
                if c.to_probe():
                    c.start_probe()
                    ok, _ = self.apis[c.name].call(c.name, 0.1)
                    c.on_complete(now(), 200 if ok else 429)
            act = [c for c in self.ctrls if not c._frozen]
            mx = max(1, sum(c.mc for c in act))
            while sub < n and sum(self._inf.values()) < mx:
                name, status = self._exec()
                if name is None:
                    earlys = [c._next_at for c in self.ctrls if c._frozen]
                    if earlys:
                        early = min(earlys)
                        if early > now():
                            adv(early - now())
                    break
                res.append({"name": name, "status": status})
                sub += 1
            if sub < n:
                adv(0.05)
        return res


def evaluate(cfg, api_rpm, batch=BATCH, seed=42):
    global SIM_TIME
    SIM_TIME = 0.0
    random.seed(seed)
    # Apply batch-aware growth mode
    effective_cfg = dict(cfg)
    if batch <= cfg.get("batch_threshold", 200):
        effective_cfg["rpm_grow_mode"] = "additive"
    apis = {m["n"]: SimAPI(api_rpm, BURST) for m in MODELS}
    ctrls = [SimCtrl(m["n"], effective_cfg) for m in MODELS]
    s = SimSched(ctrls, effective_cfg, apis)
    res = s.run(batch)
    elapsed = now()
    ok = sum(1 for r in res if r["status"] in ("ok", "fb_ok"))
    return {
        "elapsed": elapsed,
        "tput": batch / elapsed if elapsed > 0 else 0,
        "ok": ok,
        "n429": sum(c._n_429 for c in ctrls),
        "sel": dict(s._sel),
        "mc": {c.name: c.mc for c in ctrls},
        "rpm": {c.name: c._rpm_limit for c in ctrls},
        "ok_cnt": {c.name: c._total_ok for c in ctrls},
    }


def sweep(api_rpm, batch=BATCH, label=""):
    base = dict(CFG)
    variants = [
        ("baseline_mul", {"rpm_grow_mode": "multiplicative"}),
        ("baseline_add", {"rpm_grow_mode": "additive"}),
        ("est_cur_rpm.mul", {"rpm_est_mode": "current", "rpm_grow_mode": "multiplicative"}),
        ("est_cur_rpm.add", {"rpm_est_mode": "current", "rpm_grow_mode": "additive"}),
        ("est_cur+decay_n429.mul", {"rpm_est_mode": "current", "probe_decay_n429": True,
                                     "rpm_grow_mode": "multiplicative"}),
        ("est_cur+decay_n429.add", {"rpm_est_mode": "current", "probe_decay_n429": True,
                                     "rpm_grow_mode": "additive"}),
        ("full_opt_mul", {"rpm_est_mode": "current", "probe_decay_n429": True,
                          "probe_reset_cooldown": True,
                          "rpm_grow_mode": "multiplicative"}),
        ("full_opt_add", {"rpm_est_mode": "current", "probe_decay_n429": True,
                          "probe_reset_cooldown": True,
                          "rpm_grow_mode": "additive"}),
    ]
    print(f"\n{'='*110}")
    print(f"API RPM={api_rpm}, Batch={batch}")
    print(f"{'Config':<45} {'Tput':>7} {'429s':>5} {'mc':>20} {'final_rpm':>25}")
    print("-" * 110)
    best = None
    for name, ov in variants:
        c = {**base, **ov}
        r = evaluate(c, api_rpm, batch)
        mc_s = " ".join(f"{k}={v}" for k, v in r["mc"].items())
        rpm_s = " ".join(f"{k}:{v}" for k, v in sorted(r["rpm"].items()))
        print(f"{name:<45} {r['tput']:>6.2f}/s {r['n429']:>5}  {mc_s:<20} {rpm_s:<25}")
        if best is None or (r["n429"] <= 3 and r["tput"] > best["tput"]):
            best = r
    if best:
        print(f"\n  Best (<=3 429s): {best['tput']:.2f}/s, 429s={best['n429']}")
    return best


if __name__ == "__main__":
    print(f"Models: {[(m['n'], m['l']) for m in MODELS]}")
    print(f"\n--- 严格 API (30 RPM), 小批量 200 ---")
    best200 = sweep(30, 200)
    print(f"\n--- 严格 API (30 RPM), 大批量 2000 ---")
    best2000 = sweep(30, 2000)
    print(f"\n--- 宽松 API (120 RPM), 小批量 200 ---")
    best120 = sweep(120, 200)
    print(f"\n--- 宽松 API (120 RPM), 大批量 2000 ---")
    best120_2000 = sweep(120, 2000)
