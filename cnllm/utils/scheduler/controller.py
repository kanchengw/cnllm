import time
from collections import deque
from typing import Optional
import threading

MC_INIT = 1
COOLDOWN_INIT = 2.0
COOLDOWN_MAX = 32.0


class AdaptiveController:
    """简化 AdaptiveController — 自节奏 + 限流学习"""

    def __init__(self):
        self.mc = MC_INIT
        self._tp_ewma = 0.0
        self._tp_init = False
        self._rpm_limit: Optional[int] = 10  # 起步 10 RPM，低速友好，乘性增长不差这起步
        self._next_at = 0.0
        self._rate_limited = False
        self._n_429 = 0
        self._cooldown = COOLDOWN_INIT
        self._ok_since_grow = 0
        self._total_ok = 0
        self._n_backoff = 0
        self.trace = []
        self._trace_mc = MC_INIT
        self._lat_ewma = 0.0
        self._lat_ewma_init = False
        self._queue = deque()
        self._dispatch_times = deque()
        # RFC 6298 RTT 估计
        self._srtt = 0.0
        self._rttvar = 0.0
        self._srtt_init = False
        # === v2: 学习机制状态变量 ===
        self._clean_window: deque = deque()
        self._limit_learned = False
        self._revalidate_counter = 0
        self._consecutive_429 = 0
        self._consecutive_concurrency = 0    # 连续并发429计数（无成功间隔则递增）
        self._concurrency_handled = False     # 本轮 inflight 已处理过并发降级
        self._force_rpm_learn = False        # cap=1 fallthrough 覆盖 retry 跳过学习
        self._start_time = time.time()
        self._learned_rpm_limit = 0       # 锁存阶段保存的上限（不因二次429降低）
        self._last_429_ts = 0.0            # 最近一次429时间戳
        self._concurrency_cap = 100000         # 并发上限（concurrency 429 后降低）
        self._stable_counter = 0            # 连续以≥85%容量运行的计数
        self.trace.append({"mc": self.mc, "rps": self.rps, "rpm_limit": 10, "cw": 0, "t": time.time()})

    @property
    def rate_limited(self):
        return self._rate_limited

    @property
    def can_accept(self):
        return time.time() >= self._next_at

    @property
    def queue_depth(self):
        return len(self._queue)

    @property
    def consumption_rate(self):
        cutoff = time.time() - 10
        while self._dispatch_times and self._dispatch_times[0] < cutoff:
            self._dispatch_times.popleft()
        if len(self._dispatch_times) < 2:
            return 0.2
        span = self._dispatch_times[-1] - self._dispatch_times[0]
        return len(self._dispatch_times) / span if span > 0 else 0.2

    def _unfreeze(self):
        """自动解冻：窗口排到 rpm_limit 以下时恢复，无需探针"""
        if not self._rate_limited:
            return False
        if time.time() < self._next_at:
            return False
        self._prune_clean_window()
        live_cw = len(self._clean_window) - getattr(self, '_frozen_added', 0)
        if live_cw >= self._rpm_limit:
            return False
        self._rate_limited = False
        self._ok_since_grow = 0
        if self._limit_learned and self._learned_rpm_limit > 0:
            self._rpm_limit = self._learned_rpm_limit
        self._unfrozen_at = time.time()
        self._save_trace("unfrozen")
        return True

    def push(self, request):
        self._queue.append(request)

    def pop(self):
        return self._queue.popleft() if self._queue else None

    @property
    def rps(self):
        return max(1, round(self.mc / max(self._lat_ewma, 0.001))) if self._lat_ewma_init else 1

    @property
    def current_rpm(self) -> int:
        """从 _clean_window 计算实际吞吐（仅真实用户请求，不含探针/冻结）"""
        self._prune_clean_window()
        if not self._clean_window:
            return 0
        span = self._clean_window[-1] - self._clean_window[0]
        count = len(self._clean_window)
        if span < 60 and span > 0:
            return int(count / span * 60)
        return count

    def _save_trace(self, reason="grow"):
        if self.mc != self._trace_mc or reason in ("rpm_up", "rpm_down", "rpm_hold", "peak", "429", "relearn_429", "latency_warn", "learn_429", "early_429", "force_learn", "grow", "unfrozen", "429_locked", "concurrency_429", "concurrency_cap_1", "concurrency_lag", "ceiling_up", "revalidate"):
            self.trace.append({"mc": self.mc, "rps": self.rps, "reason": reason,
                               "lat": self._lat_ewma, "rpm": self.current_rpm,
                               "cw": len(self._clean_window),
                               "rpm_limit": self._rpm_limit, "t": time.time()})
            self._trace_mc = self.mc

    def _prune_clean_window(self):
        """清理 _clean_window 中超过 60 秒的过期时间戳"""
        cutoff = time.time() - 60
        while self._clean_window and self._clean_window[0] < cutoff:
            self._clean_window.popleft()

    def on_complete(self, elapsed, status_code, retry_after=0.0, rate_type="unknown", retry=False):
        # 已冻结状态下的 429（仍在连续撞墙，计数429）
        if self._rate_limited:
            if status_code == 429:
                self._consecutive_429 += 1
                self._clean_window.append(time.time())
                self._prune_clean_window()
                self._frozen_added = getattr(self, '_frozen_added', 0) + 1
                return None
            self._total_ok += 1
            if self._rpm_limit is not None and self._rpm_limit < 100000:
                self._next_at = time.time() + max(0.01, 60.0 / self._rpm_limit * self.mc - elapsed)
            else:
                self._next_at = time.time() + 0.01
            return None

        # 429 — v2: 学习机制
        if status_code == 429:
            self._n_429 += 1
            # rate_type 由调用方传入（queued_pool/_execute_single）
            
            # 判断并发：retry_after < 2s 或 rate_type 明确标记
            is_concurrency = (retry_after >= 0 and retry_after < 2.0) or rate_type == "concurrency"

            # 锁存阶段且 mc=1 已无可降并发 → 强制走 RPM（二次撞墙只能是 RPM）
            if is_concurrency and self._limit_learned and self._concurrency_cap == 1:
                is_concurrency = False

            if is_concurrency:
                # ── 并发 429 ──
                if not self._concurrency_handled or time.time() >= self._next_at:
                    # 新批次（首次或退避到期），重置保护标志
                    self._concurrency_handled = True
                    self.mc = max(1, self.mc - max(1, self.mc // 5))
                    self._concurrency_cap = min(self._concurrency_cap, self.mc)
                    self._consecutive_concurrency += 1
                    # 首次降级：设退避（后续残留请求已结束，设退避无意义）
                    delay = retry_after if retry_after > 0 else max(0.5, elapsed * 2)
                    self._next_at = time.time() + delay
                    self._clean_window.append(time.time())
                    self._prune_clean_window()

                    if self._concurrency_cap == 1 and self.mc == 1:
                        self._save_trace("concurrency_cap_1")
                        if self._consecutive_concurrency >= 3:
                            self._force_rpm_learn = True
                        else:
                            return "CONCURRENCY_LIMITED"
                    else:
                        self._save_trace("concurrency_429")
                        return "CONCURRENCY_LIMITED"
                else:
                    # 残留 inflight
                    self._clean_window.append(time.time())
                    self._prune_clean_window()
                    if self._concurrency_cap == 1 and self.mc == 1:
                        self._consecutive_concurrency += 1
                        self._save_trace("concurrency_cap_1")
                        if self._consecutive_concurrency >= 3:
                            self._force_rpm_learn = True
                        else:
                            return "CONCURRENCY_LIMITED"
                    else:
                        self._save_trace("concurrency_lag")
                        return "CONCURRENCY_LIMITED"
                # cap_1 fallthrough 到 RPM

            # ── RPM 429 路径（保底）：冻结 + 学习 ──
            if not retry or self._force_rpm_learn:
                self._force_rpm_learn = False
                # 首次/cap_1：正常学习
                if self._limit_learned:
                    # 锁存再撞 429：降 learned，ceiling 恢复对冲
                    self._learned_rpm_limit = max(10, int(self._learned_rpm_limit * 0.90))
                    self._rpm_limit = max(10, int(self._learned_rpm_limit * 0.90))
                    self._revalidate_counter = 0
                    if self.trace:
                        self.trace[-1]["learned"] = self._learned_rpm_limit
                    self._save_trace("relearn_429")
                else:
                    # 发现阶段撞 429 — 直接学习，ceiling 恢复会缓慢上探
                    self._consecutive_429 += 1
                    self._prune_clean_window()
                    clean_count = len(self._clean_window)

                    if self._consecutive_429 >= 3:
                        self._rpm_limit = max(10, int((self._rpm_limit or 10) * 0.70))
                        self._limit_learned = True
                        self._save_trace("force_learn")
                    else:
                        new_learned = max(10, int(clean_count * 0.90))
                        if self._learned_rpm_limit > 0:
                            self._learned_rpm_limit = max(self._learned_rpm_limit, new_learned)
                        else:
                            self._learned_rpm_limit = new_learned
                        self._rpm_limit = new_learned
                        self._limit_learned = True
                        if self.trace:
                            self.trace[-1]["learned"] = self._learned_rpm_limit
                        self._save_trace("learn_429")
            
            # 记录 429 到 clean_window（学习已完成，不计入学习计数）
            self._clean_window.append(time.time())
            self._prune_clean_window()
            self._last_429_ts = time.time()  # 用于计算冻结时长
            # 标准 429 冻结逻辑（含 pacing 公式）
            self._rate_limited = True
            self._frozen_added = 0
            if retry_after > 0:
                delay = retry_after
            elif self._rpm_limit is not None and self._rpm_limit < 100000:
                # pacing + cooldown 累加，cooldown 在持续 429 时指数增长
                pace_delay = max(0.01, 60.0 / self._rpm_limit * self.mc - max(elapsed, 0.5))
                delay = pace_delay + self._cooldown
                self._cooldown = min(COOLDOWN_MAX, self._cooldown * 2)
                # 窗口排水延迟：确保解冻时 len < rpm_limit
                self._prune_clean_window()
                excess = len(self._clean_window) - self._rpm_limit + 1
                if excess > 0 and excess < len(self._clean_window):
                    drain_at = self._clean_window[excess] + 60
                    drain_delay = max(0, drain_at - time.time())
                    delay = max(delay, drain_delay)
            else:
                delay = self._cooldown
                self._cooldown = min(COOLDOWN_MAX, self._cooldown * 2)
            self._next_at = time.time() + delay
            self._save_trace("429")
            return "RATE_LIMITED"

        # 200 或 API 错误（均计入 RPM 窗口）
        # ---- v2: 记录干净窗口 ----
        self._clean_window.append(time.time())
        self._prune_clean_window()
        if status_code == 200:
            self._consecutive_429 = 0
            self._consecutive_concurrency = 0
        if status_code == 200:
            self._total_ok += 1
            self._ok_since_grow += 1
            self._cooldown = COOLDOWN_INIT
        else:
            # 非 200 的 API 错误：只记 RPM 窗口，不做后续计算
            if self._rpm_limit is not None and self._rpm_limit < 100000:
                self._next_at = time.time() + max(0.01, 60.0 / self._rpm_limit * self.mc - elapsed)
            else:
                self._next_at = time.time() + 0.01
            return None

        # 延迟 EWMA（通用）
        if not self._lat_ewma_init:
            self._lat_ewma = elapsed
            self._lat_ewma_init = True
        else:
            self._lat_ewma = 0.2 * elapsed + 0.8 * self._lat_ewma

        # ---- v2: RFC 6298 延迟预警（发现阶段减 mc） ----
        # srtt=平滑RTT(稳定基线), rttvar=RTT方差(正常波动范围)
        # elapsed > srtt + 4*rttvar → 真延迟飙升, 非噪声 → mc 减半

        if not self._limit_learned and not self._rate_limited:
            if not self._srtt_init:
                self._srtt = elapsed
                self._rttvar = elapsed / 2.0
                self._srtt_init = True
            else:
                error = abs(self._srtt - elapsed)
                self._rttvar = 0.75 * self._rttvar + 0.25 * error
                self._srtt = 0.875 * self._srtt + 0.125 * elapsed
            threshold = self._srtt + 4.0 * self._rttvar
            if elapsed > threshold:
                self.mc = max(1, self.mc // 2)
                self._ok_since_grow = 0
                self._save_trace("latency_warn")

        # tp_ewma（权重用）
        tp = self.mc / max(self._lat_ewma, 0.001)
        if not self._tp_init:
            self._tp_ewma = tp
            self._tp_init = True
        else:
            self._tp_ewma = 0.3 * tp + 0.7 * self._tp_ewma

        # 自节奏 - 滑动窗口限速（锁存阶段生效）
        if self._limit_learned and self._rpm_limit is not None and self._rpm_limit < 100000:
            self._prune_clean_window()
            if len(self._clean_window) >= self._rpm_limit:
                # 过去 60s 已满，等最老的滑出
                self._next_at = self._clean_window[0] + 60
            else:
                pacing = 60.0 / self._rpm_limit * self.mc
                self._next_at = time.time() + max(0.01, max(0, pacing - elapsed))
                _pc = getattr(self, '_pace_count', 0) + 1
                self._pace_count = _pc
        else:
            self._next_at = time.time() + 0.01

        # mc 增长
        if not self._rate_limited and self._ok_since_grow >= max(5, self.mc * 5):
            if self._limit_learned and self._lat_ewma_init and self._lat_ewma > 0 and self._rpm_limit:
                # 锁存阶段：Little's Law 锚定，防止超速
                useful_mc = max(1, int(self._rpm_limit * self._lat_ewma / 60))
                new_mc = min(self.mc + 1, useful_mc * 2, self._concurrency_cap)
                if new_mc > self.mc:
                    self.mc = new_mc
                    self._ok_since_grow = 0
                    self._save_trace("grow")
            else:
                # 发现阶段：有界增长，受 _concurrency_cap 约束
                if self.mc < self._concurrency_cap:
                    self.mc += 1
                    self._ok_since_grow = 0
                    self._save_trace("grow")
                elif self._concurrency_cap >= 100000:
                    self.mc += 1  # 无限模式（默认 cap=100000 未学习时）
                    self._ok_since_grow = 0
                    self._save_trace("grow")
                # cap 已被学习时（<100000），mc 已达 limit，不增长 -> ok_since_grow 保持累计


        # ---- v2: 锁存阶段再验证（每 100 真实成功） ----
        if self._limit_learned and not self._rate_limited:
            self._revalidate_counter += 1
            if self._revalidate_counter >= 100:
                self._revalidate_counter = 0
                self._rpm_limit = (self._rpm_limit or 10) + 1
                self._save_trace("revalidate")

        # ---- v2: Ceiling 恢复（稳定运行后谨慎上探） ----
        if self._limit_learned and not self._rate_limited:
            if self.current_rpm >= self._rpm_limit * 0.85:
                self._stable_counter += 1
                if self._stable_counter >= 15:
                    self._stable_counter = 0
                    self._rpm_limit = max(self._rpm_limit + 1, int(self._rpm_limit * 1.05))
                    self._save_trace("ceiling_up")
        