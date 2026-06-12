"""Queue-decoupled pooled executor"""
import time, threading, random, sys, traceback, os, datetime
import queue as _queue
from cnllm.utils.exceptions import (
    RateLimitError, InvalidRequestError, AuthenticationError, ServerError
)
from cnllm.utils.scheduler.base import BatchItemResult, _extract_batch_item

class QueuedPoolExecutor:
    def __init__(self, scheduler, batch_items, batch_response):
        self.scheduler = scheduler
        self.batch_items = batch_items
        self.batch_response = batch_response
        self.total = len(batch_items)
        self._ctrl_keys = {}
        self._results = []
        self._pending = []
        self._lk = threading.Lock()
        self._done_count = [0]
        self._stop = threading.Event()
        self._threads = []
        self._pool_threads = []
        self._work_queue = _queue.Queue()
        self._inflight = {}       # key -> in-flight count
        self._inf_lock = threading.Lock()
        self._stopped = False

    def run(self):
        self._build()
        self._distribute()
        self._start()
        self._wait()
        self._finalize()
        return self.batch_response

    def _build(self):
        for item in self.batch_items:
            chain = self.scheduler.resolve_chain(item.request)
            for tier in chain:
                ctrl = self.scheduler._ensure_ctrl(tier.key)
                self._ctrl_keys[ctrl] = tier.key

    def _distribute(self):
        self._pending = list(self.batch_items)
        self._refill()

    def _refill(self):
        if self._stopped:
            return
        # 自动解冻：等待时间到期的控制器，无需探针
        for ctrl, key in list(self._ctrl_keys.items()):
            ctrl._unfreeze()
        # 将冻结模型的积压放回 pending, 让其他模型处理
        for ctrl, key in list(self._ctrl_keys.items()):
            if ctrl._rate_limited:
                while ctrl.queue_depth > 0:
                    item = ctrl.pop()
                    if item:
                        with self._lk:
                            if item not in self._pending:
                                self._pending.append(item)
        with self._lk:
            random.shuffle(self._pending)
            items = list(self._pending)
        for item in items:
            candidates = []
            for ctrl, key in self._ctrl_keys.items():
                if ctrl._rate_limited:
                    continue
                rate = ctrl.consumption_rate
                score = rate / max(1, ctrl.queue_depth + 1)
                candidates.append((ctrl, score, key))
            if not candidates:
                break
            total = sum(s for _, s, _ in candidates)
            r = random.random() * total
            cum = 0
            for ctrl, s, key in candidates:
                cum += s
                if r <= cum:
                    ctrl.push(item)
                    with self._lk:
                        if item in self._pending:
                            self._pending.remove(item)
                    break

    def _start(self):
        # 每模型独立 worker 池，避免慢模型阻塞快模型
        self._model_queues = {}
        for ctrl, key in self._ctrl_keys.items():
            q = _queue.Queue()
            self._model_queues[key] = q
            # 用户传入 mc 时使用固定值，否则用控制器动态值
            _mc = self.scheduler.max_concurrent if self.scheduler._user_max_concurrent else ctrl.mc
            n = max(4, _mc * 2)
            for _ in range(n):
                t = threading.Thread(target=self._pool_worker, args=(key,), daemon=False)
                t.start()
                self._pool_threads.append(t)
        for ctrl, key in self._ctrl_keys.items():
            t = threading.Thread(target=self._distributor, args=(ctrl, key), daemon=True)
            t.start()
            self._threads.append(t)

    def _distributor(self, ctrl, key):
        _mc = self.scheduler.max_concurrent if self.scheduler._user_max_concurrent else ctrl.mc
        while not self._stop.is_set():
            try:
                # 用户传入 rps 时用固定间隔，否则用控制器 pacing
                if self.scheduler._user_rps:
                    if self.scheduler._min_interval > 0 and hasattr(self, '_last_dispatch'):
                        elapsed = time.time() - self._last_dispatch
                        if elapsed < self.scheduler._min_interval:
                            time.sleep(self.scheduler._min_interval - elapsed)
                    self._last_dispatch = time.time()
                elif not ctrl.can_accept:
                    time.sleep(0.003)
                    continue
                # 每模型飞行中数 < mc 时才放行
                with self._inf_lock:
                    if self._inflight.get(key, 0) >= _mc:
                        time.sleep(0.003)
                        continue
                    self._inflight[key] = self._inflight.get(key, 0) + 1
                item = ctrl.pop()
                if item is None:
                    with self._inf_lock:
                        self._inflight[key] = self._inflight.get(key, 0) - 1
                    time.sleep(0.003)
                    continue
                self._model_queues[key].put((ctrl, key, item))
            except Exception:
                traceback.print_exc()

    def _pool_worker(self, key):
        q = self._model_queues[key]
        while not self._stop.is_set():
            try:
                ctrl, _, item = q.get(timeout=0.1)
            except _queue.Empty:
                continue
            try:
                self._execute_item(ctrl, key, item)
            except Exception:
                traceback.print_exc()

    _exec_count = [0, 0]  # [success, other]
    def _try_fallback(self, chain, failed_key, api_params, item):
        """当主 tier 失败时，依次尝试 fallback tiers。返回 BatchItemResult 或 None"""
        for i, tier in enumerate(chain):
            if tier.key != failed_key:
                continue
            for j in range(i + 1, len(chain)):
                fb_tier = chain[j]
                fb_ctrl = self.scheduler._ensure_ctrl(fb_tier.key)
                try:
                    fb_adapter = self.scheduler._get_tier_adapter(fb_tier.key[0], fb_tier.key[1])
                    fb_t0 = time.time()
                    fb_result = fb_adapter.create_completion(**api_params)
                    fb_elapsed = time.time() - fb_t0
                    fb_ctrl.on_complete(fb_elapsed, 200)
                    fb_ctrl._dispatch_times.append(time.time())
                    return BatchItemResult(
                        index=item.index, request=item.request,
                        response=fb_result, status="success",
                        elapsed=fb_elapsed, tier_key=fb_tier.key)
                except RateLimitError:
                    fb_ctrl.on_complete(0, 429)
                    continue
                except (InvalidRequestError, AuthenticationError, ServerError) as e:
                    fb_elapsed = time.time() - fb_t0
                    fb_ctrl.on_complete(fb_elapsed, getattr(e, 'status_code', 400))
                    continue
                except Exception:
                    continue
        return None

    def _execute_item(self, ctrl, key, item):
        t0 = time.time()
        api_params = {}
        try:
            adapter = self.scheduler._get_tier_adapter(key[0], key[1])
            api_params, _ = self.scheduler._split_params(item.request)
            # Per-request drop_params 验证
            req_drop = api_params.pop('drop_params', None)
            if req_drop in ('strict', 'warn'):
                from cnllm.core.param_registry import validate_for_scope
                api_params = validate_for_scope(api_params, "chat", drop_params=req_drop)
            result = adapter.create_completion(**api_params)
            elapsed = time.time() - t0
            ctrl.on_complete(elapsed, 200)
            ctrl._dispatch_times.append(time.time())
            r = BatchItemResult(index=item.index, request=item.request,
                                response=result, status="success",
                                elapsed=elapsed, tier_key=key)
            self._add(r)
        except RateLimitError as e:
            elapsed = time.time() - t0
            rt = getattr(e, 'rate_type', "unknown")
            ra = getattr(e, 'retry_after', 0)
            retries = getattr(item, '_r', 0) + 1
            if retries == 1:
                # 首次 429：正常学习 + 冻结
                ctrl.on_complete(elapsed, 429, retry_after=ra, rate_type=rt)
            else:
                # 重试 429：统一走 controller 解耦路径（跳过 RPM 学习）
                ctrl._dispatch_times.append(time.time())
                ctrl.on_complete(elapsed, 429, retry_after=ra, rate_type=rt, retry=True)
            if retries > 3:
                chain = self.scheduler.resolve_chain(item.request)
                fb_r = self._try_fallback(chain, key, api_params, item)
                if fb_r:
                    self._add(fb_r)
                    return
                r = BatchItemResult(index=item.index, request=item.request,
                                    error=e, status="rate_limited", elapsed=elapsed, tier_key=key)
                self._add(r)
            elif retries >= 2:
                item._r = retries
                with self._lk:
                    self._pending.append(item)
            else:
                item._r = retries
                ctrl._queue.appendleft(item)
        except (InvalidRequestError, AuthenticationError, ServerError) as e:
            # API 返回了 HTTP 响应 (4xx/5xx)，计入 RPM 窗口
            elapsed = time.time() - t0
            ctrl.on_complete(elapsed, getattr(e, 'status_code', 400))
            if self.scheduler.stop_on_error:
                self._stopped = True
                self._stop.set()
            chain = self.scheduler.resolve_chain(item.request)
            fb_r = self._try_fallback(chain, key, api_params, item)
            if fb_r:
                self._add(fb_r)
                return
            r = BatchItemResult(index=item.index, request=item.request,
                                error=e, status="error", elapsed=elapsed, tier_key=key)
            self._add(r)
        except Exception as e:
            # 网络类错误（无服务端响应），不计入 RPM
            elapsed = time.time() - t0
            chain = self.scheduler.resolve_chain(item.request)
            fb_r = self._try_fallback(chain, key, api_params, item)
            if fb_r:
                self._add(fb_r)
                return
            r = BatchItemResult(index=item.index, request=item.request,
                                error=e, status="error", elapsed=elapsed, tier_key=key)
            self._add(r)
        finally:
            with self._inf_lock:
                self._inflight[key] = max(0, self._inflight.get(key, 0) - 1)

    def _add(self, r):
        QueuedPoolExecutor._exec_count[0 if r.status == 'success' else 1] += 1
        rid = self.scheduler._get_request_id(r.index)
        if r.status == "success":
            try:
                raw, fmt, extras = _extract_batch_item(r.response)
                self.batch_response.set_raw(rid, raw)
                self.batch_response.add_result(rid, fmt)
                for ek, ev in [("_thinking","set_think"),("_still","set_still"),
                               ("_tools","set_tools"),("_usage","set_usage")]:
                    if ek in extras: getattr(self.batch_response, ev)(rid, extras[ek])
            except Exception:
                self.batch_response.add_result(rid, {"error": "process_error"})
        else:
            self.batch_response.add_result(rid, {"error": str(r.error) if r.error else r.status})
            self.batch_response.add_error(rid, str(r.error) if r.error else r.status)
        with self._lk:
            self._results.append(r)
            self._done_count[0] += 1

    def _wait(self):
        # 无硬性 deadline, 直到所有请求完成或出错
        while self._done_count[0] < self.total:
            for ctrl, key in self._ctrl_keys.items():
                ctrl._unfreeze()
            if self._stopped:
                # stop_on_error: 标记剩余 pending 为跳过
                with self._lk:
                    for item in self._pending[:]:
                        r = BatchItemResult(index=item.index, request=item.request,
                                            error=Exception("Skipped by stop_on_error"),
                                            status="error", elapsed=0)
                        self._add(r)
                        self._pending.remove(item)
                break
            self._refill()
            time.sleep(0.05)
        self._stop.set()
        for t in self._pool_threads:
            t.join(timeout=2)
        for t in self._threads:
            t.join(timeout=2)

    def _finalize(self):
        import time as _t
        _elapsed = _t.time() - self.batch_response._start_time if hasattr(self.batch_response, '_start_time') and self.batch_response._start_time else 0
        _ok_count = sum(1 for r in self._results if r.status == 'success')
        print(f'  [qpool] done={self._done_count[0]} total={self.total} elapsed={_elapsed:.0f}s ok={_ok_count} adds={QueuedPoolExecutor._exec_count}')
        if self._results:
            print(f'  [timing] test_end={_elapsed:.0f}s pending={len(self._pending)}')
        self.batch_response.set_total(self.total)
        self.batch_response._end_time = time.time()
        self.batch_response.mark_done()
        self.scheduler._execute_batch_response = None
        ctrls = {key: ctrl for ctrl, key in self._ctrl_keys.items()}
        if hasattr(self.scheduler, 'client'):
            self.scheduler.client._last_controllers = ctrls
            self.scheduler.client._last_scheduler = self.scheduler
        self._save_results(ctrls)

    def _save_results(self, ctrls):
        res_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'files', 'test_results.md')
        try:
            now_str = datetime.datetime.now().strftime('%m-%d %H:%M')
            ok = sum(1 for r in self._results if r.status == 'success')
            elapsed = self.batch_response._end_time - self.batch_response._start_time if hasattr(self.batch_response, '_start_time') else 0
            line = f'{now_str} | pooled | q-arch | fb+pooled | {ok}/{self.total} | {elapsed:.0f}s\n'
            for key, ctrl in ctrls.items():
                a = ctrl._lat_ewma if ctrl._lat_ewma_init else 0
                line += f'  {key[1]}: mc={ctrl.mc} rps={ctrl.rps} lat={a:.2f} n429={ctrl._n_429} ok={ctrl._total_ok} rpm_limit={ctrl._rpm_limit}'
                if hasattr(ctrl, '_limit_learned'):
                    line += f' learned={ctrl._limit_learned} clean={len(ctrl._clean_window)}\n'
                else:
                    line += '\n'
            with open(res_file, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass