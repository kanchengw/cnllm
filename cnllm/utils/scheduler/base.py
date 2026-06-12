"""

批量调度基类模块 — 数据类 + 公共函数 + BatchScheduler 基类

"""

from dataclasses import dataclass, field

from typing import Any, List, Optional, Iterator, AsyncIterator, Callable, Dict


@dataclass
class Tier:
    """Fallback 链中的一级"""
    api_key: str
    model: str

    @property
    def key(self):
        return (self.api_key, self.model)


@dataclass
class StagedRequest:
    """挂起的请求"""
    request: dict
    chain: List['Tier']
    tier_index: int
    retry_at: float

# 调度器内部参数 key，需要从请求中剥离，不传给 adapter
SCHED_KEYS = frozenset({
    "model", "api_key", "fallback_models", "base_url",
    "_input_type", "_orig_idx", "stream",
    "timeout", "max_retries", "retry_delay",
})

import time

import asyncio

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from concurrent.futures import TimeoutError

from cnllm.utils.exceptions import CNLLMError, RateLimitError, ServerError, TimeoutError, NetworkError, AuthenticationError

from cnllm.core.accumulators.batch_accumulator import (

    BatchResponse,

    BatchStreamAccumulator,

    AsyncBatchStreamAccumulator,

)

from cnllm.core.accumulators.embedding_accumulator import EmbeddingResponse

import logging



logger = logging.getLogger(__name__)





def _extract_batch_item(response):

    """从批量单项响应中提取 (raw, formatted, extras) 元组



    chat.create() 可能返回:

    - dict: 原生/已格式化的字典 → raw 和 formatted 相同

    - NonStreamAccumulator: 累积器对象 (.process() 返回 self)

      → raw = _response(原生), formatted = _data(OpenAI格式)

    - StreamAccumulator: 流式累积器 → raw = _chunks(原生chunks列表), formatted = finalize()



    Returns:

        (raw_resp, formatted_dict, extras)

    """

    extras = {}

    if isinstance(response, dict):

        if "usage" in response:

            extras["_usage"] = response["usage"]

        return response, response, extras



    if hasattr(response, 'finalize'):

        # 确保迭代完成，_chunks 已填充

        try:

            for _ in response:

                pass

        except Exception:

            pass

        raw = list(response._chunks) if hasattr(response, '_chunks') else []

        extras["_still"] = response.still

        extras["_thinking"] = response.think

        extras["_tools"] = response.tools

        if hasattr(response, 'usage'):

            try:

                u = response.usage

                if u:

                    extras["_usage"] = u

            except Exception:

                pass

        formatted = response.finalize()

        return raw, formatted, extras



    if hasattr(response, '_response') and hasattr(response, 'process'):

        raw = response._response

        while not isinstance(raw, dict) and hasattr(raw, '_response'):

            raw = raw._response

        formatted = getattr(response, '_data', None)

        if formatted is None:

            try:

                import inspect

                if inspect.iscoroutinefunction(response.process):

                    formatted = response._response

                else:

                    formatted = response.process()

                    if not isinstance(formatted, dict):

                        formatted = response._response

            except Exception:

                formatted = response._response

        # 从 OpenAI 格式 formatted 中提取 still/think/tools（避免跨请求共享 adapter._cnllm_extra）

        if formatted and isinstance(formatted, dict):

            _choices = formatted.get("choices", [])

            if _choices and isinstance(_choices, list) and len(_choices) > 0:

                _choice = _choices[0]

                if isinstance(_choice, dict):

                    _msg = _choice.get("message", {})

                    _content = _msg.get("content", "")

                    if _content:

                        extras["_still"] = _content

                    _reasoning = _msg.get("reasoning_content", "")

                    if _reasoning:

                        extras["_thinking"] = _reasoning

                    _tools = _msg.get("tool_calls")

                    if _tools:

                        extras["_tools"] = _tools

        if hasattr(response, 'usage'):

            try:

                u = response.usage

                if u:

                    extras["_usage"] = u

            except Exception:

                pass

        return raw, formatted, extras

    return response, response, extras





@dataclass

class BatchItem:

    """批量任务项"""

    request: Any

    index: int

    priority: int = 0

    request_id: str = ""





@dataclass

class BatchItemResult:

    """单个请求结果"""

    index: int

    request: Any

    response: Optional[dict] = None

    error: Optional[Exception] = None

    elapsed: float = 0.0

    status: str = "pending"

    request_id: str = ""

    tier_key: Optional[tuple] = None





@dataclass

class BatchResult:

    """批量结果"""

    results: List[BatchItemResult]

    total: int

    success_count: int

    error_count: int

    elapsed: float

    errors: List[Exception]



    @property

    def responses(self) -> List[dict]:

        return [r.response for r in self.results if r.status == "success"]



    @property

    def failed_indexes(self) -> List[int]:

        return [r.index for r in self.results if r.status == "error"]





@dataclass

class BatchItemStreamResult:

    """单个流式请求结果"""

    index: int

    request: Any

    chunk: Optional[dict] = None

    error: Optional[Exception] = None

    status: str = "pending"

    stream_id: Optional[str] = None

    content: str = ""



    @property

    def is_done(self) -> bool:

        return self.status in ("done", "error")



    @property

    def is_error(self) -> bool:

        return self.status == "error"



    def copy(self):

        return BatchItemStreamResult(

            index=self.index,

            request=self.request,

            chunk=self.chunk,

            error=self.error,

            status=self.status,

            stream_id=self.stream_id,

            content=self.content,

        )





from cnllm.utils.scheduler.controller import AdaptiveController
import threading


class BatchScheduler:

    """同步批量调度器"""



    def __init__(

        self,

        client: Any,

        max_concurrent: int = 3,

        rps: float = 0,

        timeout: Optional[float] = None,

        stop_on_error: bool = False,

        callbacks: Optional[List[Callable]] = None,

        max_retries: int = None,

        retry_delay: float = None,

        custom_ids: Optional[List[str]] = None,

        controllers: Optional[Dict] = None,
        fallback_config: Optional[Dict] = None,
        performance: bool = False,
        _user_max_concurrent: bool = False,
        _user_rps: bool = False,

    ):

        self.client = client

        self.max_concurrent = max_concurrent
        self._user_max_concurrent = _user_max_concurrent
        self._user_rps = _user_rps

        self.rps = rps

        self._min_interval = 1.0 / self.rps if self.rps > 0 else 0

        self.timeout = timeout

        self.stop_on_error = stop_on_error

        self.callbacks = callbacks or []

        self.custom_ids = custom_ids

        self.max_retries = max_retries

        self.retry_delay = retry_delay

        self.controllers = controllers if controllers is not None else {}
        self.fallback_config = fallback_config or {}
        self.performance = performance
        self._inflight_lock = threading.Lock()
        self._in_flight = {}
        self._selection_counts = {}
        self._adapter_cache: Dict = {}

        self._adapter = None

        self._execute_batch_response = None

        self._stop_flagged = False



    def _get_adapter(self):

        if self._adapter is None:

            self._adapter = self.client._get_adapter(self.client.model, self.client.api_key)

            self._init_adapter_defaults()

        return self._adapter



    def _init_adapter_defaults(self):

        adapter = self._get_adapter()

        if adapter:

            if self.timeout is None:

                self.timeout = adapter.timeout

            if self.max_retries is None:

                self.max_retries = adapter.max_retries

            if self.retry_delay is None:

                self.retry_delay = adapter.retry_delay



    def _ctrl_key(self):
        try:
            a = self._get_adapter()
            return (a.api_key, a.model)
        except Exception:
            return None

    def _split_params(self, request: dict) -> tuple:
        """拆分为 (api_params, sched_params)"""
        api_params = {k: v for k, v in request.items() if k not in SCHED_KEYS}
        sched_params = {k: v for k, v in request.items() if k in SCHED_KEYS}
        return api_params, sched_params

    def _get_tier_adapter(self, api_key: str, model: str):
        """获取或创建 (api_key, model) 对应的 adapter"""
        key = (api_key, model)
        if key not in self._adapter_cache:
            self._adapter_cache[key] = self.client._get_adapter(
                model=model, api_key=api_key,
                timeout=self.timeout,
                max_retries=1,
                retry_delay=0,
            )
        return self._adapter_cache[key]

    def _ensure_ctrl(self, key):
        """确保 controller 存在（统一 pooled 模式）"""
        if key not in self.controllers:
            self.controllers[key] = AdaptiveController()
        return self.controllers[key]

    def resolve_chain(self, request: dict) -> list:
        """从请求中解析 fallback 链"""
        _, sched = self._split_params(request)
        chain = [Tier(
            api_key=sched.get("api_key") or getattr(self.client, 'api_key', ''),
            model=sched.get("model") or getattr(self.client, 'model', ''),
        )]
        fb_config = sched.get("fallback_models") or self.fallback_config
        if fb_config:
            for fb_model, fb_cfg in fb_config.items():
                chain.append(Tier(
                    api_key=fb_cfg.get("api_key", chain[0].api_key),
                    model=fb_model,
                ))
        return chain

    def should_divert(self, ctrl, chain) -> bool:
        """第一级 controller 压力大时概率性分流"""
        if ctrl._rpm_limit is None or len(chain) < 2:
            return False
        usage = ctrl.current_rpm / ctrl._rpm_limit
        if usage <= 0.7:
            return False
        prob = min(0.5, (usage - 0.7) / 0.3 * 0.5)
        import random
        return random.random() < prob


    def _get_request_id(self, index: int) -> str:

        if self.custom_ids and index < len(self.custom_ids):

            return self.custom_ids[index]

        return f"request_{index}"

    def _execute_tier(self, request: dict, tier, chain, tier_index: int):
        """直调 adapter 执行单个 tier，返回 BatchItemResult"""
        api_params, _ = self._split_params(request)
        adapter = self._get_tier_adapter(tier.api_key, tier.model)
        t0 = time.time()
        try:
            result = adapter.create_completion(**api_params)
            return BatchItemResult(
                index=-1, request=request, response=result,
                status="success", elapsed=time.time() - t0,
                tier_key=tier.key,
            )
        except RateLimitError as e:
            rt = getattr(e, 'rate_type', "unknown")
            return BatchItemResult(
                index=-1, request=request, error=e,
                status="rate_limited", elapsed=time.time() - t0,
                tier_key=tier.key,
            )
        except (ServerError, TimeoutError, NetworkError, AuthenticationError) as e:
            return BatchItemResult(
                index=-1, request=request, error=e,
                status="rate_limited", elapsed=time.time() - t0,
                tier_key=tier.key,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return BatchItemResult(
                index=-1, request=request, error=e,
                status="rate_limited", elapsed=time.time() - t0,
                tier_key=tier.key,
            )

    def stage(self, key, request, chain, tier_index, retry_at):
        """将请求挂起在指定 controller 的 staging 队列"""
        if not hasattr(self, "_staging_dict"):
            self._staging_dict = {}
        self._staging_dict.setdefault(key, []).append(
            StagedRequest(request=request, chain=chain,
                          tier_index=tier_index, retry_at=retry_at))

    def drain_staging(self, key):
        """取出指定 key 的所有 staging 请求"""
        if not hasattr(self, "_staging_dict"):
            return []
        return self._staging_dict.pop(key, [])

    def _send_probe(self, key):
        """对指定 (api_key, model) 发一条探针"""
        ctrl = self.controllers.get(key)
        if not ctrl or not ctrl.time_to_probe():
            return
        ctrl.start_probe()
        adapter = self._get_tier_adapter(key[0], key[1])
        t0 = time.time()
        try:
            adapter.create_completion(
                messages=[{"role": "user", "content": "ping"}])
            ctrl.on_complete(time.time() - t0, 200)
        except Exception:
            ctrl.on_complete(0, 429)


    def execute(self, requests: List[Any], priorities: Optional[List[int]] = None) -> BatchResponse:

        from cnllm.core.accumulators.batch_accumulator import BatchResponse



        if self._execute_batch_response is not None:

            batch_response = self._execute_batch_response

        else:

            batch_response = BatchResponse()

            self._execute_batch_response = batch_response

        start_time = time.time()

        batch_response._start_time = start_time



        if not requests:

            batch_response.set_total(0)

            batch_response._end_time = time.time()

            batch_response.mark_done()

            self._execute_batch_response = None

            return batch_response



        batch_items = []

        for i, request in enumerate(requests):

            if request is None:

                continue

            priority = priorities[i] if priorities and i < len(priorities) else 0

            batch_items.append(BatchItem(request=request, index=i, priority=priority, request_id=self._get_request_id(i)))



        batch_items.sort(key=lambda x: -x.priority)
        from cnllm.utils.scheduler.queued_pool import QueuedPoolExecutor
        return QueuedPoolExecutor(self, batch_items, batch_response).run()




    def _select_tier(self, chain):
        """按通过量加权选择 tier；全员等待时破格选最快可用者"""
        import random, time as _time
        with self._inflight_lock:
            candidates = []
            for tier in chain:
                ctrl = self.controllers.get(tier.key)
                if ctrl and not ctrl.can_accept:
                    continue
                if ctrl and self._in_flight.get(tier.key, 0) >= ctrl.mc:
                    continue
                if not ctrl or not ctrl._lat_ewma_init:
                    candidates.append((tier, 1.0))
                else:
                    w = max(0.3, ctrl._tp_ewma)
                    if ctrl._rpm_limit is not None and ctrl._rpm_limit > 0:
                        usage = ctrl.current_rpm / ctrl._rpm_limit
                        w *= max(0.1, 1.0 - usage)
                    candidates.append((tier, w))
            if candidates:
                total = sum(w for _, w in candidates)
                r = random.random() * total
                cum = 0
                for tier, w in candidates:
                    cum += w
                    if r <= cum:
                        self._selection_counts[tier.model] = self._selection_counts.get(tier.model, 0) + 1
                        self._in_flight[tier.key] = self._in_flight.get(tier.key, 0) + 1
                        return tier
                self._selection_counts[candidates[-1][0].model] = self._selection_counts.get(candidates[-1][0].model, 0) + 1
                self._in_flight[candidates[-1][0].key] = self._in_flight.get(candidates[-1][0].key, 0) + 1
                return candidates[-1][0]
            # 死时间断路器：全员等待/冻结，选 _next_at 最近的非冻结模型
            soonest = None
            soonest_at = float('inf')
            for tier in chain:
                ctrl = self.controllers.get(tier.key)
                if ctrl and ctrl._rate_limited:
                    continue
                if ctrl and self._in_flight.get(tier.key, 0) >= ctrl.mc:
                    continue
                if ctrl and ctrl._next_at < soonest_at:
                    soonest_at = ctrl._next_at
                    soonest = tier
            if soonest is None:
                return None
            self._selection_counts[soonest.model] = self._selection_counts.get(soonest.model, 0) + 1
            self._in_flight[soonest.key] = self._in_flight.get(soonest.key, 0) + 1
        # 锁外等待到模型可用
        wait = soonest_at - _time.time()
        if wait > 0:
            _time.sleep(wait)
        return soonest

    def _execute_single(self, index: int, request: Any) -> BatchItemResult:

        try:
            if self._stop_flagged:

                return BatchItemResult(status="error", error=Exception("Stopped by stop_on_error"))

            # 将 string / 对象 统一为 dict
            if isinstance(request, str):
                req_dict = {"prompt": request}
            elif isinstance(request, dict):
                req_dict = request
            elif hasattr(request, 'to_dict'):
                req_dict = request.to_dict()
            else:
                raise ValueError(f"Invalid request type: {type(request).__name__}")

            # 构建链并遍历
            chain = self.resolve_chain(req_dict)

            if self.performance:
                selected = self._select_tier(chain)
                if selected is None:
                    # all rate-limited: return rate_limited so main loop puts it in staging
                    return BatchItemResult(index=index, request=request, status="rate_limited", error=Exception("All tiers rate-limited"), elapsed=0)
                self._ensure_ctrl(selected.key)
                result = self._execute_tier(req_dict, selected, chain, 0)
                result.index = index
                if result.status == "success":
                    return result
                # 通知被选中的 tier 的 controller 发生了限流（否则 controller 不知自己被 429）
                _sel_ctrl = self.controllers.get(result.tier_key)
                if _sel_ctrl:
                    _ra = getattr(result.error, 'retry_after', 0.0) if result.error else 0.0
                    try:
                        _sel_ctrl.on_complete(result.elapsed, 429, retry_after=_ra)
                    except Exception:
                        pass
                # Release in_flight slot: selected tier's request is done, avoid permanent exclusion
                with self._inflight_lock:
                    if selected.key in self._in_flight:
                        self._in_flight[selected.key] = max(0, self._in_flight[selected.key] - 1)
                # save first result for fallback (if all tiers fail, return this for staging)
                _first_result = result
                for t in chain:
                    if t.key == selected.key:
                        continue
                    tc = self.controllers.get(t.key)
                    if tc and (tc._rate_limited or not tc.can_accept):
                        continue
                    self._ensure_ctrl(t.key)
                    result = self._execute_tier(req_dict, t, chain, 0)
                    result.index = index
                    if result.status == "success":
                        return result
                    # 通知该 fallback tier 的 controller 发生了限流
                    _fb_ctrl = self.controllers.get(result.tier_key)
                    if _fb_ctrl:
                        try:
                            _fb_ctrl.on_complete(result.elapsed, 429)
                        except Exception:
                            pass
                # all tiers failed: return first result as rate_limited for staging retry
                return _first_result
            for tier_idx, tier in enumerate(chain):
                tier_ctrl = self._ensure_ctrl(tier.key)
                if tier_ctrl._rate_limited and tier_idx < len(chain) - 1:
                    continue
                tier_result = self._execute_tier(req_dict, tier, chain, tier_idx)
                tier_result.index = index

                if tier_result.status == "success":
                    return tier_result

                if tier_result.status == "rate_limited":
                    # 更新对应 controller，继续试下一级
                    ctrl = self.controllers.get(tier.key)
                    if ctrl:
                        ra = getattr(tier_result.error, 'retry_after', 0.0) if tier_result.error else 0.0
                        ra = getattr(tier_result.error, 'retry_after', 0.0) if tier_result.error else 0.0
                    rt = getattr(tier_result.error, 'rate_type', "unknown") if tier_result.error else "unknown"
                    ctrl.on_complete(tier_result.elapsed, 429, retry_after=ra, rate_type=rt)
                    continue

            # 最后一级错误透传
            if tier_result.status == "rate_limited":
                return tier_result
            return BatchItemResult(
                index=index, request=request,
                status="error",
                error=tier_result.error or Exception("All tiers failed"),
                elapsed=tier_result.elapsed,
            )
        except Exception as e:
            return BatchItemResult(
                index=index, request=request,
                status="error",
                error=e,
                elapsed=0,
            )



    def _notify_callback(self, result: BatchItemResult):

        for callback in self.callbacks:

            try:

                if asyncio.iscoroutinefunction(callback):

                    asyncio.create_task(callback(result))

                else:

                    callback(result)

            except Exception as e:

                logging.error(f"Callback error: {e}")


BATCH_LEVEL_KEYS = frozenset({
    "max_concurrent", "rps", "stop_on_error",
    "callbacks", "custom_ids", "requests",
})



def _normalize_batch_requests(
    requests_arg=None,
    prompt=None,
    messages=None,
    per_request_defaults=None,
):
    """将用户输入规范化为统一的请求对象列表。"""
    if requests_arg is not None:
        if len(requests_arg) == 0:
            raise TypeError("requests 列表不能为空")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("与 requests 共存时 prompt 必须为字符串，而非列表")
        if messages is not None and (not isinstance(messages, list) or
                                      (messages and not isinstance(messages[0], dict))):
            raise TypeError("与 requests 共存时 messages 必须为单组消息列表，而非列表的列表")
        final_requests = []
        for i, req in enumerate(requests_arg):
            if not isinstance(req, dict):
                raise TypeError(f"requests[{i}] 必须是 dict 类型")
            for batch_key in BATCH_LEVEL_KEYS:
                if batch_key in req and batch_key != "requests":
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"batch() 参数 '{batch_key}' 在 requests[{i}] 中未生效。"
                        f"请在 batch() 全局参数中配置 '{batch_key}'，"
                        f"例如: batch(..., {batch_key}={req[batch_key]})"
                    )
                    req = {k: v for k, v in req.items() if k != batch_key}
            if "prompt" not in req and "messages" not in req:
                if prompt is not None:
                    req["prompt"] = prompt
                elif messages is not None:
                    req["messages"] = messages
            if "prompt" not in req and "messages" not in req:
                raise TypeError(f"requests[{i}] 必须包含 'prompt' 或 'messages' 字段")
            if req.get("prompt") == "":
                raise TypeError(f"requests[{i}] 的 prompt 不能为空字符串")
            if req.get("messages") == []:
                raise TypeError(f"requests[{i}] 的 messages 不能为空列表")
            per_request = req.copy()
            if per_request_defaults:
                defaults = {k: v for k, v in per_request_defaults.items()
                             if k not in per_request}
                per_request = {**defaults, **per_request}
            per_request["_input_type"] = "prompt" if "prompt" in per_request else "messages"
            final_requests.append(per_request)
        return final_requests

    if prompt is not None and messages is not None:
        raise TypeError("batch() 只接受 prompt 或 messages 其中之一，不能同时提供")
    if prompt is None and messages is None:
        raise TypeError("batch() 需要提供 requests 或 prompt 或 messages 参数")
    defaults = per_request_defaults or {}
    if prompt is not None:
        if not isinstance(prompt, list):
            raise TypeError("batch() 的 prompt 参数必须为字符串列表（独立模式），"
                            "如需单个请求请使用 chat.create()")
        prompt = [p for p in prompt if p != ""]
        if len(prompt) == 0:
            raise TypeError("prompt 列表不能为空且不能全为空字符串")
        return [{**defaults, "prompt": p, "_input_type": "prompt"} for p in prompt]
    else:
        if not isinstance(messages, list) or (messages and not isinstance(messages[0], list)):
            raise TypeError("batch() 的 messages 参数必须为消息列表的列表（独立模式），"
                            "如需单个请求请使用 chat.create()")
        messages = [m for m in messages if m and len(m) > 0]
        if len(messages) == 0:
            raise TypeError("messages 列表不能为空")
        return [{**defaults, "messages": m, "_input_type": "messages"} for m in messages]

