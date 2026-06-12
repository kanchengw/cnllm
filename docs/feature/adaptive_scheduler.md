# 自适应调度算法设计文档

## 一、整体架构

### 1.1 执行引擎

所有请求统一走 `QueuedPoolExecutor`。`performance` 参数仅用于控制 `_build()` 创建控制器的数量（单模型 vs 全 tier 池化）。

```
execute() → QueuedPoolExecutor.run()
               ├─ _build()       创建控制器（单模型/全tier）
               ├─ _distribute()  将全部请求放入 pending
               ├─ _start()       启动 worker 线程 + distributor 线程
               ├─ _wait()        等待完成, 每50ms unfreeze+refill
               └─ _finalize()    收尾, 保存结果
```

### 1.2 分发层（`_refill`）

```
_refill():
  1.  调用所有 ctrl._unfreeze() — 窗口已排空的控制器自动解冻
  2.  drain 冻结模型的队列积压 → pending（等其他模型接手）
  3.  跳过 _rate_limited 模型
  4.  剩余可用模型中加权随机分发
      score = consumption_rate / (queue_depth + 1)

_distributor(ctrl, key):
  循环检查 can_accept（_next_at）→ inflight < mc → pop → push 到 model 队列
  worker 独立线程池, 每模型 max(4, mc×2) 个
```

### 1.3 Worker 池

```
_start():
  每模型独立 worker 池，避免慢模型阻塞快模型
  n_workers = max(4, ctrl.mc × 2)
  worker 从模型专属队列拉取 → 执行 adapter.create_completion()
```

---

## 二、自适应控制器（AdaptiveController）

### 2.1 变量总表

| 变量 | 初始值 | 说明 |
|------|--------|------|
| `mc` | 1 | 并发槽位数 |
| `_rpm_limit` | 10 | 限流目标（起步值，锁存后生效） |
| `_learned_rpm_limit` | 0 | 历史最佳限流值（锁存后只降不升基础值） |
| `_rate_limited` | False | 是否冻结（RPM 429 冻结，并发不冻结） |
| `_limit_learned` | False | 是否已学习 rpm_limit（发现阶段→锁存阶段） |
| `_concurrency_cap` | 100000 | 并发上限（concurrency 429 后降低） |
| `_consecutive_429` | 0 | 连续 RPM 429 计数 |
| `_consecutive_concurrency` | 0 | 连续并发 429 计数（无成功间隔则递增） |
| `_clean_window` | deque() | 过去 60s 所有请求时间戳（200+429，不含探针） |
| `_stable_counter` | 0 | Ceiling 恢复计数 |
| `_revalidate_counter` | 0 | 再验证计数 |
| `_cooldown` | 2.0 | RPM 冻结冷却时间（指数增长，上限 32s） |
| `_srtt` / `_rttvar` | 0 | RFC 6298 RTT 估计 |
| `_next_at` | 0.0 | 下次可发送时间戳 |
| `_last_429_ts` | 0.0 | 最近一次 429 时间戳 |
| `_ok_since_grow` | 0 | 上次 mc 增长后成功次数 |
| `_force_rpm_learn` | False | cap=1 连续≥3次时强制学 RPM（覆盖 retry 跳过） |
| `_concurrency_handled` | False | 本轮 inflight 已处理过并发降级（防止同批多次降） |
| `_frozen_added` | — | 冻结期残留 429 新增的 clean_window 条目数（解冻时排除） |

**已移除变量对照**（文档→代码）：

| 旧文档用名 | 当前代码 | 状态 |
|-----------|---------|------|
| `_probe_mode` | — | 已移除（窗口监控替代探针） |
| `_mc_limit` | `_concurrency_cap` | 已重命名 |
| `_concurrency_fallback` | — | 已移除 |
| `_recovery_deadline` | — | 已移除（排水期移除） |
| `_concurrency_backoff_until` | — | 已移除（并发不再冻结） |

### 2.2 控制器生命周期

```
                ┌─────────────────────────────────────┐
                │         发现阶段                      │
                │  _limit_learned=False                 │
                │  mc 自由增长，_rpm_limit=10 占位       │
                │  _next_at = now + 0.01（不限速）      │
                └──────────┬──────────────────────────┘
                           │ 首次 RPM 429
                           ▼
                ┌─────────────────────────────────────┐
                │         锁存阶段                      │
                │  _limit_learned=True                  │
                │  _rpm_limit = learned                 │
                │  滑动窗口限速，mc Little's Law 锚定    │
                │  ceiling 上探（revalidate/ceiling_up）│
                │  再次 RPM 429 → learned × 0.90 重学   │
                └─────────────────────────────────────┘

并发 429 独立于上述生命周期，任何时候都可能触发，
且不改变 _limit_learned 状态。
```

### 2.3 发现阶段（`_limit_learned=False`）

```
节奏:
  _next_at = now + 0.01（不限速）
  mc 以 max(5, mc×5) 次成功 +1 的速度增长，上限受 _concurrency_cap 约束（不突破已学习的并发上限）
  速率 = mc / latency，跑满 API 带宽
  rpm_limit = 10（占位，不影响行为）

延迟预警:
  srtt+4×rttvar → mc//=2（发现阶段有效，锁存后关闭）

出口: RPM 429 → 学习（跳转到锁存阶段）
      并发 429 → mc -= max(1, mc//5)，不改变 _limit_learned
      并发 429 (mc=1, cap=1) → 转 RPM 路径学习（cap已到底）
```

### 2.4 锁存阶段（`_limit_learned=True`）

```
节奏:
  滑动窗口限速（详细见第四节）
  len(_clean_window) < _rpm_limit → _next_at = 60/rpm_limit × mc（考虑并发）
  len(_clean_window) ≥ _rpm_limit → 等最老滑出

mc 增长:
  Little's Law 锚定: useful_mc = rpm_limit × lat_ewma / 60
  new_mc = min(mc + 1, useful_mc × 2, _concurrency_cap)

再次 RPM 429:
  _learned_rpm_limit = max(10, _learned_rpm_limit × 0.90)
  _rpm_limit = _learned_rpm_limit
  冻结（同首次）

Ceiling 上探:
  再验证: 每 100 成功 → rpm_limit + 1
  ceiling_up: ≥85% 利用率持续 15 次 → max(rpm+1, rpm×1.05)
```

---

## 三、两种 429 类型—完全解耦

### 3.1 类型判定

```python
is_concurrency = (retry_after >= 0 and retry_after < 2.0) or rate_type == "concurrency"
```

判定顺序：`retry_after < 2s` 是明确的并发信号；`rate_type` 由适配器从 API 错误文案/Header 解析（关键词匹配规则详见 `docs/429type.md`）。其余所有情况走 RPM 路径（保底）。

### 3.2 Concurrency 429（并发）

```
mc = max(1, mc - max(1, mc // 5))
_concurrency_cap = min(_concurrency_cap, mc)
_consecutive_concurrency += 1

```
mc 降级（每批 inflight 只一次）:
  if not _concurrency_handled:
      _concurrency_handled = True
      mc = max(1, mc - max(1, mc // 5))
      _concurrency_cap = min(_concurrency_cap, mc)
      _consecutive_concurrency += 1
      delay = retry_after if > 0 else max(0.5, elapsed × 2)
      _next_at = now + delay
      记录到 _clean_window

      调整 cap_1 检查:
        if cap==1 and mc==1:
            if _consecutive_concurrency >= 3:
                _force_rpm_learn = True  # fallthrough RPM
            else:
                return "CONCURRENCY_LIMITED"  # 退避重试
        else:
            return "CONCURRENCY_LIMITED"

残留 inflight（_concurrency_handled=True）:
  只写 _clean_window（API 端已统计）
  if cap==1 and mc==1:
      _consecutive_concurrency += 1
      if >= 3: _force_rpm_learn = True  # fallthrough RPM
      else: return
  else: return "concurrency_lag"

锁存阶段过滤（二次撞墙保护）:
  if is_concurrency AND _limit_learned AND _concurrency_cap == 1:
      is_concurrency = False  # 无可降并发，强制走 RPM
```

**不冻结、不学 RPM**（正常路径）。首次 `mc=1, cap=1` 时退避重试，连续 3+ 次才强制学 RPM。

`_concurrency_handled` 确保每批 inflight 只降一次 mc。以 `_next_at` 过期作为"新批次开始"的判定。`_consecutive_concurrency` 仅用于 cap_1 的连续计数，控制 RPM 学习触发。

---

## 八、池化分发算法（QueuedPoolExecutor）

### 8.1 两层执行架构

```
                 QueuedPoolExecutor.run()
                        │
            ┌───────────┴───────────┐
            │                       │
      _refill（分发层）        _distributor（调度层）
            │                       │
   pending → 控制器队列    控制器队列 → worker 池
            │                       │
   每50ms 批量分发           逐条实时调度
   加权随机分配               inflight ≤ mc 控制
```

`QueuedPoolExecutor` 和 `AdaptiveController` 的协作：

| 组件 | 职责 | 控制维度 |
|------|------|---------|
| `_refill()` | 将 pending 请求分配到各控制器的队列 | `score = rate / (queue_depth + 1)` 加权随机、不关心 timing |
| `_distributor` 线程 | 从控制器队列逐条取出，按 pacing 发送给 worker | `_next_at` (timing)、`mc` (并发) |
| `_pool_worker` 线程 | 执行实际 HTTP 调用 | 无控制，纯执行 |
| `AdaptiveController` | 决策何时可以发送、冻结、退避 | `_rate_limited`、`_next_at`、`mc` |

### 8.2 请求完整生命周期

```
1. _normalize_batch_requests() → batch_items
2. _distribute() → 全部放入 self._pending
3. _refill() 循环:
   a. 遍历所有 ctrl._unfreeze()              ← 解冻到期的控制器
   b. 遍历 _rate_limited 的 ctrl，drain 队列到 pending  ← 冻结模型的任务转移
   c. 对 pending 随机 shuffle
   d. 遍历 pending, 加权选 ctrl → ctrl.push(item)  ← 分发
4. _distributor 线程（每个 ctrl 一个）:
   a. ctrl.can_accept? (time >= _next_at)
   b. inflight < mc?
   c. ctrl.pop() → 放入 model_queue
5. _pool_worker 线程（每模型 N 个）:
   a. 从 model_queue 取 item
   b. adapter.create_completion()
   c. ctrl.on_complete(status_code)          ← 通知 controller
   d. _add(result) → _done_count++           ← 计入完成
6. _done_count == total → _wait() 退出
7. _finalize() → 收尾、保存结果
```

### 8.3 分发策略（`_refill`）

```
_refill() 每 50ms 执行一次:

1. 解冻阶段: 遍历所有 ctrl._unfreeze()
   → 排空窗口的控制器自动解冻（_rate_limited=False）

2. Drain 阶段: 遍历冻结控制器
   if ctrl._rate_limited:
       drain 全部队列积压 → self._pending
   → 其他模型可接手这些请求

3. 分发阶段:
   random.shuffle(pending)               ← 公平轮转
   for item in items:
       candidates = []
       for ctrl in all_controllers:
           if ctrl._rate_limited: continue  ← 跳过冻结
           rate = ctrl.consumption_rate     ← 近 10s 实际吞吐
           score = rate / (queue_depth + 1)  ← 加权公式
           candidates.append((ctrl, score))

       加权随机选择:
         total = sum(scores)
         r = random() × total
         cum = 0
         for ctrl, score in candidates:
             cum += score
             if r <= cum:
                 ctrl.push(item)          ← 放入控制器队列
                 pending.remove(item)
                 break
```

**加权公式 `score = consumption_rate / (queue_depth + 1)` 的效果：**

| 场景 | 控制器 A | 控制器 B | 效果 |
|------|---------|---------|------|
| 队列空、A 快 | rate=3, depth=0 → score=3 | rate=0.5, depth=0 → score=0.5 | A 更大概率 |
| 队列空、B 快 | rate=0.5, depth=0 → score=0.5 | rate=3, depth=0 → score=3 | B 更大概率 |
| A 堵了 | rate=2, depth=5 → score=0.33 | rate=2, depth=0 → score=2 | B 优先（depth 惩罚） |

### 8.4 实时调度（`_distributor` 线程）

每个 controller 有**一个独立的 `_distributor` 线程**：

```
while not _stop:
    if not ctrl.can_accept:          # time < _next_at → 等待
        sleep(3ms); continue
    if inflight >= ctrl.mc:          # 并发已满 → 等待
        sleep(3ms); continue
    inflight++
    item = ctrl.pop()                 # 从控制器队列取出
    if item is None:                  # 队列空
        inflight--; continue
    model_queue.put(item)            # 交给 worker
```

**双重节流：**

| 检查 | 控制什么 | 由谁设置 |
|------|---------|---------|
| `can_accept` (time ≥ `_next_at`) | **发送速率**（RPM） | `AdaptiveController._next_at` |
| `inflight < mc` | **并发深度** | `AdaptiveController.mc` |

两者同时满足才能发送。`_next_at` 控制"多久发一条"，`mc` 控制"同时能发几条"。两者相乘得到总吞吐上限。

### 8.5 Worker 池

```
_start():
  for each controller:
      n_workers = max(4, ctrl.mc × 2)
      for _ in range(n_workers):
          Thread(target=_pool_worker).start()
      Thread(target=_distributor).start()
```

- **每模型独立 worker 池**：慢模型不阻塞快模型
- **worker 数 ≥ mc×2**：确保池中有空闲 worker 等待，不因 worker 不足限制并发
- **`_pool_worker`** 从 `_model_queues[key]` 拉取 → 执行 `adapter.create_completion()` → 回调 `ctrl.on_complete()` → `self._add()`

### 8.6 重试链与 pending 生命周期

```
首次 429 (retries==1):
  ctrl.on_complete(429, retry=False)   ← controller 学习/冻结
  item._r = 1
  ctrl._queue.appendleft(item)         ← 放回队首立即重试

第二次 429 (retries==2):
  ctrl.on_complete(429, retry=True)    ← controller 跳过学习，仅冻结
  item._r = 2
  self._pending.append(item)           ← 进入 pending，等 refill 重新分发

第三次 429 (retries==3):
  同上，retries=3 → pending

第四次 429 (retries==4, >3):
  self._add(BatchItemResult(status="rate_limited"))  ← 最终失败
```

**pending 中的请求如何回到执行流：**

```
请求 → 429 → _pending.append(item)
    → _refill() 每 50ms 运行
    → 从 pending 取出 → 选非冻结 ctrl → ctrl.push(item)
    → _distributor 检测到队列非空 → pop → worker 执行
    → 成功 → _add(r) → _done_count++
    → 又 429 → 循环...
```

**pending 去重保护：** `if item not in self._pending:` 确保不重复入队。

### 8.7 `performance` 模式（池化 vs 单模型）

```
_build():
  if scheduler.performance:
      # 池化：每个 fallback tier 都创建独立 controller
      for tier in chain:
          ctrl = ensure_ctrl(tier.key)
  else:
      # 单模型：只创建 primary
      tier = chain[0]
      ctrl = ensure_ctrl(tier.key)
```

| 模式 | 控制器数量 | 适用场景 |
|------|-----------|---------|
| **单模型** (`performance=False`) | 1 个 | 单 API key、fallback 由适配器层处理 |
| **池化** (`performance=True`) | 每个 tier 各 1 个 | 多 API key、多模型并行，快模型不被慢模型阻塞 |

池化模式下各 tier 独立 controller，`_refill` 的加权随机分发会自然地将请求分配给当前最快的 tier。

### 8.8 线程安全模型

| 数据 | 保护方式 | 访问者 |
|------|---------|--------|
| `_pending` | `self._lk` (Lock) | `_refill`、`_execute_item` |
| `_results` | `self._lk` | `_add` |
| `_done_count` | `self._lk` (list 引用) | `_add`、`_wait` |
| `_inflight` | `self._inf_lock` (Lock) | `_distributor`、`_execute_item` |
| `ctrl._queue` | deque 原子操作 (GIL) | `_distributor` pop、`_refill` push |
| `_model_queues[key]` | `queue.Queue` 内置锁 | `_distributor` put、`_pool_worker` get |
| `batch_response` | GIL + dict 原子操作 | `_add`（在 `_lk` 外） |

### 8.9 QueuedPoolExecutor 与 AdaptiveController 协同总览

```
                   批处理请求 (N 条)
                        │
                        ▼
              ┌─────────────────────┐
              │  QueuedPoolExecutor  │
              │                     │
              │  _pending (N 条)     │◄── 429 重试也回到这里
              │       │             │
              │   _refill() 分发     │
              │       │             │
              │  ctrl._queue 各模型   │
              │       │             │
              │  _distributor 调度    │── can_accept (time) ──┐
              │       │             │── inflight < mc  ──────┤
              │  _model_queues      │                        │
              │       │             │                        │
              │  _pool_workers 执行  │── ctrl.on_complete() ──┘
              │       │             │
              │  _add(result)       │── _done_count++
              └──────┬──────────────┘
                     │
                     ▼
              batch_response (N/N 成功)

                    AdaptiveController
              ┌─────────────────────────┐
              │  429 判定               │ is_concurrency / RPM
              │  mc 降级 / cap 学习      │
              │  RPM 学习 + 冻结         │
              │  _next_at pacing         │ 60/rpm × mc - elapsed
              │  _clean_window 限速       │ 滑动窗口 60s
              │  _unfreeze 自动解冻       │
              │  ceiling_up / revalidate  │ 限流值上探
              └─────────────────────────┘
```

**数据流：** 请求 → pending → `_refill` → 队列 → `_distributor` → worker → API → `on_complete` → controller 决策 → `_add` → 完成

**控制流：** `_next_at` + `mc` + `_rate_limited` 由 controller 设置，`_distributor` 和 `_refill` 读取执行。controller 做决策，executor 做执行，职责完全分离。

**锁定阶段 `len < limit` 分支的 pacing 公式：**

```
改前: _next_at = now + 60/rpm_limit × mc
      问题: latency + pacing 双倍累加，实际间隔 = max(latency, pacing) 而非 latency + pacing
改后: _next_at = now + max(0.01, max(0, 60/rpm_limit × mc - elapsed))
```

`- elapsed` 减去已等待的响应时间，使间隔 = `max(latency, pacing)`。

## 六、冻结期残留请求保护

冻结期间仍可能有冻结前发出的请求返回 429，记入 `_clean_window` 但不延长冻结：

```
frozen 429:
  _clean_window.append(time.time())  // API 端已统计
  _frozen_added += 1                 // 标记为冻结期新增
  return None                        // 不降级、不学 RPM
```

`_unfreeze` 检查时排除冻结期条目：

```
live_cw = len(cw) - _frozen_added
if live_cw >= rpm_limit: return False  // 只看冻结前的窗口
```

## 七、发现阶段 mc 增长修正

```
改前: self._ok_since_grow = 0 和 _save_trace("grow") 在 if/elif 外无条件执行
      → mc=1, cap=1 时 _ok_since_grow 每 5 次成功归零，永远涨不上去

改后: 仅在 mc 实际增长时才重置 _ok_since_grow 和保存 trace
```