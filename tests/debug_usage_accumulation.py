"""
Mock 测试：验证所有批量路径的 usage 累积逻辑
独立于 cnllm 包运行，不依赖 httpx
"""
import threading
from typing import Dict, Any
from dataclasses import dataclass, field

_DEFAULT_KEEP = frozenset({"still", "think", "tools"})


@dataclass
class BatchResponse:
    _results: Dict = field(default_factory=dict)
    _think: Dict[str, str] = field(default_factory=dict)
    _still: Dict[str, str] = field(default_factory=dict)
    _tools: Dict = field(default_factory=dict)
    _raw: Dict = field(default_factory=dict)
    _usage: Dict[str, Any] = field(default_factory=dict)
    _errors: Dict = field(default_factory=dict)
    _keep: frozenset = field(default_factory=lambda: _DEFAULT_KEEP)
    _success_count: int = 0
    _fail_count: int = 0
    _fields_cleared: bool = False
    _done: bool = False
    _in_for_loop: bool = False
    _condition: threading.Condition = field(default_factory=threading.Condition)

    def add_result(self, rid, data):
        self._results[rid] = data
        self._success_count += 1
        with self._condition:
            self._condition.notify_all()

    def add_error(self, rid, error):
        self._errors[rid] = str(error)
        self._fail_count += 1

    def set_usage(self, rid, value):
        if not self._usage:
            self._usage = dict(value)
        else:
            for k, v in value.items():
                if isinstance(v, (int, float)) and isinstance(self._usage.get(k), (int, float)):
                    self._usage[k] = self._usage.get(k, 0) + v
                else:
                    self._usage[k] = v

    def set_still(self, r, v): self._still[r] = v
    def set_think(self, r, v): self._think[r] = v
    def set_tools(self, r, v): self._tools[r] = v
    def set_raw(self, r, v): self._raw[r] = v

    def mark_done(self):
        self._done = True
        with self._condition:
            self._condition.notify_all()

    def _clear_non_kept_fields(self):
        self._fields_cleared = True
        if "*" in self._keep: return
        if "results" not in self._keep: self._results.clear()
        if "errors" not in self._keep: self._errors.clear()
        if "think" not in self._keep: self._think.clear()
        if "still" not in self._keep: self._still.clear()
        if "tools" not in self._keep: self._tools.clear()
        if "raw" not in self._keep: self._raw.clear()

    def to_dict(self, results=None, think=None, still=None, tools=None,
                raw=None, errors=None, usage=None, status=None):
        data = {}
        if status is not False:
            data["status"] = {"success_count": self._success_count, "fail_count": self._fail_count, "total": self._success_count + self._fail_count, "elapsed": "0.00s"}
        if usage is not False:
            data["usage"] = dict(self._usage)
        _explicit = any(v is not None for v in (results, think, still, tools, raw, errors))
        for field, param in [("results", results), ("think", think), ("still", still),
                             ("tools", tools), ("raw", raw), ("errors", errors)]:
            if param is True: data[field] = dict(getattr(self, f"_{field}"))
            elif param is False: continue
            elif _explicit: continue
            elif "*" in self._keep or field in self._keep: data[field] = dict(getattr(self, f"_{field}"))
        return data

    @property
    def usage(self): return dict(self._usage)

    @property
    def status(self):
        return {"success_count": self._success_count, "fail_count": self._fail_count,
                "total": self._success_count + self._fail_count, "elapsed": "0.00s"}


br = BatchResponse()
br.set_usage("request_0", {"prompt_tokens": 5, "completion_tokens": 50})
assert br.usage == {"prompt_tokens": 5, "completion_tokens": 50}, f"T1: {br.usage}"
print("PASS T1: set_usage 单次")

br2 = BatchResponse()
br2.set_usage("req0", {"prompt_tokens": 5, "completion_tokens": 50})
br2.set_usage("req1", {"prompt_tokens": 3, "completion_tokens": 30})
assert br2.usage == {"prompt_tokens": 8, "completion_tokens": 80}, f"T2: {br2.usage}"
print("PASS T2: set_usage 累积")

br3 = BatchResponse()
br3.set_usage("req0", {"prompt_tokens": 5})
assert br3.usage["prompt_tokens"] == 5, "T3: key access"
print("PASS T3: 扁平 key 访问")

br4 = BatchResponse()
br4.set_usage("req0", {"prompt_tokens": 5})
br4.set_still("req0", "hi")
d = br4.to_dict()
assert "still" in d and "usage" in d and "results" not in d, f"T4: {d.keys()}"
print("PASS T4: to_dict() 无参")

d2 = br4.to_dict(results=True)
assert "results" in d2 and "still" not in d2, f"T5: {d2.keys()}"
print("PASS T5: to_dict(results=True)")

br6 = BatchResponse()
br6.set_usage("req0", {"prompt_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 50}})
assert br6.to_dict()["usage"]["completion_tokens_details"]["reasoning_tokens"] == 50
print("PASS T6: 嵌套 usage")

br7 = BatchResponse()
br7.set_usage("req0", {"prompt_tokens": 5})
br7._clear_non_kept_fields()
assert br7.usage == {"prompt_tokens": 5}, "T7: cleared"
print("PASS T7: clear 不影响 usage")

print("\\n全部通过")
