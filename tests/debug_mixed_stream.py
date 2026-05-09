"""
Debug: 混合流式 batch still/think 提取
直接 import 模块文件，绕过 httpx 依赖
"""
import sys, os, importlib.util

# 直接加载需要的模块，不触发 cnllm.__init__ 的 httpx 依赖
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # 先加载依赖
    for dep_name in list(sys.modules.keys()):
        if dep_name.startswith('cnllm.core') or dep_name.startswith('cnllm.utils'):
            pass  # already loaded
    spec.loader.exec_module(mod)
    return mod

# 手动构建模块依赖链
base_mod = load_module('cnllm.core.accumulators.base',
    os.path.join(os.path.dirname(__file__), '..', 'cnllm', 'core', 'accumulators', 'base.py'))

single_mod = load_module('cnllm.core.accumulators.single_accumulator',
    os.path.join(os.path.dirname(__file__), '..', 'cnllm', 'core', 'accumulators', 'single_accumulator.py'))

batch_mod = load_module('cnllm.utils.batch',
    os.path.join(os.path.dirname(__file__), '..', 'cnllm', 'utils', 'batch.py'))

from cnllm.utils.batch import _extract_batch_item
from cnllm.core.accumulators.single_accumulator import NonStreamAccumulator, StreamAccumulator
from unittest.mock import MagicMock

# 构造 mock adapter
adapter = MagicMock()
adapter._cnllm_extra = {}
adapter._raw_response = None
adapter.__class__.__name__ = 'MockAdapter'

# ===== 测试1: 非流式 → path 3 =====
print("=== 测试1: NonStreamAccumulator → _extract_batch_item path 3 ===")

raw_dict = {
    "id": "test",
    "choices": [{"message": {"content": "hello response", "role": "assistant"}, "index": 0, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10}
}

ns = NonStreamAccumulator(raw_dict, adapter)
ns.process()

print(f"  _response type: {type(ns._response).__name__}")
print(f"  _response is dict: {isinstance(ns._response, dict)}")
print(f"  has _response: {hasattr(ns, '_response')}")
print(f"  has process: {hasattr(ns, 'process')}")
print(f"  _data: {type(ns._data).__name__ if hasattr(ns, '_data') and ns._data else 'NONE'}")
if hasattr(ns, '_data') and ns._data:
    c = ns._data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  _data content: '{c}'")

raw, formatted, extras = _extract_batch_item(ns)
print(f"  extras keys: {list(extras.keys())}")
print(f"  extras['_still']: '{extras.get('_still', 'NOT SET')}'")
print(f"  formatted is dict: {isinstance(formatted, dict)}")
if isinstance(formatted, dict):
    c = formatted.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  formatted content: '{c}'")


# ===== 测试2: 流式 → path 2 =====
print("\n=== 测试2: StreamAccumulator → _extract_batch_item path 2 ===")

chunks = [
    {"choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]},
    {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
    {"choices": [{"delta": {"content": " world"}, "index": 0}]},
    {"choices": [{"delta": {}}, {"index": 0, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5}},
]

def stream_iter():
    for c in chunks:
        yield c

adapter2 = MagicMock()
adapter2._cnllm_extra = {}
adapter2._raw_response = None

def fake_accumulate(chunk):
    delta = (chunk.get("choices") or [{}])[0].get("delta", {}) if chunk.get("choices") else {}
    content = delta.get("content", "")
    if content:
        if "_still" not in adapter2._cnllm_extra:
            adapter2._cnllm_extra["_still"] = ""
        adapter2._cnllm_extra["_still"] += content

adapter2._accumulate_extra_fields = fake_accumulate

sa = StreamAccumulator(stream_iter(), adapter2)

raw, formatted, extras = _extract_batch_item(sa)
print(f"  extras keys: {list(extras.keys())}")
print(f"  extras['_still']: '{extras.get('_still', 'NOT SET')}'")
if isinstance(raw, dict):
    delta = raw.get("choices", [{}])[0].get("delta", {}) if raw.get("choices") else {}
    print(f"  raw delta content: '{delta.get('content', '')}'")


# ===== 测试3: set_still 行为 =====
print("\n=== 测试3: BatchResponse.set_still / set_think ===")
ba_path = os.path.join(os.path.dirname(__file__), '..', 'cnllm', 'core', 'accumulators', 'batch_accumulator.py')
ba_mod = load_module('cnllm.core.accumulators.batch_accumulator', ba_path)
from cnllm.core.accumulators.batch_accumulator import BatchResponse

br = BatchResponse()
br.set_still("request_0", "Hello world")
br.set_think("request_0", "thinking text")
br.set_still("request_1", "hi response")
br.set_think("request_1", "")

print(f"  still: {len(br.still)} keys = {list(br.still.keys())}")
for k in br.still:
    print(f"    {k}: '{br.still[k][:50]}'")
print(f"  think: {len(br.think)} keys = {list(br.think.keys())}")
for k in br.think:
    print(f"    {k}: '{br.think[k][:50]}'")
PYEOF