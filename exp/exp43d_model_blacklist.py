import sys
sys.path.insert(0, r"c:\Users\wkc_1\Desktop\Paper\CNLLM")

from cnllm.core.vendor.qwen import QwenAdapter

# 验证_model_params黑名单存在
adapter = QwenAdapter(api_key="test", model="qwen3.5-plus")
unsupported = adapter._model_params.get("qwen3.5-plus", set())
print(f"qwen3.5-plus unsupported params: {unsupported}")

# 验证enable_search在黑名单中
assert "enable_search" in unsupported, "enable_search should be blacklisted"
print("✓ Model blacklist verification passed")
