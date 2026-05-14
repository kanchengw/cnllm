"""
pytest 全局配置
"""
# httpx 已在 cnllm/entry/http.py 和 cnllm/core/embedding.py 模块级导入，
# 不受 sys.modules stub 影响，无需额外恢复。
