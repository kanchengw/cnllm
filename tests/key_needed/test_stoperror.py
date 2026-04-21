#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stop_on_error 真实API测试
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['CNLLM_CONFIG_DIR'] = 'glm'


def get_api_key():
    key = os.environ.get('GLM_API_KEY', '')
    if not key or key == 'GLM_API_KEY':
        key = os.environ.get('ZC_API_KEY', '')
    return key


API_KEY = get_api_key()
KIMI_API_KEY = os.environ.get('KIMI_API_KEY', '')
print(f"API Keys - GLM: {API_KEY[:10]}..., KIMI: {KIMI_API_KEY[:10]}...")

test_results = []


def log_test(name: str, passed: bool, detail: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    test_results.append({"name": name, "status": status, "passed": passed, "detail": detail})
    print(f"  {status}: {name} {detail}")


# ==================== Embedding Batch stop_on_error 真实API测试 ====================
print("\n" + "="*60)
print("Embedding Batch stop_on_error 真实API测试 (embedding-3-pro)")
print("="*60)

from cnllm.core.vendor.glm import GLMEmbeddingAdapter

embed_client = GLMEmbeddingAdapter(
    api_key=API_KEY,
    model='embedding-3-pro',
    base_url='https://open.bigmodel.cn/api/paas_v4/'
)


# 测试1: stop_on_error=False 正常完成
print("\n--- 测试1: stop_on_error=False 正常完成 ---")
try:
    resp = embed_client.create_batch(input=['hello', 'world', 'test'], stop_on_error=False)
    log_test("Embedding返回", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("total=3", resp.total == 3, f"total={resp.total}")
    log_test("success=3", len(resp.success) == 3, f"success={len(resp.success)}")
    log_test("fail=0", len(resp.fail) == 0, f"fail={len(resp.fail)}")
    log_test("dimension=4096", resp.dimension == 4096, f"dimension={resp.dimension}")
except Exception as e:
    log_test("Embedding默认", False, f"错误: {e}")


# 测试2: stop_on_error=True 正常完成 (无错误时不应抛异常)
print("\n--- 测试2: stop_on_error=True 正常完成 ---")
try:
    resp = embed_client.create_batch(input=['a', 'b'], stop_on_error=True)
    log_test("stop_on_error=True无错误返回BatchResponse", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("total=2", resp.total == 2, f"total={resp.total}")
    log_test("success=2", len(resp.success) == 2, f"success={len(resp.success)}")
except Exception as e:
    log_test("stop_on_error=True正常", False, f"错误: {e}")


# 测试3: 验证results内容可用
print("\n--- 测试3: results内容验证 ---")
try:
    resp = embed_client.create_batch(input=['test1', 'test2'], stop_on_error=False)
    result1 = resp.results[0]
    result2 = resp.results['request_1']
    log_test("results[0]有data", result1 and 'data' in result1, f"有data字段")
    log_test("results[request_1]有data", result2 and 'data' in result2, f"有data字段")
    log_test("embedding长度", len(result1.get('data', [{}])[0].get('embedding', [])) == 4096, f"4096维")
except Exception as e:
    log_test("results验证", False, f"错误: {e}")


# 测试4: custom_ids + stop_on_error
print("\n--- 测试4: custom_ids + stop_on_error ---")
try:
    resp = embed_client.create_batch(
        input=['c1', 'c2'],
        custom_ids=['my_id1', 'my_id2'],
        stop_on_error=False
    )
    log_test("custom_ids返回custom", resp.success == ['my_id1', 'my_id2'], f"success={resp.success}")
    log_test("custom_ids results", resp.results['my_id1'] is not None, "results[my_id1]有数据")
except Exception as e:
    log_test("custom_ids", False, f"错误: {e}")


# 测试5: 空输入
print("\n--- 测试5: 空输入 ---")
try:
    resp = embed_client.create_batch(input=[], stop_on_error=False)
    log_test("空输入返回", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("空输入total=0", resp.total == 0, f"total={resp.total}")
except Exception as e:
    log_test("空输入", False, f"错误: {e}")


# 测试6: 单个输入
print("\n--- 测试6: 单个输入 ---")
try:
    resp = embed_client.create_batch(input='single', stop_on_error=False)
    log_test("单个输入返回", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("单个输入total=1", resp.total == 1, f"total={resp.total}")
except Exception as e:
    log_test("单个输入", False, f"错误: {e}")


# ==================== Chat Batch 真实API测试 ====================
print("\n" + "="*60)
print("Chat Batch 真实API测试")
print("="*60)

from cnllm.entry.client import CNLLM

chat_client = CNLLM(
    model='glm-4.7',
    api_key=API_KEY,
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)


# 测试7: Chat stop_on_error=False
print("\n--- 测试7: Chat stop_on_error=False ---")
try:
    resp = chat_client.chat.batch(['hello', 'world'], stop_on_error=False, max_concurrent=2)
    log_test("Chat返回BatchResponse", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("Chat total=2", resp.total == 2, f"total={resp.total}")
    log_test("Chat success=2", len(resp.success) == 2, f"success={len(resp.success)}")
except Exception as e:
    log_test("Chat默认", False, f"错误: {e}")


# 测试8: Chat stop_on_error=True
print("\n--- 测试8: Chat stop_on_error=True ---")
try:
    resp = chat_client.chat.batch(['你好', '世界'], stop_on_error=True, max_concurrent=2)
    log_test("Chat stop_on_error=True返回", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("Chat stop_on_error=True success", len(resp.success) >= 1, f"success={len(resp.success)}")
except Exception as e:
    log_test("Chat stop_on_error", False, f"错误: {e}")


# 测试9: Chat results访问
print("\n--- 测试9: Chat results访问 ---")
try:
    resp = chat_client.chat.batch(['test1', 'test2'], stop_on_error=False)
    r0 = resp.results[0]
    r1 = resp.results['request_1']
    log_test("Chat results[0]", r0 and 'choices' in r0, "有choices")
    log_test("Chat results[request_1]", r1 and 'choices' in r1, "有choices")
except Exception as e:
    log_test("Chat results", False, f"错误: {e}")


# ==================== Kimi Chat Batch 真实API测试 ====================
print("\n" + "="*60)
print("Kimi Chat Batch 真实API测试 (kimi-k2.6)")
print("="*60)

from cnllm.entry.client import CNLLM as KimiCNLLM

kimi_client = KimiCNLLM(
    model='kimi-k2.6',
    api_key=KIMI_API_KEY,
    base_url='https://api.moonshot.cn/v1'
)


# 测试10: Kimi Chat stop_on_error=False
print("\n--- 测试10: Kimi Chat stop_on_error=False ---")
try:
    resp = kimi_client.chat.batch(['hello', 'world'], stop_on_error=False, max_concurrent=2)
    log_test("Kimi Chat返回BatchResponse", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("Kimi Chat total=2", resp.total == 2, f"total={resp.total}")
    log_test("Kimi Chat success=2", len(resp.success) == 2, f"success={len(resp.success)}")
except Exception as e:
    log_test("Kimi Chat默认", False, f"错误: {e}")


# 测试11: Kimi Chat stop_on_error=True
print("\n--- 测试11: Kimi Chat stop_on_error=True ---")
try:
    resp = kimi_client.chat.batch(['你好', '世界'], stop_on_error=True, max_concurrent=2)
    log_test("Kimi Chat stop_on_error=True返回", hasattr(resp, 'results'), f"type={type(resp).__name__}")
    log_test("Kimi Chat stop_on_error=True success", len(resp.success) >= 1, f"success={len(resp.success)}")
except Exception as e:
    log_test("Kimi Chat stop_on_error", False, f"错误: {e}")


# 统计
passed = sum(1 for r in test_results if r['passed'])
failed = sum(1 for r in test_results if not r['passed'])
total = len(test_results)

print("\n" + "="*60)
print(f"真实API测试: {passed}/{total} 通过, {failed}/{total} 失败")
print("="*60)

# 写入报告
report_path = "c:/Users/wkc_1/Desktop/CN/test_stoperror_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# stop_on_error 真实API测试报告\n\n")
    f.write(f"## 测试概览\n\n")
    f.write(f"| 测试数 | 通过 | 失败 |\n")
    f.write(f"|--------|------|------|\n")
    f.write(f"| {total} | {passed} | {failed} |\n\n")
    f.write("---\n\n")
    f.write("## 测试结果\n\n")
    for r in test_results:
        status = "[PASS]" if r['passed'] else "[FAIL]"
        f.write(f"- {status} {r['name']}: {r['detail']}\n")
    f.write(f"\n**总结**: {passed}/{total} 通过 ({passed*100//total}%)\n")