#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chat Batch全面测试（Mock方式）
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CNLLM_CONFIG_DIR'] = 'glm'

test_results = []

def log_test(name: str, passed: bool, detail: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    test_results.append({"name": name, "status": status, "passed": passed, "detail": detail})
    print(f"  {status}: {name}")
    if detail:
        print(f"         {detail}")

def create_mock_response(prompt):
    return {
        'choices': [{'message': {'content': f'response for {prompt}'}}],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 10, 'total_tokens': 15}
    }

# ==================== Chat Batch基础测试 ====================
print("\n" + "="*60)
print("Chat Batch - 基础字段访问测试 (Mock)")
print("="*60)

mock_create = MagicMock(side_effect=create_mock_response)

# 需要在导入前patch
with patch('cnllm.core.adapter.BaseAdapter.create_completion', mock_create):
    from cnllm.entry.client import CNLLM
    
    client = CNLLM(model='test-model', api_key='test-key')
    resp = client.chat.batch(['hello', 'world'])
    
    # success字段
    success = resp.success
    log_test("success字段返回List", 
            isinstance(success, list) and len(success) == 2,
            f"type={type(success).__name__}, len={len(success)}")
    
    # success内容格式
    log_test("success内容为request_X格式",
             success == ['request_0', 'request_1'],
             f"value={success}")
    
    # fail字段
    fail = resp.fail
    log_test("fail字段返回List",
             isinstance(fail, list),
             f"value={fail}")
    
    # results字段
    results = resp.results
    log_test("results字段返回BatchResults",
             hasattr(results, '__getitem__'),
             f"type={type(results).__name__}")
    
    # results.keys()返回
    keys = list(results.keys())
    log_test("results.keys()返回ID列表",
             keys == ['request_0', 'request_1'],
             f"value={keys}")
    
    # results[str]访问
    r0 = results.get('request_0')
    log_test("results['request_0']访问",
             r0 is not None and 'choices' in r0,
             f"found={r0 is not None}")
    
    # elapsed字段
    elapsed = resp.elapsed
    log_test("elapsed字段返回float",
             isinstance(elapsed, (int, float)),
             f"value={elapsed}")
    
    # total字段
    total = resp.total
    log_test("total字段返回int",
             isinstance(total, int) and total == 2,
             f"value={total}")
    
    # success_count
    sc = resp.success_count
    log_test("success_count返回int",
             isinstance(sc, int) and sc == 2,
             f"value={sc}")
    
    # fail_count
    fc = resp.fail_count
    log_test("fail_count返回int",
             isinstance(fc, int) and fc == 0,
             f"value={fc}")
    
    # response[str]访问
    r0_resp = resp['request_0']
    log_test("response['request_0']访问",
             r0_resp is not None and 'choices' in r0_resp,
             f"found={r0_resp is not None}")
    
    # response[0]整数索引
    r0_int = resp[0]
    log_test("response[0]整数索引(无custom_ids)",
             r0_int is not None and 'choices' in r0_int,
             f"found={r0_int is not None}")
    
    # results[0]整数索引 (Chat Batch用BatchResults)
    r0_results = results[0]
    log_test("results[0]整数索引",
             r0_results is not None and 'choices' in r0_results,
             f"found={r0_results is not None}")

# ==================== Chat Batch Custom IDs测试 ====================
print("\n" + "="*60)
print("Chat Batch - Custom IDs功能测试 (Mock)")
print("="*60)

mock_create2 = MagicMock(side_effect=create_mock_response)

with patch('cnllm.core.adapter.BaseAdapter.create_completion', mock_create2):
    # 需要重新导入
    from cnllm.entry.client import CNLLM
    
    client = CNLLM(model='test-model', api_key='test-key')
    resp = client.chat.batch(['hello', 'world'], custom_ids=['chat_1', 'chat_2'])
    
    # success返回custom_ids
    success = resp.success
    log_test("success返回custom_ids",
             success == ['chat_1', 'chat_2'],
             f"value={success}")
    
    # results.keys()返回custom_ids
    keys = list(resp.results.keys())
    log_test("results.keys()返回custom_ids",
             keys == ['chat_1', 'chat_2'],
             f"value={keys}")
    
    # results[str]访问custom_id
    r = resp.results.get('chat_1')
    log_test("results['chat_1']访问",
             r is not None and 'choices' in r,
             f"found={r is not None}")
    
    # response['chat_1']访问
    r_resp = resp['chat_1']
    log_test("response['chat_1']访问",
             r_resp is not None and 'choices' in r_resp,
             f"found={r_resp is not None}")
    
    # response[0] -> custom_ids[0]
    r0_int = resp[0]
    log_test("response[0]映射到custom_ids[0]",
             r0_int is not None and 'choices' in r0_int,
             f"found={r0_int is not None}")
    
    # results['request_0']应该无效
    r_request0 = resp.results.get('request_0')
    log_test("results['request_0']应返回None(定制后)",
             r_request0 is None,
             f"value={r_request0}")
    
    # results[0]支持整数索引（Chat Batch用BatchResults）
    r0_results = results[0]
    log_test("results[0]支持整数索引",
             r0_results is not None and 'choices' in r0_results,
             f"found={r0_results is not None}")
    
    # get方法
    g = resp.get('chat_1')
    log_test("response.get('chat_1')方法",
             g is not None and 'choices' in g,
             f"found={g is not None}")
    
    g_int = resp.get(0)
    log_test("response.get(0)整数方法",
             g_int is not None and 'choices' in g_int,
             f"found={g_int is not None}")
    
    # __contains__
    contains = 'chat_1' in resp
    log_test("'chat_1' in response",
             contains == True,
             f"value={contains}")
    
    contains_int = 0 in resp
    log_test("0 in response(整数)",
             contains_int == True,
             f"value={contains_int}")
    
    # items()
    items = list(resp.results.items())
    log_test("results.items()返回迭代器",
             len(items) == 2,
             f"count={len(items)}")
    
    # values()
    values = list(resp.results.values())
    log_test("results.values()返回迭代器",
             len(values) == 2,
             f"count={len(values)}")

# ==================== Chat Batch回调测试 ====================
print("\n" + "="*60)
print("Chat Batch - 回调功能测试 (Mock)")
print("="*60)

callback_results = []

def my_callback(item_result):
    callback_results.append({
        'request_id': getattr(item_result, 'request_id', None),
        'status': getattr(item_result, 'status', None),
        'result': getattr(item_result, 'result', None),
        'error': getattr(item_result, 'error', None),
    })

mock_create3 = MagicMock(side_effect=create_mock_response)

with patch('cnllm.core.adapter.BaseAdapter.create_completion', mock_create3):
    from cnllm.entry.client import CNLLM
    
    client = CNLLM(model='test-model', api_key='test-key')
    resp = client.chat.batch(['test1', 'test2'], callbacks=[my_callback])
    
    log_test("回调函数被调用",
             len(callback_results) > 0,
             f"被调用次数={len(callback_results)}")
    
    if callback_results:
        cr = callback_results[0]
        log_test("回调包含request_id",
                 'request_id' in cr,
                 f"keys={cr.keys()}")
        log_test("回调包含status",
                 'status' in cr,
                 f"status={cr.get('status')}")

# ==================== 统计 ====================
passed = sum(1 for r in test_results if r['passed'])
failed = sum(1 for r in test_results if not r['passed'])
total = len(test_results)

print("\n" + "="*60)
print(f"Chat Batch测试总结: {passed}/{total} 通过, {failed}/{total} 失败")
print("="*60)

# 追加报告
with open("c:/Users/wkc_1/Desktop/CN/test_report.md", "a", encoding="utf-8") as f:
    f.write("\n\n## Chat Batch测试报告\n\n")
    for r in test_results:
        status = "[PASS]" if r['passed'] else "[FAIL]"
        f.write(f"- {status} {r['name']}: {r['detail']}\n")
    f.write(f"\n**总结**: {passed}/{total} 通过, {failed}/{total} 失败\n")