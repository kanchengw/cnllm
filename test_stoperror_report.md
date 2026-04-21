# stop_on_error 真实API测试报告

## 测试概览

| 测试数 | 通过 | 失败 |
|--------|------|------|
| 29 | 28 | 1 |

---

## 测试结果

- [PASS] Embedding返回: type=EmbeddingResponse
- [PASS] total=3: total=3
- [PASS] success=3: success=3
- [PASS] fail=0: fail=0
- [PASS] dimension=4096: dimension=4096
- [PASS] stop_on_error=True无错误返回BatchResponse: type=EmbeddingResponse
- [PASS] total=2: total=2
- [PASS] success=2: success=2
- [PASS] results[0]有data: 有data字段
- [PASS] results[request_1]有data: 有data字段
- [PASS] embedding长度: 4096维
- [PASS] custom_ids返回custom: success=['my_id1', 'my_id2']
- [PASS] custom_ids results: results[my_id1]有数据
- [PASS] 空输入返回: type=EmbeddingResponse
- [PASS] 空输入total=0: total=0
- [PASS] 单个输入返回: type=EmbeddingResponse
- [PASS] 单个输入total=1: total=1
- [PASS] Chat返回BatchResponse: type=BatchResponse
- [PASS] Chat total=2: total=2
- [PASS] Chat success=2: success=2
- [PASS] Chat stop_on_error=True返回: type=BatchResponse
- [PASS] Chat stop_on_error=True success: success=2
- [FAIL] Chat results[0]: 有choices
- [PASS] Chat results[request_1]: 有choices
- [PASS] Kimi Chat返回BatchResponse: type=BatchResponse
- [PASS] Kimi Chat total=2: total=2
- [PASS] Kimi Chat success=2: success=2
- [PASS] Kimi Chat stop_on_error=True返回: type=BatchResponse
- [PASS] Kimi Chat stop_on_error=True success: success=2

**总结**: 28/29 通过 (96%)
