import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['CNLLM_CONFIG_DIR'] = 'glm'

from cnllm.core.vendor.glm import GLMEmbeddingAdapter

api_key = os.environ.get('GLM_API_KEY', '')
if not api_key or api_key == 'GLM_API_KEY':
    api_key = os.environ.get('ZC_API_KEY', '')

print(f'Using API key: {api_key[:10]}...')

client = GLMEmbeddingAdapter(
    api_key=api_key,
    model='embedding-3-pro',
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

print('\n=== Test 1: batch with input=[list] - basic ===')
response = client.create_batch(input=['你好世界', 'hello world'])
print(f'Success: {response.success_count}, Fail: {response.fail_count}')
print(f'Success IDs: {response.success}')
print(f'Fail IDs: {response.fail}')
print(f'Results keys: {list(response.results.keys())}')

print('\n=== Test 2: batch with custom_ids ===')
response = client.create_batch(
    input=['文本1', '文本2', '文本3'],
    custom_ids=['doc_001', 'doc_002', 'doc_003']
)
print(f'Success count: {response.success_count}')
print(f'Success IDs (期望显示自定义ID): {response.success}')
print(f'Results keys (期望显示自定义ID): {list(response.results.keys())}')
print(f'Fail IDs: {response.fail}')

print('\n=== Test 3: access results with custom_id ===')
try:
    result_doc001 = response.results['doc_001']
    print(f'results["doc_001"] OK: {result_doc001.get("data", [{}])[0].get("embedding", [])[:5]}...')
except Exception as e:
    print(f'results["doc_001"] Error: {e}')

try:
    result_0 = response.results[0]
    print(f'results[0] OK: {result_0.get("data", [{}])[0].get("embedding", [])[:5]}...')
except Exception as e:
    print(f'results[0] Error: {e}')

try:
    result_request0 = response.results['request_0']
    print(f'results["request_0"] OK: {result_request0.get("data", [{}])[0].get("embedding", [])[:5]}...')
except Exception as e:
    print(f'results["request_0"] Error: {e}')

print('\n=== Test 4: single string input ===')
response = client.create_batch(input='hello single')
print(f'Success: {response.success_count}')
print(f'Success IDs: {response.success}')
print(f'Results keys: {list(response.results.keys())}')
if response.results:
    emb = response.results.get('request_0', {}).get("data", [{}])[0].get("embedding", [])
    print(f'Dimension: {len(emb)}')

print('\n=== Test 5: batch via Namespace (client.embeddings.batch) ===')
from cnllm.entry.client import CNLLM
client2 = CNLLM(
    model='embedding-3-pro',
    api_key=api_key,
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)
response = client2.embeddings.batch(
    input=['hello', 'world'],
    custom_ids=['id1', 'id2']
)
print(f'Success: {response.success_count}')
print(f'Success IDs: {response.success}')
print(f'Results keys: {list(response.results.keys())}')
print(f'Dimension: {response.dimension}')

print('\n=== Test 6: try __getitem__ access ===')
try:
    item = response['id1']
    print(f'response["id1"] OK: dimension = {len(item.get("data", [{}])[0].get("embedding", []))}')
except Exception as e:
    print(f'response["id1"] Error: {e}')

try:
    item = response[0]
    print(f'response[0] OK: dimension = {len(item.get("data", [{}])[0].get("embedding", []))}')
except Exception as e:
    print(f'response[0] Error: {e}')

print('\n=== All tests completed! ===')