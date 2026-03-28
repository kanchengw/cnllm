import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
import os
from cnllm import CNLLM
from cnllm.utils.exceptions import ModelNotSupportedError, FallbackError

api_key = os.getenv('MINIMAX_API_KEY')

print('='*60)
print('验证：调用入口不传模型，走主模型的场景')
print('='*60)

print('\n【场景1】主模型有效 + 无fb')
print('预期：直接用主模型')
try:
    client = CNLLM(model='minimax-m2.7', api_key=api_key)
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print(f'结果：成功, model={result.get("model")}')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景2】主模型有效 + 有fb')
print('预期：先用主模型')
try:
    client = CNLLM(model='minimax-m2.7', api_key=api_key, fallback_models={'minimax-m2.5': api_key})
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print(f'结果：成功, model={result.get("model")}')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景3】主模型无效 + 无fb')
print('预期：失败')
try:
    client = CNLLM(model='invalid-primary', api_key=api_key)
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print(f'结果：成功 (不应该) model={result.get("model")}')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景4】主模型无效 + fb有效')
print('预期：走fb，成功')
try:
    client = CNLLM(model='invalid-primary', api_key=api_key, fallback_models={'minimax-m2.5': api_key})
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print(f'结果：成功, model={result.get("model")}')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景5】主模型无效 + fb无效')
print('预期：失败')
try:
    client = CNLLM(model='invalid-primary', api_key=api_key, fallback_models={'invalid-fb': api_key})
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print('结果：成功 (不应该)')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景6】主模型无效 + fb多个(都无效)')
print('预期：失败')
try:
    client = CNLLM(model='invalid-primary', api_key=api_key, fallback_models={'invalid-fb1': api_key, 'invalid-fb2': api_key})
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print('结果：成功 (不应该)')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n【场景7】主模型无效 + fb多个(fb1无效, fb2有效)')
print('预期：走fb2，成功')
try:
    client = CNLLM(model='invalid-primary', api_key=api_key, fallback_models={'invalid-fb1': api_key, 'minimax-m2.5': api_key})
    result = client.chat.create(messages=[{'role': 'user', 'content': 'hi'}])
    print(f'结果：成功, model={result.get("model")}')
except Exception as e:
    print(f'结果：失败 {type(e).__name__}')

print('\n' + '='*60)
print('验证完成')
print('='*60)
