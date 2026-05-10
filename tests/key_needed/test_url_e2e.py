"""
E2E 测试：用户传入不同 base_url 格式的真实 API 调用。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM
from cnllm.utils.exceptions import AuthenticationError

MODEL = "mimo-v2.5"
API_KEY = os.getenv("XIAOMI_API_KEY")

P = 0; F = 0
def t(name, fn):
    global P, F
    if not API_KEY:
        print(f"  SKIP: {name} (no XIAOMI_API_KEY)"); return
    try:
        fn(); P += 1
        print(f"  PASS: {name}")
    except AuthenticationError:
        P += 1
        print(f"  PASS: {name} (URL OK, auth failed)")
    except Exception as e:
        import traceback; F += 1
        print(f"  FAIL: {name}: {e}"); traceback.print_exc()

def _1():
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com/v1")
    resp = client.chat.create(prompt="1+1=?", stream=False)
    print(f"    回复: {resp['choices'][0]['message']['content']}")
t("base_url=/v1 (规则2)", _1)

def _2():
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com")
    resp = client.chat.create(prompt="2+2=?", stream=False)
    print(f"    回复: {resp['choices'][0]['message']['content']}")
t("base_url=域名 (规则3/4)", _2)

def _3():
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com/")
    resp = client.chat.create(prompt="3+3=?", stream=False)
    print(f"    回复: {resp['choices'][0]['message']['content']}")
t("base_url=域名+斜杠", _3)

print(f"\n{'='*40}")
print(f"结果: {P} 通过, {F} 失败 / {P + F} 总")
sys.exit(1 if F else 0)
