"""
E2E 测试：用户传入不同 base_url 格式的真实 API 调用。
"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM
from cnllm.utils.exceptions import AuthenticationError

MODEL = "mimo-v2.5"
API_KEY = os.getenv("XIAOMI_API_KEY")

def test_base_url_v1():
    """base_url=/v1 (规则2)"""
    if not API_KEY:
        pytest.skip("XIAOMI_API_KEY not set")
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com/v1")
    try:
        resp = client.chat.create(prompt="1+1=?", stream=False)
        assert resp["choices"][0]["message"]["content"]
        print(f"  回复: {resp['choices'][0]['message']['content']}")
    except AuthenticationError:
        print("  (URL OK, auth failed)")
    finally:
        client.close()

def test_base_url_domain():
    """base_url=域名 (规则3/4)"""
    if not API_KEY:
        pytest.skip("XIAOMI_API_KEY not set")
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com")
    try:
        resp = client.chat.create(prompt="2+2=?", stream=False)
        assert resp["choices"][0]["message"]["content"]
        print(f"  回复: {resp['choices'][0]['message']['content']}")
    except AuthenticationError:
        print("  (URL OK, auth failed)")
    finally:
        client.close()

def test_base_url_domain_slash():
    """base_url=域名+斜杠"""
    if not API_KEY:
        pytest.skip("XIAOMI_API_KEY not set")
    client = CNLLM(model=MODEL, api_key=API_KEY, base_url="https://token-plan-cn.xiaomimimo.com/")
    try:
        resp = client.chat.create(prompt="3+3=?", stream=False)
        assert resp["choices"][0]["message"]["content"]
        print(f"  回复: {resp['choices'][0]['message']['content']}")
    except AuthenticationError:
        print("  (URL OK, auth failed)")
    finally:
        client.close()
