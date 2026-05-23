"""Quick test: ernie-5.0 thinking -> .think / .raw"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()
from cnllm import CNLLM

api_key = os.getenv("BAIDU_API_KEY")
if not api_key:
    print("SKIP: no BAIDU_API_KEY")
    sys.exit(0)

client = CNLLM(model="ernie-5.0", api_key=api_key)
resp = client.chat.create(
    messages=[{"role": "user", "content": "用一句话介绍北京"}],
    thinking=True,
)
print("=== RAW ===")
print(resp.raw)
print()
print(f".think: {repr(resp.think)}")
print(f".still: {repr(resp.still)}")
print(f"think length: {len(resp.think) if resp.think else 0}")
