import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from cnllm import CNLLM, MINIMAX_API_KEY

print("=" * 60)
print("MiniMax 三种调用方式测试")
print("=" * 60)

client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

def verify_openai_format(resp, name):
    required_keys = ["id", "object", "created", "model", "choices", "usage"]
    for key in required_keys:
        if key not in resp:
            print(f"  [FAIL] {name}: missing key '{key}'")
            return False

    choice = resp["choices"][0]
    if "message" not in choice:
        print(f"  [FAIL] {name}: missing 'message' in choices")
        return False

    msg = choice["message"]
    if "role" not in msg or "content" not in msg:
        print(f"  [FAIL] {name}: missing 'role' or 'content' in message")
        return False

    print(f"  [OK] {name}: OpenAI 格式验证通过")
    return True

print("\n[1] 极简调用 client('提示词')...")
try:
    resp = client("用一句话介绍自己")
    verify_openai_format(resp, "极简调用")
    print(f"      回复: {resp['choices'][0]['message']['content'][:40]}...")
except Exception as e:
    print(f"  [FAIL] 极简调用: {e}")

print("\n[2] 标准调用 client.chat.create(prompt='提示词')...")
try:
    resp = client.chat.create(prompt="用一句话介绍自己")
    verify_openai_format(resp, "标准调用")
    print(f"      回复: {resp['choices'][0]['message']['content'][:40]}...")
except Exception as e:
    print(f"  [FAIL] 标准调用: {e}")

print("\n[3] 完整调用 client.chat.create(messages=[...])...")
try:
    resp = client.chat.create(
        messages=[
            {"role": "user", "content": "用一句话介绍自己"}
        ]
    )
    verify_openai_format(resp, "完整调用")
    print(f"      回复: {resp['choices'][0]['message']['content'][:40]}...")
except Exception as e:
    print(f"  [FAIL] 完整调用: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
