from cnllm import CNLLM, MINIMAX_API_KEY


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode())


def test_minimax_three_call_methods():
    """MiniMax 三种调用方式测试"""
    client = CNLLM(model="minimax-m2.7", api_key=MINIMAX_API_KEY)

    def verify_openai_format(resp, name):
        required_keys = ["id", "object", "created", "model", "choices", "usage"]
        for key in required_keys:
            assert key in resp, f"{name}: missing key '{key}'"

        choice = resp["choices"][0]
        assert "message" in choice, f"{name}: missing 'message' in choices"

        msg = choice["message"]
        assert "role" in msg and "content" in msg, f"{name}: missing 'role' or 'content'"
        return True

    print("\n" + "=" * 60)
    print("MiniMax 三种调用方式测试")
    print("=" * 60)

    print("\n[1] 极简调用 client('提示词')...")
    resp = client("用一句话介绍自己")
    verify_openai_format(resp, "极简调用")
    content = resp['choices'][0]['message']['content']
    safe_print(f"      回复: {content}")
    print(f"      [PASS] 极简调用")

    print("\n[2] 标准调用 client.chat.create(prompt='提示词')...")
    resp = client.chat.create(prompt="用一句话介绍自己")
    verify_openai_format(resp, "标准调用")
    content = resp['choices'][0]['message']['content']
    safe_print(f"      回复: {content}")
    print(f"      [PASS] 标准调用")

    print("\n[3] 完整调用 client.chat.create(messages=[...])...")
    resp = client.chat.create(
        messages=[
            {"role": "user", "content": "用一句话介绍自己"}
        ]
    )
    verify_openai_format(resp, "完整调用")
    content = resp['choices'][0]['message']['content']
    safe_print(f"      回复: {content}")
    print(f"      [PASS] 完整调用")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_minimax_three_call_methods()
