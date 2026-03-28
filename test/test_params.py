from cnllm.core.params import get_provider_name, PROVIDER_PARAMS


def test_get_provider_name():
    print("\n" + "=" * 60)
    print("test_get_provider_name")
    print("=" * 60)
    result1 = get_provider_name("minimax-m2.7")
    result2 = get_provider_name("minimax-m2.5")
    result3 = get_provider_name("doubao")
    result4 = get_provider_name("kimi")
    result5 = get_provider_name("unknown")
    print(f"'minimax-m2.7' -> '{result1}'")
    print(f"'minimax-m2.5' -> '{result2}'")
    print(f"'doubao' -> '{result3}'")
    print(f"'kimi' -> '{result4}'")
    print(f"'unknown' -> '{result5}' (默认)")
    assert get_provider_name("minimax-m2.7") == "minimax"
    assert get_provider_name("minimax-m2.5") == "minimax"
    assert get_provider_name("doubao") == "doubao"
    assert get_provider_name("kimi") == "kimi"
    assert get_provider_name("unknown") == "minimax"
    print("[PASS]")


def test_provider_params():
    print("\n" + "=" * 60)
    print("test_provider_params")
    print("=" * 60)
    providers = list(PROVIDER_PARAMS.keys())
    print(f"支持的提供商: {providers}")
    assert "minimax" in PROVIDER_PARAMS
    assert "doubao" in PROVIDER_PARAMS
    assert "kimi" in PROVIDER_PARAMS
    print("[PASS]")


def test_minimax_config():
    print("\n" + "=" * 60)
    print("test_minimax_config")
    print("=" * 60)
    minimax = PROVIDER_PARAMS["minimax"]
    print(f"minimax.create.supported: {minimax['create'].get('supported', {})}")
    assert "init" in minimax
    assert "create" in minimax
    assert "group_id" in minimax["create"]["supported"]
    assert "temperature" in minimax["create"]["supported"]
    assert "messages" in minimax["create"]["supported"]
    assert "max_tokens" in minimax["create"]["supported"]
    assert "stream" in minimax["create"]["supported"]
    print("[PASS]")


if __name__ == "__main__":
    test_get_provider_name()
    test_provider_params()
    test_minimax_config()
    print("\n" + "=" * 60)
    print("全部测试完成！")
    print("=" * 60)
