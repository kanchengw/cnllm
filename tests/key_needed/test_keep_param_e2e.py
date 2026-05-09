"""
E2E 测试：keep 参数
- 默认保留 still/think/tools + 统计字段
- keep=[] 仅保留统计字段
- 非 keep 字段迭代后返回空容器
- to_dict 行为一致
"""
import os
import sys
import time
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v

from cnllm import CNLLM

CHAT_MODEL = "deepseek-v4-flash"
CHAT_KEY = os.environ.get("DEEPSEEK_API_KEY")
requires_chat = pytest.mark.skipif(not CHAT_KEY, reason="需要 DEEPSEEK_API_KEY")


@requires_chat
def test_1_default_keep():
    """默认 keep：循环后 still/think/tools/usage 保留，raw/results 为空"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
    )
    for chunk in acc:
        pass

    # 保留字段
    assert len(acc.still) == 2
    assert len(acc.think) == 2
    usage = acc.usage
    assert usage["prompt_tokens"] > 0

    # 统计字段
    counts = acc.status
    assert counts["total"] == 2
    print(f"  PASS: 默认 keep — still={len(acc.still)}, usage={usage}")


@requires_chat
def test_2_keep_empty():
    """keep=[]：仅保留统计字段，所有数据字段为空"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
        keep=[],
    )
    for chunk in acc:
        pass

    for field_name in ["still", "think", "tools", "raw", "results"]:
        val = getattr(acc, field_name)
        assert len(val) == 0, f"keep=[] 后 {field_name} 应为空, 实际 {len(val)}"

    usage = acc.usage
    assert usage["prompt_tokens"] > 0
    counts = acc.status
    assert counts["total"] == 2
    print(f"  PASS: keep=[] — 仅统计字段保留")


@requires_chat
def test_3_keep_all():
    """keep=all：所有字段保留"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
        keep=["still", "think", "tools", "raw", "results"],
    )
    for chunk in acc:
        pass

    assert len(acc.raw) > 0
    assert len(acc.results) > 0
    assert len(acc.still) > 0
    assert len(acc.think) > 0
    usage = acc.usage
    assert usage["prompt_tokens"] > 0
    print(f"  PASS: keep=all — 所有字段保留")


@requires_chat
def test_4_to_dict_default():
    """默认 to_dict：status/usage 始终包含 + 默认 keep 字段"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
    )
    for chunk in acc:
        pass

    d = acc.to_dict()
    # 统计字段始终存在
    assert "status" in d
    assert isinstance(d["status"]["elapsed"], str) and d["status"]["elapsed"].endswith("s")
    assert d["status"]["success_count"] >= 0
    assert "usage" in d
    assert d["usage"]["prompt_tokens"] > 0
    # 默认 keep 字段
    assert "think" in d
    assert "still" in d
    assert "tools" in d
    # 未 keep 字段在迭代后不加入 dict
    assert "raw" not in d
    assert "results" not in d
    print(f"  PASS: to_dict 默认 — keys={sorted(d.keys())}")


@requires_chat
def test_5_to_dict_explicit():
    """to_dict(results=True)：强制包含 results"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
    )
    for chunk in acc:
        pass

    d = acc.to_dict(results=True)
    assert "results" in d
    assert "usage" in d
    print(f"  PASS: to_dict(results=True) — keys={sorted(d.keys())}")


@requires_chat
def test_6_to_dict_keep_empty():
    """keep=[] + to_dict：仅含统计字段"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    acc = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
        stream=True,
        keep=[],
    )
    for chunk in acc:
        pass

    d = acc.to_dict()
    assert "usage" in d
    assert d["usage"]["prompt_tokens"] > 0
    assert d["status"]["success_count"] >= 0
    # keep=[] 时，数据字段不加入 dict，仅保留元数据
    for field in ["think", "still", "tools", "raw", "results"]:
        assert field not in d, f"keep=[] to_dict 不应含 {field}"
    print(f"  PASS: keep=[] to_dict — keys={sorted(d.keys())}")


@requires_chat
def test_7_sync_non_streaming():
    """同步非流式 + 默认 keep"""
    client = CNLLM(model=CHAT_MODEL, api_key=CHAT_KEY)
    resp = client.chat.batch(
        requests=[{"prompt": "hello"}, {"prompt": "hi"}],
    )
    for _ in resp:
        pass

    assert len(resp.still) == 2
    usage = resp.usage
    assert usage["prompt_tokens"] > 0
    assert len(resp.raw) == 0
    assert len(resp.results) == 0
    print(f"  PASS: 同步非流式 keep")


def run_all():
    tests = [
        test_1_default_keep,
        test_2_keep_empty,
        test_3_keep_all,
        test_4_to_dict_default,
        test_5_to_dict_explicit,
        test_6_to_dict_keep_empty,
        test_7_sync_non_streaming,
    ]
    passed, failed, errors = 0, 0, []

    for test in tests:
        try:
            test()
            passed += 1
            time.sleep(15)
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
    if errors:
        print("失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    run_all()
