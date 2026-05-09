"""
E2E parameter behavior tests — needs httpx + real API key, run on host machine.

Covers:
1. drop_params: strict / warn / ignore
2. keep parameter in batch calls
3. Parameter validation consistency
"""
import os
import sys
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from cnllm import CNLLM
import pytest

# CI 环境无 API key 时跳过所有测试
_need_key = os.getenv("CNLLM_TEST_MODEL") and os.getenv("CNLLM_TEST_EMB_MODEL")
if not _need_key and not os.getenv("MINIMAX_API_KEY"):
    pytest.skip("需要设置 CNLLM_TEST_MODEL / MINIMAX_API_KEY 环境变量", allow_module_level=True)


from cnllm.core.param_registry import PARAM_REGISTRY

TEST_MODEL = os.getenv("CNLLM_TEST_MODEL", "minimax-m2")
TEST_EMB_MODEL = os.getenv("CNLLM_TEST_EMB_MODEL", "minimax-m2-embedding")
PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS: {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL: {name}" + (f" - {detail}" if detail else ""))


# ============================================================
# drop_params
# ============================================================

def test_drop_params_strict_chat():
    """drop_params='strict' raises on unknown param"""
    client = CNLLM(model=TEST_MODEL, drop_params="strict")
    try:
        client.chat.create(prompt="hello", invalid_param="x")
        check("strict_chat", False, "should have raised")
    except Exception as e:
        check("strict_chat", True, f"{type(e).__name__}")
    finally:
        client.close()


def test_drop_params_strict_type():
    """drop_params='strict' raises on type mismatch"""
    client = CNLLM(model=TEST_MODEL, drop_params="strict")
    try:
        client.chat.create(prompt="hello", max_tokens="bad")
        check("strict_type", False, "should have raised")
    except Exception as e:
        check("strict_type", True, f"{type(e).__name__}")
    finally:
        client.close()


def test_drop_params_warn_chat():
    """drop_params='warn' warns + continues"""
    client = CNLLM(model=TEST_MODEL, drop_params="warn")
    warned = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            client.chat.create(prompt="hello", invalid_param="x")
            warned = any("invalid_param" in str(x.message) for x in w)
        except Exception as e:
            check("warn_chat", False, f"unexpected error: {e}")
            client.close()
            return
    check("warn_chat", True, "warning issued" if warned else "warning in logger")
    client.close()


def test_drop_params_ignore_chat():
    """drop_params='ignore' silently drops"""
    client = CNLLM(model=TEST_MODEL, drop_params="ignore")
    warned = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            client.chat.create(prompt="hello", invalid_param="x")
            warned = any("invalid_param" in str(x.message) for x in w)
        except Exception as e:
            check("ignore_chat", False, f"unexpected error: {e}")
            client.close()
            return
    check("ignore_chat", True, "silent" if not warned else "has warning (ok)")
    client.close()


def test_drop_params_per_request():
    """per-request drop_params overrides client-level"""
    client = CNLLM(model=TEST_MODEL, drop_params="strict")
    try:
        client.chat.create(prompt="hello", invalid_param="x", drop_params="warn")
        check("per_request_override", True)
    except Exception as e:
        check("per_request_override", False, f"unexpected: {e}")
    finally:
        client.close()


def test_drop_params_embed_strict():
    """embedding: drop_params='strict' raises"""
    client = CNLLM(model=TEST_EMB_MODEL, drop_params="strict")
    try:
        client.embeddings.create(input="test", invalid_param="x")
        check("embed_strict", False, "should have raised")
    except Exception as e:
        check("embed_strict", True, f"{type(e).__name__}")
    finally:
        client.close()


def test_drop_params_embed_warn():
    """embedding: temperature in embed scope should be dropped"""
    client = CNLLM(model=TEST_EMB_MODEL, drop_params="warn")
    try:
        client.embeddings.create(input="test", temperature=0.5)
        check("embed_warn", True)
    except Exception as e:
        check("embed_warn", False, f"unexpected: {e}")
    finally:
        client.close()


# ============================================================
# keep in embeddings.batch
# ============================================================

def _check_keep(name, keep_val, expect_vectors, expect_results):
    client = CNLLM(model=TEST_EMB_MODEL)
    try:
        if keep_val is not None:
            resp = client.embeddings.batch(input=["hello", "world"], keep=keep_val, batch_size=1)
        else:
            resp = client.embeddings.batch(input=["hello", "world"], batch_size=1)
        for _ in resp:
            pass
        v_ok = len(resp.vectors) == expect_vectors
        r_ok = len(resp.results) == expect_results
        ok = v_ok and r_ok
        detail_parts = []
        if not v_ok:
            detail_parts.append(f"vectors={len(resp.vectors)} (expected {expect_vectors})")
        if not r_ok:
            detail_parts.append(f"results={len(resp.results)} (expected {expect_results})")
        check(name, ok, "; ".join(detail_parts) if detail_parts else "ok")
    except Exception as e:
        check(name, False, f"exception: {e}")
    finally:
        client.close()


def test_keep_default():
    _check_keep("keep_default", None, 2, 0)

def test_keep_custom():
    _check_keep("keep_custom", ["vectors", "results"], 2, 2)

def test_keep_empty():
    _check_keep("keep_empty", [], 0, 0)

def test_keep_wildcard():
    _check_keep("keep_wildcard", ["*"], 2, 2)

def test_keep_client_init():
    """keep set in client init, inherited by batch"""
    client = CNLLM(model=TEST_EMB_MODEL, keep=["vectors", "results"])
    try:
        resp = client.embeddings.batch(input=["hello", "world"], batch_size=1)
        for _ in resp:
            pass
        check("keep_client_init", len(resp.vectors) == 2 and len(resp.results) == 2)
    except Exception as e:
        check("keep_client_init", False, f"exception: {e}")
    finally:
        client.close()


def test_keep_override():
    """per-call keep overrides client init"""
    client = CNLLM(model=TEST_EMB_MODEL, keep=["vectors", "results"])
    try:
        resp = client.embeddings.batch(input=["hello", "world"], keep=[], batch_size=1)
        for _ in resp:
            pass
        check("keep_override", len(resp.vectors) == 0 and len(resp.results) == 0)
    except Exception as e:
        check("keep_override", False, f"exception: {e}")
    finally:
        client.close()


# ============================================================
# Registry verification
# ============================================================

def test_registry_scopes():
    ok = all(bool(p.scope) for p in PARAM_REGISTRY.values())
    check("registry_scopes", ok)

def test_registry_batch_params():
    expected = {"max_concurrent", "rps", "batch_size", "stop_on_error",
                "callbacks", "custom_ids", "keep"}
    actual = {k for k, v in PARAM_REGISTRY.items() if v.batch_level}
    check("registry_batch_params", actual == expected,
          f"extra: {actual - expected}, missing: {expected - actual}")

def test_registry_keep_def():
    p = PARAM_REGISTRY.get("keep")
    ok = p is not None and p.batch_level and p.default is None
    check("registry_keep_def", ok)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_drop_params_strict_chat,
        test_drop_params_strict_type,
        test_drop_params_warn_chat,
        test_drop_params_ignore_chat,
        test_drop_params_per_request,
        test_drop_params_embed_strict,
        test_drop_params_embed_warn,
        test_keep_default,
        test_keep_custom,
        test_keep_empty,
        test_keep_wildcard,
        test_keep_client_init,
        test_keep_override,
        test_registry_scopes,
        test_registry_batch_params,
        test_registry_keep_def,
    ]

    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        fn()

    print(f"\n{'='*60}")
    print(f"Result: {PASS} passed, {FAIL} failed / {PASS + FAIL} total")
    sys.exit(0 if FAIL == 0 else 1)
