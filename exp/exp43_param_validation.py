
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time
import logging
import io
import traceback

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

CNLLM_ROOT = os.path.join(os.path.dirname(__file__), "..", "CNLLM")
sys.path.insert(0, CNLLM_ROOT)

from cnllm.core.param_registry import (
    validate_for_scope,
    PARAM_REGISTRY,
    _SKIP_FIELDS,
)
from cnllm.utils.exceptions import InvalidRequestError


def _capture_logs(func, *args, **kwargs):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("cnllm.core.param_registry")
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        result = func(*args, **kwargs)
        logs = buf.getvalue()
        return result, logs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def experiment_a_drop_params():
    from cnllm import CNLLM

    print("\n" + "=" * 60)
    print("  Experiment (a): drop_params Three-Level Strategy")
    print("=" * 60)

    results = []

    unsupported_params = {"fake_param": "value", "nonexistent_field": True}

    for mode in ["warn", "ignore", "strict"]:
        try:
            client = CNLLM(
                model="deepseek-chat",
                api_key=API_KEY,
                timeout=30,
                max_retries=1,
                drop_params=mode,
            )

            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            handler.setLevel(logging.WARNING)
            cnllm_logger = logging.getLogger("cnllm")
            cnllm_logger.addHandler(handler)
            prev_level = cnllm_logger.level
            cnllm_logger.setLevel(logging.WARNING)

            t0 = time.time()
            try:
                resp = client.chat.create(
                    messages=[{"role": "user", "content": "Say hi"}],
                    max_tokens=10,
                    temperature=0.3,
                    **unsupported_params,
                )
                elapsed = time.time() - t0
                logs = buf.getvalue()
                content = resp.still if hasattr(resp, "still") else str(resp)[:80]
                results.append({
                    "test": f"unsupported_params_drop_params={mode}",
                    "drop_params": mode,
                    "status": "call_succeeded",
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "content_preview": content[:60],
                    "warning_logged": bool(logs.strip()),
                    "log_snippet": logs[:200].strip() if logs.strip() else "",
                })
            finally:
                cnllm_logger.removeHandler(handler)
                cnllm_logger.setLevel(prev_level)

        except InvalidRequestError as e:
            results.append({
                "test": f"unsupported_params_drop_params={mode}",
                "drop_params": mode,
                "status": "blocked_by_InvalidRequestError",
                "error_type": "InvalidRequestError",
                "error_msg": str(e)[:200],
                "identifies_param": "fake_param" in str(e) or "nonexistent_field" in str(e),
                "provides_remediation": "drop_params" in str(e),
            })
        except TypeError as e:
            results.append({
                "test": f"unsupported_params_drop_params={mode}",
                "drop_params": mode,
                "status": "blocked_by_TypeError",
                "error_type": "TypeError",
                "error_msg": str(e)[:200],
            })
        except Exception as e:
            results.append({
                "test": f"unsupported_params_drop_params={mode}",
                "drop_params": mode,
                "status": "unexpected_error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
            })
        time.sleep(1)

    type_mismatch_cases = [
        ("strict", "temperature", "hot"),
        ("warn", "temperature", "hot"),
        ("ignore", "temperature", "hot"),
    ]
    for mode, param_name, param_val in type_mismatch_cases:
        try:
            client = CNLLM(
                model="deepseek-chat",
                api_key=API_KEY,
                timeout=30,
                max_retries=1,
                drop_params=mode,
            )

            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            handler.setLevel(logging.WARNING)
            cnllm_logger = logging.getLogger("cnllm")
            cnllm_logger.addHandler(handler)
            prev_level = cnllm_logger.level
            cnllm_logger.setLevel(logging.WARNING)

            t0 = time.time()
            try:
                resp = client.chat.create(
                    messages=[{"role": "user", "content": "Say hi"}],
                    max_tokens=10,
                    **{param_name: param_val},
                )
                elapsed = time.time() - t0
                logs = buf.getvalue()
                content = resp.still if hasattr(resp, "still") else str(resp)[:80]
                results.append({
                    "test": f"type_mismatch_drop_params={mode}",
                    "drop_params": mode,
                    "param": param_name,
                    "value": param_val,
                    "status": "call_succeeded_type_mismatch_ignored",
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "content_preview": content[:60],
                    "warning_logged": bool(logs.strip()),
                    "log_snippet": logs[:200].strip() if logs.strip() else "",
                })
            finally:
                cnllm_logger.removeHandler(handler)
                cnllm_logger.setLevel(prev_level)

        except TypeError as e:
            results.append({
                "test": f"type_mismatch_drop_params={mode}",
                "drop_params": mode,
                "param": param_name,
                "value": param_val,
                "status": "blocked_by_TypeError",
                "error_type": "TypeError",
                "error_msg": str(e)[:200],
                "identifies_param": param_name in str(e),
                "identifies_expected_type": "float" in str(e) or "int" in str(e),
            })
        except Exception as e:
            results.append({
                "test": f"type_mismatch_drop_params={mode}",
                "drop_params": mode,
                "param": param_name,
                "value": param_val,
                "status": "unexpected_error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
            })
        time.sleep(1)

    for r in results:
        print(f"  [{r['test']}] status={r['status']}")
        if "warning_logged" in r:
            print(f"    warning_logged={r['warning_logged']}")
        if "error_msg" in r:
            print(f"    error: {r['error_msg'][:100]}")

    return results


def experiment_b_param_classification():
    print("\n" + "=" * 60)
    print("  Experiment (b): Parameter Classification Accuracy")
    print("=" * 60)

    vendor_yaml = {
        "optional_fields": {
            "search": {},
            "enable_thinking": {},
            "base_url": {"skip": True},
        },
        "required_fields": {
            "api_key": {"skip": True},
            "model": None,
        },
    }

    test_cases = [
        {
            "category": "standard_param",
            "description": "PARAM_REGISTRY registered + scope match -> accepted",
            "params": {"temperature": 0.7, "max_tokens": 100},
            "scope": "chat",
            "drop_params": "warn",
            "expected_behavior": "accepted",
            "expected_keys": ["temperature", "max_tokens"],
        },
        {
            "category": "yaml_skip_marker",
            "description": "YAML skip markers (api_key, base_url) -> skipped",
            "params": {"api_key": "sk-test", "base_url": "https://api.test.com"},
            "scope": "chat",
            "drop_params": "warn",
            "expected_behavior": "skipped",
            "expected_keys": [],
        },
        {
            "category": "batch_level_param",
            "description": "batch_level=True params in create scope -> treated as unknown",
            "params": {"max_concurrent": 5, "rps": 2.0},
            "scope": "chat",
            "drop_params": "warn",
            "expected_behavior": "rejected_as_unknown",
            "expected_keys": [],
        },
        {
            "category": "vendor_specific_param",
            "description": "YAML optional_fields whitelist -> accepted",
            "params": {"search": True},
            "scope": "chat",
            "drop_params": "warn",
            "expected_behavior": "accepted",
            "expected_keys": ["search"],
        },
        {
            "category": "unknown_param",
            "description": "No match in PARAM_REGISTRY or YAML -> drop_params policy",
            "params": {"fake_param": "value"},
            "scope": "chat",
            "drop_params": "warn",
            "expected_behavior": "rejected_as_unknown",
            "expected_keys": [],
        },
    ]

    results = []
    for tc in test_cases:
        result, logs = _capture_logs(
            validate_for_scope,
            tc["params"],
            tc["scope"],
            vendor_yaml,
            tc["drop_params"],
        )
        actual_keys = sorted(result.keys())
        expected_keys = sorted(tc["expected_keys"])
        classification_correct = actual_keys == expected_keys

        entry = {
            "category": tc["category"],
            "description": tc["description"],
            "input_params": tc["params"],
            "expected_behavior": tc["expected_behavior"],
            "actual_keys": actual_keys,
            "expected_keys": expected_keys,
            "classification_correct": classification_correct,
            "warning_logged": bool(logs.strip()),
        }
        results.append(entry)

        status_mark = "PASS" if classification_correct else "FAIL"
        print(f"  [{status_mark}] {tc['category']}: expected={expected_keys}, actual={actual_keys}")

    strict_cases = [
        {
            "category": "unknown_param_strict",
            "description": "Unknown param with drop_params=strict -> InvalidRequestError",
            "params": {"fake_param": "value"},
            "scope": "chat",
            "drop_params": "strict",
            "expected_behavior": "raises_InvalidRequestError",
        },
        {
            "category": "batch_level_strict",
            "description": "Batch-level param with drop_params=strict -> InvalidRequestError",
            "params": {"max_concurrent": 5},
            "scope": "chat",
            "drop_params": "strict",
            "expected_behavior": "raises_InvalidRequestError",
        },
    ]

    for tc in strict_cases:
        raised = False
        error_type = None
        error_msg = ""
        try:
            validate_for_scope(tc["params"], tc["scope"], vendor_yaml, tc["drop_params"])
        except InvalidRequestError as e:
            raised = True
            error_type = "InvalidRequestError"
            error_msg = str(e)[:200]
        except TypeError as e:
            raised = True
            error_type = "TypeError"
            error_msg = str(e)[:200]

        entry = {
            "category": tc["category"],
            "description": tc["description"],
            "expected_behavior": tc["expected_behavior"],
            "raised": raised,
            "error_type": error_type,
            "error_msg": error_msg,
            "classification_correct": raised and error_type == "InvalidRequestError",
        }
        results.append(entry)

        status_mark = "PASS" if entry["classification_correct"] else "FAIL"
        print(f"  [{status_mark}] {tc['category']}: raised={raised}, type={error_type}")

    total = len(results)
    passed = sum(1 for r in results if r["classification_correct"])
    print(f"\n  Classification accuracy: {passed}/{total} ({passed/total*100:.0f}%)")

    return results


def experiment_c_observability():
    print("\n" + "=" * 60)
    print("  Experiment (c): Observability Comparison (CNLLM vs LiteLLM)")
    print("=" * 60)

    results = {}

    cnllm_result = _test_cnllm_observability()
    results["cnllm"] = cnllm_result

    litellm_result = _test_litellm_observability()
    results["litellm"] = litellm_result

    comparison = {
        "cnllm_observability": cnllm_result.get("warning_logged", False),
        "litellm_observability": litellm_result.get("warning_logged", False)
            if litellm_result.get("available", False) else "N/A",
        "cnllm_identifies_dropped_param": cnllm_result.get("identifies_dropped_param", False),
        "litellm_identifies_dropped_param": litellm_result.get("identifies_dropped_param", "N/A"),
    }
    results["comparison"] = comparison

    print(f"\n  CNLLM observability: {comparison['cnllm_observability']}")
    print(f"  CNLLM identifies dropped param: {comparison['cnllm_identifies_dropped_param']}")
    print(f"  LiteLLM available: {litellm_result.get('available', False)}")
    if litellm_result.get("available", False):
        print(f"  LiteLLM observability: {comparison['litellm_observability']}")
        print(f"  LiteLLM identifies dropped param: {comparison['litellm_identifies_dropped_param']}")
    else:
        print("  LiteLLM not installed, skipping comparison")

    return results


def _test_cnllm_observability():
    from cnllm import CNLLM

    try:
        client = CNLLM(
            model="deepseek-chat",
            api_key=API_KEY,
            timeout=30,
            max_retries=1,
            drop_params="warn",
        )

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        cnllm_logger = logging.getLogger("cnllm")
        cnllm_logger.addHandler(handler)
        prev_level = cnllm_logger.level
        cnllm_logger.setLevel(logging.WARNING)

        try:
            resp = client.chat.create(
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=10,
                fake_param="test_value",
                nonexistent_option=True,
            )
            logs = buf.getvalue()
            content = resp.still if hasattr(resp, "still") else str(resp)[:80]
            return {
                "available": True,
                "status": "ok",
                "warning_logged": bool(logs.strip()),
                "identifies_dropped_param": "fake_param" in logs or "nonexistent_option" in logs,
                "log_snippet": logs[:300].strip(),
                "content_preview": content[:60],
            }
        finally:
            cnllm_logger.removeHandler(handler)
            cnllm_logger.setLevel(prev_level)

    except Exception as e:
        return {
            "available": True,
            "status": "error",
            "error": str(e)[:200],
        }


def _test_litellm_observability():
    try:
        import litellm
    except ImportError:
        return {"available": False, "reason": "litellm not installed"}

    try:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        litellm_logger = logging.getLogger("litellm")
        litellm_logger.addHandler(handler)
        prev_level = litellm_logger.level
        litellm_logger.setLevel(logging.WARNING)

        try:
            resp = litellm.completion(
                model="deepseek/deepseek-chat",
                api_key=API_KEY,
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=10,
                fake_param="test_value",
                nonexistent_option=True,
            )
            logs = buf.getvalue()
            return {
                "available": True,
                "status": "ok",
                "warning_logged": bool(logs.strip()),
                "identifies_dropped_param": "fake_param" in logs or "nonexistent_option" in logs,
                "log_snippet": logs[:300].strip(),
            }
        finally:
            litellm_logger.removeHandler(handler)
            litellm_logger.setLevel(prev_level)

    except Exception as e:
        return {
            "available": True,
            "status": "error",
            "error_type": type(e).__name__,
            "error_msg": str(e)[:200],
        }


def main():
    all_results = {}

    print("=" * 60)
    print("  EXPERIMENT 4.3: C2 Parameter Validation")
    print("=" * 60)

    print("\n>>> Running Experiment (a): drop_params Three-Level Strategy")
    all_results["exp_a_drop_params"] = experiment_a_drop_params()

    print("\n>>> Running Experiment (b): Parameter Classification Accuracy")
    all_results["exp_b_classification"] = experiment_b_param_classification()

    print("\n>>> Running Experiment (c): Observability Comparison")
    all_results["exp_c_observability"] = experiment_c_observability()

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp43_param_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    exp_a = all_results["exp_a_drop_params"]
    warn_ok = any(r["drop_params"] == "warn" and r["status"] == "call_succeeded" for r in exp_a)
    ignore_ok = any(r["drop_params"] == "ignore" and r["status"] == "call_succeeded" for r in exp_a)
    strict_blocked = any(
        r["drop_params"] == "strict"
        and r["status"] in ("blocked_by_InvalidRequestError", "blocked_by_TypeError")
        for r in exp_a
    )
    type_strict_blocked = any(
        "type_mismatch" in r.get("test", "") and r["drop_params"] == "strict"
        and r["status"] == "blocked_by_TypeError"
        for r in exp_a
    )
    type_warn_ok = any(
        "type_mismatch" in r.get("test", "") and r["drop_params"] == "warn"
        and r["status"] == "call_succeeded_type_mismatch_ignored"
        for r in exp_a
    )
    print(f"  (a) warn: call proceeds = {warn_ok}")
    print(f"  (a) ignore: call proceeds = {ignore_ok}")
    print(f"  (a) strict: call blocked = {strict_blocked}")
    print(f"  (a) type_mismatch+strict: TypeError raised = {type_strict_blocked}")
    print(f"  (a) type_mismatch+warn: warning logged, call proceeds = {type_warn_ok}")

    exp_b = all_results["exp_b_classification"]
    total_b = len(exp_b)
    passed_b = sum(1 for r in exp_b if r["classification_correct"])
    print(f"  (b) Classification accuracy: {passed_b}/{total_b} ({passed_b/total_b*100:.0f}%)")

    exp_c = all_results["exp_c_observability"]
    comp = exp_c.get("comparison", {})
    print(f"  (c) CNLLM observability: {comp.get('cnllm_observability')}")
    print(f"  (c) LiteLLM observability: {comp.get('litellm_observability')}")


if __name__ == "__main__":
    main()
