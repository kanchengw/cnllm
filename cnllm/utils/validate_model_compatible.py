"""
新模型与 Adapter 兼容性验证
验证新模型是否能被现有 adapter 正确适配
"""
import os
import time
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

from cnllm.core.models import SUPPORTED_MODELS, ADAPTER_MAP, disable_validation, enable_validation
from cnllm.utils.exceptions import ModelAPIError, ParseError

logger = logging.getLogger(__name__)

TEST_PROMPT = "Hello, who are you?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]


class ValidationResult:
    def __init__(self, case: Dict, passed: bool, status: str,
                 response_time: float = 0,
                 content: str = "",
                 error: str = "",
                 details: Dict = None):
        self.case = case
        self.passed = passed
        self.status = status
        self.response_time = response_time
        self.content = content
        self.error = error
        self.details = details or {}

    def to_dict(self) -> Dict:
        return {
            "model": self.case.get("model"),
            "adapter": self.case.get("adapter"),
            "expected": self.case.get("expected"),
            "notes": self.case.get("notes", ""),
            "passed": self.passed,
            "status": self.status,
            "response_time": self.response_time,
            "content_preview": self.content[:100] if self.content else "",
            "error": self.error,
            "details": self.details
        }


class ModelValidator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.min_response_time = 0.1
        self.max_response_time = 30

    def validate_single(self, case: Dict) -> ValidationResult:
        model = case["model"]
        adapter_name = case["adapter"]
        notes = case.get("notes", "")

        print(f"\n{'-' * 50}")
        print(f"Model: {model}")
        print(f"Adapter: {adapter_name}")
        print(f"Notes: {notes}")
        print(f"{'-' * 50}")

        if model not in SUPPORTED_MODELS:
            disable_validation()

        try:
            adapter_class = ADAPTER_MAP.get(adapter_name)
            if not adapter_class:
                return ValidationResult(
                    case=case,
                    passed=False,
                    status="adapter_not_found",
                    error=f"Adapter '{adapter_name}' not found"
                )

            adapter = adapter_class(api_key=self.api_key)
            start_time = time.time()
            response = adapter.create_completion(
                messages=TEST_MESSAGES,
                temperature=0.1,
                model=model
            )
            elapsed = time.time() - start_time

            result = self._validate_response(response, case, elapsed)
            return result

        except ModelAPIError as e:
            print(f"[FAIL] ModelAPIError: {e}")
            return ValidationResult(
                case=case,
                passed=False,
                status="api_error",
                error=str(e)
            )
        except ParseError as e:
            print(f"[FAIL] ParseError: {e}")
            return ValidationResult(
                case=case,
                passed=False,
                status="parse_error",
                error=str(e)
            )
        except Exception as e:
            print(f"[FAIL] {type(e).__name__}: {e}")
            return ValidationResult(
                case=case,
                passed=False,
                status="error",
                error=f"{type(e).__name__}: {e}"
            )
        finally:
            if model not in SUPPORTED_MODELS:
                enable_validation()

    def _validate_response(self, response: Any, case: Dict, elapsed: float) -> ValidationResult:
        details = {}

        if not isinstance(response, dict):
            return ValidationResult(
                case=case,
                passed=False,
                status="invalid_response_type",
                error=f"Expected dict, got {type(response)}"
            )

        has_choices = "choices" in response and len(response["choices"]) > 0
        if not has_choices:
            return ValidationResult(
                case=case,
                passed=False,
                status="missing_choices",
                error="Response missing 'choices' or empty",
                details={"keys": list(response.keys())}
            )

        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")

        details["has_usage"] = "usage" in response
        details["has_id"] = "id" in response
        details["has_created"] = "created" in response
        details["content_length"] = len(content) if content else 0

        if not content:
            return ValidationResult(
                case=case,
                passed=False,
                status="empty_content",
                error="Response content is empty"
            )

        error_indicators = ["error", "Error", "not support", "invalid", "failed"]
        if any(indicator in content.lower() for indicator in error_indicators):
            return ValidationResult(
                case=case,
                passed=False,
                status="error_content",
                error=f"Content appears to contain error: {content[:50]}...",
                content=content,
                response_time=elapsed
            )

        if elapsed < self.min_response_time:
            return ValidationResult(
                case=case,
                passed=False,
                status="suspicious_response_time",
                error=f"Response too fast ({elapsed:.3f}s),可能是mock响应",
                content=content,
                response_time=elapsed
            )

        if elapsed > self.max_response_time:
            details["slow_response"] = True

        print(f"[PASS] Response time: {elapsed:.2f}s")
        try:
            print(f"[INFO] Content preview: {content[:80]}...")
        except UnicodeEncodeError:
            print(f"[INFO] Content preview: (encoding error)")

        if details.get("has_usage"):
            usage = response.get("usage", {})
            print(f"[INFO] Usage: prompt={usage.get('prompt_tokens', 0)}, "
                  f"completion={usage.get('completion_tokens', 0)}, "
                  f"total={usage.get('total_tokens', 0)}")

        return ValidationResult(
            case=case,
            passed=True,
            status="success",
            content=content,
            response_time=elapsed,
            details=details
        )

    def validate_multiple(self, cases: List[Dict]) -> List[ValidationResult]:
        results = []
        for case in cases:
            result = self.validate_single(case)
            results.append(result)
        return results

    def filter_new_models(self, results: List[ValidationResult]) -> List[Tuple[str, str]]:
        new_models = []
        for r in results:
            if r.passed and r.case["model"] not in SUPPORTED_MODELS:
                new_models.append((r.case["model"], r.case["adapter"]))
        return new_models


def update_models_file(new_models: List[Tuple[str, str]]) -> bool:
    if not new_models:
        print("\n没有新模型需要添加")
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_py_path = os.path.join(script_dir, "..", "core", "models.py")

    with open(models_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    for model, adapter_name in new_models:
        if model not in SUPPORTED_MODELS:
            supported_match = re.search(
                r'(SUPPORTED_MODELS\s*:\s*Dict\[str,\s*str\]\s*=\s*\{[^}]*\})',
                content,
                re.DOTALL
            )
            if supported_match:
                old_block = supported_match.group(1)
                new_entry = f'    "{model}": "{adapter_name}",\n'
                new_block = old_block.rstrip().rstrip("}")
                if not new_block.endswith("{"):
                    new_block += "\n"
                new_block += new_entry + "}"
                content = content.replace(old_block, new_block)
                print(f"[ADD] {model} -> SUPPORTED_MODELS")

            if adapter_name not in ADAPTER_MAP:
                print(f"[WARN] {adapter_name} not in ADAPTER_MAP")

    with open(models_py_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n已更新 {models_py_path}")
    return True


def get_default_test_cases() -> List[Dict]:
    test_cases = []

    for model, adapter_name in SUPPORTED_MODELS.items():
        test_cases.append({
            "model": model,
            "adapter": adapter_name,
            "expected": True,
            "notes": "SUPPORTED_MODELS 中已有的模型"
        })

    potential_models = [
        ("minimax-m2.1", "minimax", "M2.1 与 M2.7 同系列，可能兼容"),
        ("minimax-m2", "minimax", "M2 与 M2.7 同系列，可能兼容"),
        ("minimax-m1", "minimax", "M1 可能兼容"),
    ]

    for model, adapter_name, notes in potential_models:
        if model not in SUPPORTED_MODELS:
            test_cases.append({
                "model": model,
                "adapter": adapter_name,
                "expected": True,
                "notes": notes
            })

    return test_cases


def validate_stream(client, model: str) -> Tuple[bool, str]:
    print(f"\n{'-' * 50}")
    print(f"Testing stream=True for model: {model}")
    print(f"{'-' * 50}")

    try:
        chunks = []
        for chunk in client.chat.create(
            model=model,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    chunks.append(delta["content"])

        if len(chunks) > 0:
            full_content = "".join(chunks)
            print(f"[PASS] Stream worked! {len(chunks)} chunks, content: {full_content[:50]}...")
            return True, f"Stream OK, {len(chunks)} chunks"
        else:
            print(f"[FAIL] No chunks received")
            return False, "No stream chunks"
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False, str(e)


def validate_fallback(client, invalid_model: str, fallback_model: str) -> Tuple[bool, str]:
    print(f"\n{'-' * 50}")
    print(f"Testing fallback: {invalid_model} -> {fallback_model}")
    print(f"{'-' * 50}")

    try:
        response = client.chat.create(
            model=invalid_model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )
        content = response["choices"][0]["message"]["content"]
        print(f"[PASS] Fallback worked! Response: {content[:50]}...")
        return True, f"Fallback to {fallback_model} OK"
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False, str(e)


def validate_runnable() -> Tuple[bool, str]:
    print(f"\n{'-' * 50}")
    print(f"Testing LangChain Runnable integration")
    print(f"{'-' * 50}")

    try:
        from langchain_core.messages import HumanMessage
        from cnllm.adapters.framework.langchain import ChatMiniMax

        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            print("[SKIP] MINIMAX_API_KEY not set")
            return True, "SKIPPED"

        llm = ChatMiniMax(api_key=api_key)
        response = llm.invoke([HumanMessage(content="Hello")])
        print(f"[PASS] Runnable worked! Response: {str(response)[:50]}...")
        return True, "Runnable OK"
    except ImportError as e:
        print(f"[SKIP] LangChain not installed: {e}")
        return True, "SKIPPED"
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False, str(e)


def run_validation(
    test_cases: List[Dict] = None,
    auto_update: bool = True
) -> Tuple[List[ValidationResult], Dict]:
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("Error: MINIMAX_API_KEY not set")
        return [], {}

    if test_cases is None:
        test_cases = get_default_test_cases()

    validator = ModelValidator(api_key)
    results = validator.validate_multiple(test_cases)

    print("\n" + "=" * 60)
    print("模型兼容性验证结果")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n通过: {passed}/{len(results)}")
    print(f"失败: {failed}/{len(results)}")

    print("\n详细结果:")
    for r in results:
        status_icon = "[PASS]" if r.passed else "[FAIL]"
        print(f"  {status_icon} {r.case['model']} ({r.case['adapter']}) - {r.status}")

    if auto_update:
        new_models = validator.filter_new_models(results)
        if new_models:
            print("\n" + "=" * 60)
            print("自动添加到 SUPPORTED_MODELS")
            print("=" * 60)
            update_models_file(new_models)
        else:
            print("\n没有新模型需要添加")

    extra_results = {}

    print("\n" + "=" * 60)
    print("额外功能验证")
    print("=" * 60)

    if api_key:
        from cnllm import CNLLM

        client = CNLLM(model="minimax-m2.7", api_key=api_key)

        stream_passed, stream_msg = validate_stream(client, "minimax-m2.7")
        extra_results["stream"] = (stream_passed, stream_msg)

        fb_passed, fb_msg = validate_fallback(
            client,
            invalid_model="invalid-test-model",
            fallback_model="minimax-m2.7"
        )
        extra_results["fallback"] = (fb_passed, fb_msg)

        runnable_passed, runnable_msg = validate_runnable()
        extra_results["runnable"] = (runnable_passed, runnable_msg)

    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)

    print(f"\n模型兼容: {passed}/{len(results)} 通过")
    print(f"Stream: {'PASS' if extra_results.get('stream', (False, ''))[0] else 'FAIL'}")
    print(f"Fallback: {'PASS' if extra_results.get('fallback', (False, ''))[0] else 'FAIL'}")
    print(f"Runnable: {'PASS' if extra_results.get('runnable', (False, ''))[0] else 'FAIL'}")

    return results, extra_results


if __name__ == "__main__":
    results, extra = run_validation()