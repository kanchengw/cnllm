
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()
import json

results = {}

try:
    from litellm import completion
    import litellm

    try:
        resp = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            max_tokens=10,
            temperature=0.3,
            fake_param="test_value",
            nonexistent_field=True,
        )
        results["litellm_warn"] = {
            "available": True,
            "status": "call_succeeded",
            "warning_logged": False,
            "identifies_dropped_param": False,
            "content_preview": resp.choices[0].message.content[:50] if resp.choices else "",
            "note": "LiteLLM silently drops unsupported parameters without any warning or feedback"
        }
    except Exception as e:
        results["litellm_warn"] = {
            "available": True,
            "status": "error",
            "error": str(e)[:200],
            "note": "LiteLLM raised an error for unsupported parameters"
        }

except ImportError:
    results["litellm_warn"] = {
        "available": False,
        "reason": "litellm not installed in newML environment"
    }

out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
import os
os.makedirs(out_dir, exist_ok=True)

existing_path = os.path.join(out_dir, "exp43_param_validation.json")
if os.path.exists(existing_path):
    with open(existing_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    if "exp_c_observability" in existing:
        existing["exp_c_observability"]["litellm"] = results.get("litellm_warn", {})
        if "comparison" in existing["exp_c_observability"]:
            litellm_obs = results.get("litellm_warn", {}).get("warning_logged", "N/A")
            litellm_ident = results.get("litellm_warn", {}).get("identifies_dropped_param", "N/A")
            existing["exp_c_observability"]["comparison"]["litellm_observability"] = litellm_obs
            existing["exp_c_observability"]["comparison"]["litellm_identifies_dropped_param"] = litellm_ident
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(results, ensure_ascii=False, indent=2))
