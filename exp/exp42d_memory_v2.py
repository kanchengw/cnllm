
# CI skip guard: skip if required API keys are not set
if not os.environ.get("CI") and not os.getenv("DEEPSEEK_API_KEY"):
    print("SKIP: missing API keys (set in .env or GitHub secrets)")
    sys.exit(0)
import sys
import os
import json
import time
import gc

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, r"c:\Users\wkc_1\Desktop\Paper\CNLLM")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}


def get_json_size(data, default=0):
    """Calculate JSON serialized size in bytes."""
    try:
        if data:
            return len(json.dumps(data, default=str, ensure_ascii=False))
    except Exception:
        pass
    return default


def get_obj_memory(obj):
    """Rough memory estimate for an object using sys.getsizeof recursively."""
    seen = set()

    def _sizeof(o):
        oid = id(o)
        if oid in seen:
            return 0
        seen.add(oid)
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(_sizeof(k) + _sizeof(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(_sizeof(i) for i in o)
        return size

    return _sizeof(obj)


def measure_single_request_validation():
    """Verify that thinking=True and tools parameters work correctly."""
    from cnllm import CNLLM

    client = CNLLM(
        model="deepseek-chat",
        api_key=API_KEY,
        timeout=30,
        max_retries=1,
        drop_params="ignore"
    )

    print("\n[Validation] Single request with thinking=True and tools")
    print("-" * 50)

    resp = client.chat.create(
        messages=[{"role": "user", "content": "What's the weather in Beijing today?"}],
        thinking=True,
        tools=[WEATHER_TOOL]
    )

    # Extract content and reasoning
    content = ""
    reasoning_content = ""
    tool_calls = []

    # Handle different response structures
    if hasattr(resp, 'choices') and resp.choices:
        choice = resp.choices[0]
        message = choice.message if hasattr(choice, 'message') else choice.get('message', {})

        # Content
        if hasattr(message, 'content'):
            content = message.content or ""
        elif isinstance(message, dict):
            content = message.get('content', '') or ''

        # Reasoning content (different models use different field names)
        for attr in ['reasoning_content', 'thinking', 'reasoning']:
            val = getattr(message, attr, None)
            if val is None and isinstance(message, dict):
                val = message.get(attr)
            if val:
                reasoning_content = val
                break

        # Tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = message.tool_calls
        elif isinstance(message, dict) and message.get('tool_calls'):
            tool_calls = message['tool_calls']

    # Also check CNLLM extra fields
    think_extra = ""
    still_extra = ""
    tools_extra = []

    if hasattr(client.chat, 'think'):
        try:
            t = client.chat.think
            if t:
                think_extra = t
        except Exception:
            pass
    if hasattr(client.chat, 'still'):
        try:
            s = client.chat.still
            if s:
                still_extra = s
        except Exception:
            pass
    if hasattr(client.chat, 'tools'):
        try:
            tl = client.chat.tools
            if tl:
                tools_extra = tl
        except Exception:
            pass

    validation = {
        "content_length": len(content),
        "content_preview": content[:200],
        "reasoning_content_length": len(reasoning_content),
        "reasoning_content_preview": reasoning_content[:200] if reasoning_content else "",
        "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "tool_calls_preview": str(tool_calls)[:200] if tool_calls else "",
        "cnllm_think_length": len(think_extra) if isinstance(think_extra, str) else 0,
        "cnllm_still_length": len(still_extra) if isinstance(still_extra, str) else 0,
        "cnllm_tools_count": len(tools_extra) if isinstance(tools_extra, (list, dict)) else 0,
    }

    print(f"  content_length:        {validation['content_length']}")
    print(f"  reasoning_content_len: {validation['reasoning_content_length']}")
    print(f"  tool_calls_count:      {validation['tool_calls_count']}")
    print(f"  cnllm_think_length:    {validation['cnllm_think_length']}")
    print(f"  cnllm_still_length:    {validation['cnllm_still_length']}")
    print(f"  cnllm_tools_count:     {validation['cnllm_tools_count']}")

    all_valid = True
    if not validation['content_length']:
        print("  WARNING: content is empty")
        all_valid = False
    if not validation['reasoning_content_length'] and not validation['cnllm_think_length']:
        print("  WARNING: reasoning/thinking content is empty")
        all_valid = False
    if not validation['tool_calls_count'] and not validation['cnllm_tools_count']:
        print("  NOTE: no tool calls (model may not have triggered)")

    print(f"  Validation: {'PASSED' if all_valid else 'NEEDS REVIEW'}")

    return validation


def measure_batch_memory_v2(keep_config, n_requests=20):
    """Measure memory using direct object size measurement."""
    from cnllm import CNLLM

    gc.collect()

    client = CNLLM(
        model="deepseek-chat",
        api_key=API_KEY,
        timeout=30,
        max_retries=1,
        drop_params="ignore"
    )

    prompts = [f"What's the weather in city {i}? Brief answer." for i in range(n_requests)]

    kwargs = {
        "prompt": prompts,
        "max_concurrent": 3,
        "rps": 2,
        "thinking": True,
        "tools": [WEATHER_TOOL]
    }

    if keep_config is not None:
        kwargs["keep"] = keep_config

    resp = client.chat.batch(**kwargs)

    # Iterate to trigger completion and field clearing
    for r in resp:
        pass

    # Now measure all field sizes
    results_data = {}
    errors_data = {}
    think_data = {}
    still_data = {}
    tools_data = {}
    raw_data = {}

    # Access through internal attributes to avoid triggering warnings
    results_data = dict(resp._results) if hasattr(resp, '_results') else {}
    errors_data = dict(resp._errors) if hasattr(resp, '_errors') else {}
    think_data = dict(resp._think) if hasattr(resp, '_think') else {}
    still_data = dict(resp._still) if hasattr(resp, '_still') else {}
    tools_data = dict(resp._tools) if hasattr(resp, '_tools') else {}
    raw_data = dict(resp._raw) if hasattr(resp, '_raw') else {}

    # Calculate sizes
    measurements = {
        "results": {
            "json_bytes": get_json_size(results_data),
            "obj_bytes": get_obj_memory(results_data),
            "item_count": len(results_data)
        },
        "errors": {
            "json_bytes": get_json_size(errors_data),
            "obj_bytes": get_obj_memory(errors_data),
            "item_count": len(errors_data)
        },
        "think": {
            "json_bytes": get_json_size(think_data),
            "obj_bytes": get_obj_memory(think_data),
            "item_count": len(think_data),
            "avg_chars": sum(len(str(v)) for v in think_data.values()) // max(len(think_data), 1)
        },
        "still": {
            "json_bytes": get_json_size(still_data),
            "obj_bytes": get_obj_memory(still_data),
            "item_count": len(still_data),
            "avg_chars": sum(len(str(v)) for v in still_data.values()) // max(len(still_data), 1)
        },
        "tools": {
            "json_bytes": get_json_size(tools_data),
            "obj_bytes": get_obj_memory(tools_data),
            "item_count": len(tools_data)
        },
        "raw": {
            "json_bytes": get_json_size(raw_data),
            "obj_bytes": get_obj_memory(raw_data),
            "item_count": len(raw_data)
        }
    }

    # Total BatchResponse object memory
    total_obj_memory = get_obj_memory(resp)

    # Metadata size
    status_data = resp.status if hasattr(resp, 'status') else {}
    usage_data = resp.usage if hasattr(resp, 'usage') else {}
    metadata = {
        "status": status_data,
        "usage": usage_data,
        "keep": str(resp._keep) if hasattr(resp, '_keep') else "",
        "fields_cleared": resp._fields_cleared if hasattr(resp, '_fields_cleared') else False
    }

    # Check if any request triggered tool calls
    tool_call_triggered = any(
        len(v) > 0 for v in tools_data.values()
    ) if tools_data else False

    # Sample content from first successful request
    sample_content = ""
    sample_reasoning = ""
    if results_data:
        first_key = next(iter(results_data))
        first_result = results_data[first_key]
        if isinstance(first_result, dict):
            choices = first_result.get('choices', [])
            if choices:
                msg = choices[0].get('message', {})
                sample_content = msg.get('content', '') or ''
                sample_reasoning = msg.get('reasoning_content', '') or ''
        elif hasattr(first_result, '_data'):
            # BatchResponseItem
            sample_content = first_result._still
            sample_reasoning = first_result._think

    return {
        "keep": str(keep_config) if keep_config is not None else "default",
        "keep_value": keep_config,
        "total_obj_memory_bytes": total_obj_memory,
        "measurements": measurements,
        "metadata": metadata,
        "tool_call_triggered": tool_call_triggered,
        "sample_content_length": len(sample_content),
        "sample_reasoning_length": len(sample_reasoning),
        "n_requests": n_requests,
        "success_count": status_data.get('success_count', 0),
        "fail_count": status_data.get('fail_count', 0)
    }


def print_table_row(label, data):
    """Print a formatted table row."""
    m = data['measurements']
    print(f"  {label}")
    print(f"    Total object memory: {data['total_obj_memory_bytes']:,} bytes ({data['total_obj_memory_bytes']/1024:.1f} KB)")
    print(f"    Fields cleared:      {data['metadata']['fields_cleared']}")
    print(f"    Keep config:         {data['metadata']['keep']}")
    print(f"    Success/Total:       {data['success_count']}/{data['n_requests']}")
    print(f"    Tool call triggered: {data['tool_call_triggered']}")
    print(f"    Sample content len:  {data['sample_content_length']}")
    print(f"    Sample reasoning:    {data['sample_reasoning_length']}")
    print(f"    {'Field':<12} {'JSON bytes':>12} {'Obj bytes':>12} {'Items':>6}")
    print(f"    {'-'*44}")

    total_json = 0
    total_obj = 0
    for field in ['results', 'errors', 'think', 'still', 'tools', 'raw']:
        fm = m[field]
        total_json += fm['json_bytes']
        total_obj += fm['obj_bytes']
        print(f"    {field:<12} {fm['json_bytes']:>12,} {fm['obj_bytes']:>12,} {fm['item_count']:>6}")

    print(f"    {'-'*44}")
    print(f"    {'TOTAL':<12} {total_json:>12,} {total_obj:>12,}")
    print()


def run_experiment():
    """Run the full experiment with 4 keep configurations."""
    all_results = []
    validation_result = None

    print("=" * 60)
    print("  EXPERIMENT 4.2d_v2: Batch Memory with thinking + tools")
    print("  Using direct object size measurement (not tracemalloc)")
    print("=" * 60)

    # Step 1: Validate parameters
    print("\n[Step 1] Validating thinking=True and tools parameters...")
    try:
        validation_result = measure_single_request_validation()
        all_results.append({"validation": validation_result})
    except Exception as e:
        print(f"  ERROR in validation: {e}")
        validation_result = {"error": str(e)}

    # Step 2: Run 4 keep configurations
    print("\n[Step 2] Running batch memory experiment (4 keep configs)...")
    print("-" * 60)

    keep_configs = [
        (None, "default (still, think, tools + metadata)"),
        ([], "[] (metadata only)"),
        (["tools"], "['tools'] (tools + metadata)"),
        (["results"], "['results'] (results + metadata)"),
    ]

    for keep_config, label in keep_configs:
        print(f"\n  Running: keep={label}...")
        try:
            result = measure_batch_memory_v2(keep_config, n_requests=20)
            all_results.append(result)
            print_table_row(f"keep={label}", result)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(2)

    # Step 3: Save results
    out_dir = r"c:\Users\wkc_1\Desktop\Paper\exp\results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp42d_memory_v2.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")

    return all_results, validation_result


def analyze_results(all_results, validation_result):
    """Analyze and print the experiment results."""
    print("\n" + "=" * 60)
    print("  DATA ANALYSIS")
    print("=" * 60)

    # Validation summary
    print("\n[Validation Summary]")
    if validation_result and 'error' not in validation_result:
        print(f"  content_length:        {validation_result.get('content_length', 0)}")
        print(f"  reasoning_content_len: {validation_result.get('reasoning_content_length', 0)}")
        print(f"  tool_calls_count:      {validation_result.get('tool_calls_count', 0)}")
        print(f"  cnllm_think_length:    {validation_result.get('cnllm_think_length', 0)}")
        print(f"  cnllm_still_length:    {validation_result.get('cnllm_still_length', 0)}")
        print(f"  cnllm_tools_count:     {validation_result.get('cnllm_tools_count', 0)}")
    else:
        print("  Validation failed or not available")

    # Memory comparison
    print("\n[Memory Comparison by Keep Config]")
    print(f"  {'Config':<40} {'Total Obj (KB)':>14} {'think (KB)':>11} {'still (KB)':>11} {'tools (KB)':>11} {'results (KB)':>13} {'raw (KB)':>10}")
    print(f"  {'-'*110}")

    results = [r for r in all_results if 'measurements' in r]
    for r in results:
        m = r['measurements']
        config = r['keep']
        total_kb = r['total_obj_memory_bytes'] / 1024
        think_kb = m['think']['obj_bytes'] / 1024
        still_kb = m['still']['obj_bytes'] / 1024
        tools_kb = m['tools']['obj_bytes'] / 1024
        results_kb = m['results']['obj_bytes'] / 1024
        raw_kb = m['raw']['obj_bytes'] / 1024

        print(f"  {config:<40} {total_kb:>14.1f} {think_kb:>11.1f} {still_kb:>11.1f} {tools_kb:>11.1f} {results_kb:>13.1f} {raw_kb:>10.1f}")

    # Analysis of field contributions
    print("\n[Field Contribution Analysis]")
    if results:
        default_result = results[0]
        m = default_result['measurements']
        total_obj = sum(field['obj_bytes'] for field in m.values())

        print(f"  Default config field contributions to total object memory:")
        for field in ['results', 'errors', 'think', 'still', 'tools', 'raw']:
            fm = m[field]
            pct = (fm['obj_bytes'] / total_obj * 100) if total_obj > 0 else 0
            print(f"    {field:<12}: {fm['obj_bytes']:>10,} bytes ({pct:>5.1f}%)")

    # Key findings
    print("\n[Key Findings]")
    if len(results) >= 4:
        default_mem = results[0]['total_obj_memory_bytes']
        empty_mem = results[1]['total_obj_memory_bytes']
        tools_mem = results[2]['total_obj_memory_bytes']
        results_mem = results[3]['total_obj_memory_bytes']

        print(f"  1. Default vs metadata-only: {default_mem:,} vs {empty_mem:,} bytes ({(default_mem-empty_mem)/max(default_mem,1)*100:.0f}% reduction)")
        print(f"  2. Tools-only memory:        {tools_mem:,} bytes")
        print(f"  3. Results-only memory:      {results_mem:,} bytes")
        print(f"  4. think field present:      {results[0]['measurements']['think']['item_count'] > 0}")
        print(f"  5. still field present:      {results[0]['measurements']['still']['item_count'] > 0}")
        print(f"  6. tools field present:      {results[0]['measurements']['tools']['item_count'] > 0}")
        print(f"  7. Tool calls triggered:     {results[0]['tool_call_triggered']}")


def main():
    all_results, validation_result = run_experiment()
    analyze_results(all_results, validation_result)


if __name__ == "__main__":
    main()
