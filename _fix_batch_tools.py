import os, sys
os.chdir('/sessions/serene-gracious-ptolemy/mnt/CNLLM')
sys.path.insert(0, '.')
from edit_tool import edit_file

# Fix 1: batch_accumulator.py — handle both Dict and List for _tools[rid]
OLD1 = '''        converted = {}
        for rid, tc_dict in self._tools.items():
            converted[rid] = [dict(tc) for tc in tc_dict.values()]'''
NEW1 = '''        converted = {}
        for rid, tc_dict in self._tools.items():
            if isinstance(tc_dict, dict):
                converted[rid] = [dict(tc) for tc in tc_dict.values()]
            else:
                converted[rid] = [dict(tc) for tc in tc_dict]'''

edit_file(
    'cnllm/core/accumulators/batch_accumulator.py',
    OLD1, NEW1,
    description='fix BatchResponse.tools: handle list values'
)

# Fix 2: test_field_accumulation.py — is_dict -> is_list reference
edit_file(
    'tests/key_needed/test_field_accumulation.py',
    '        print(f"\\n  核查 1: .tools 类型 - {\'\\u2713 PASS\' if is_dict else \'\\u2717 FAIL\'}")',
    '        print(f"\\n  核查 1: .tools 类型 - {\'\\u2713 PASS\' if is_list else \'\\u2717 FAIL\'}")',
    description='fix is_dict -> is_list reference'
)

print('done')
