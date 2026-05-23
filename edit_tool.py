"""
安全编辑工具：编辑 -> 验证 -> 备份，单步完成。
禁止直接使用 Write/Edit 工具修改任何代码文件。

用法:
  from edit_tool import edit_file, backup_file, backup_all, prune_backups
  edit_file(path, old_text, new_text, description="简要说明")
  edit_file(path, old_text, new_text, replace_all=True)
  backup_file(path)
  backup_all()
"""
import shutil, os, sys, glob, py_compile, time
from datetime import datetime, timezone, timedelta


def _get_project_root():
    for start in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        candidate = start
        for _ in range(5):
            if os.path.isdir(os.path.join(candidate, 'backups')):
                return candidate
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    return os.getcwd()


def _bak_dir(path):
    project = _get_project_root()
    bdir = os.path.join(project, 'backups', os.path.basename(path))
    os.makedirs(bdir, exist_ok=True)
    return bdir


def _timestamp():
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(time.time(), tz).strftime('%Y%m%d_%H%M%S')


def _prune_one_dir(bak_dir, max_keep=10):
    files = sorted(glob.glob(os.path.join(bak_dir, '*.*')))
    while len(files) > max_keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except PermissionError:
            continue


def prune_backups(max_keep=10):
    project = _get_project_root()
    bak_root = os.path.join(project, 'backups')
    if not os.path.isdir(bak_root):
        return
    for d in os.listdir(bak_root):
        dpath = os.path.join(bak_root, d)
        if os.path.isdir(dpath):
            _prune_one_dir(dpath, max_keep)


def backup_file(path, description=""):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ts = _timestamp()
    bdir = _bak_dir(path)
    bak_path = os.path.join(bdir, f'{os.path.basename(path)}.{ts}')
    shutil.copy2(path, bak_path)
    assert os.path.getsize(path) == os.path.getsize(bak_path), "备份不完整"
    _prune_one_dir(bdir)
    label = description or os.path.basename(path)
    print(f'  BACKUP {label} -> {bak_path}')
    return bak_path


def backup_all():
    project = _get_project_root()
    dirs = ['cnllm/core', 'cnllm/core/accumulators', 'cnllm/core/vendor',
            'cnllm/entry', 'cnllm/utils', 'cnllm']
    count = 0
    for rel_dir in dirs:
        d = os.path.join(project, rel_dir)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, '*.py'))):
            if f.endswith('__init__.py'):
                continue
            try:
                backup_file(f)
                count += 1
            except Exception as e:
                print(f'  SKIP {f}: {e}')
    try:
        backup_file(os.path.join(project, 'edit_tool.py'))
        count += 1
    except Exception:
        pass
    print(f'\n备份完成: {count} 个文件')
    return count


def edit_file(path, old_text, new_text, description="", replace_all=False):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 1. 读取
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 检查匹配次数
    count = content.count(old_text)
    if count == 0:
        raise ValueError(f"旧文本不存在!\n  file={path}\n  text={old_text[:100]}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"旧文本出现 {count} 次, 用 replace_all=True 替换全部,\n"
            f"  或提供更多上下文使 old_text 唯一:\n  file={path}\n  text={old_text[:100]}"
        )

    # 3. 编辑
    new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # 4. 编译验证
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        raise RuntimeError(f"编译失败: {e}")

    # 5. 行数验证
    with open(path, 'r') as f:
        lines = f.readlines()
    last_line = lines[-1].rstrip('\n') if lines else ""
    if len(last_line) < 3 and all(c in ' \t' for c in last_line):
        raise RuntimeError("文件末尾只有空白字符，可能截断")

    # 6. 备份编辑结果（成功后备份，不是编辑前）
    ts = _timestamp()
    bdir = _bak_dir(path)
    bak_path = os.path.join(bdir, f'{os.path.basename(path)}.{ts}')
    shutil.copy2(path, bak_path)
    assert os.path.getsize(path) == os.path.getsize(bak_path), "备份不完整"
    _prune_one_dir(bdir)

    # 7. 完成
    new_lines = len(lines)
    label = description or os.path.basename(path)
    action = "REPLACE_ALL" if replace_all else "EDIT"
    print(f'  {action} {label} ({new_lines} lines) -> {bak_path}')
