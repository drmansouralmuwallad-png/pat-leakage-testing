"""Static baselines, which are data-independent and run once per pipeline source.

1. Time-offset check (TOC). A syntactic version of the time-stamp principle of Kaufman et al.
   (2012): every value derived from the target series must be read at least one step in the
   past. The check flags
     R1  a .shift(k) call whose offset can be 0 or negative (literal or loop-range offsets);
     R2  a window or difference operation (rolling, ewm, expanding, diff, pct_change) applied
         directly to a target-series variable, with no shift in between.
   Target-series variables are names assigned from df['y'] or from expressions whose root is
   such a variable.

2. The data-flow leakage analyser of Yang et al. (2022), run unmodified (commit a7d038b) with
   its customised pyright and Souffle 2.5. A source is flagged when the analyser reports a
   pre-processing, overlap or multi-test leak.

usage: python static_baselines.py <sources_root> <out.json>
"""
import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

LA_DIR = os.environ.get('LEAKAGE_ANALYSIS_DIR', '/home/claude/leakage-analysis')
LA_PY = os.environ.get('LEAKAGE_ANALYSIS_PYTHON', '/home/claude/la-venv/bin/python')
WINDOW_OPS = {'rolling', 'ewm', 'expanding', 'diff', 'pct_change'}


def _const_env(tree):
    env = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int)):
            env[node.targets[0].id] = node.value.value
    return env


def _eval(node, env):
    try:
        return eval(compile(ast.Expression(node), '<e>', 'eval'), {'__builtins__': {}}, dict(env))
    except Exception:
        return None


def _root_name(node):
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _reads_target_column(node):
    for n in ast.walk(node):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'df'
                and isinstance(n.slice, ast.Constant) and n.slice.value == 'y'):
            return True
    return False


def time_offset_check(src):
    tree = ast.parse(src)
    env = _const_env(tree)
    targets = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            if _reads_target_column(node.value) or _root_name(node.value) in targets:
                targets.add(node.targets[0].id)
    loop_vals = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == 'range'):
            args = [_eval(a, env) for a in node.iter.args]
            if all(isinstance(a, int) for a in args):
                loop_vals[node.target.id] = list(range(*args))
    findings = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr = node.func.attr
        if attr == 'shift' and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in loop_vals:
                vals = loop_vals[arg.id]
            else:
                v = _eval(arg, env)
                vals = [v] if isinstance(v, int) else []
            if vals and min(vals) <= 0:
                findings.append(('R1', node.lineno))
        if attr in WINDOW_OPS and isinstance(node.func.value, ast.Name) \
                and node.func.value.id in targets:
            findings.append(('R2', node.lineno))
    return findings


def yang_analyser(path):
    """Run the analyser of Yang et al. on a copy of the script. Returns (findings, seconds)."""
    tmp = tempfile.mkdtemp(prefix='la_')
    dst = os.path.join(tmp, 'pipeline.py')
    shutil.copy(path, dst)
    t0 = time.perf_counter()
    proc = subprocess.run([LA_PY, '-m', 'src.main', dst], cwd=LA_DIR, capture_output=True,
                          text=True, timeout=900)
    secs = time.perf_counter() - t0
    fact = os.path.join(tmp, 'pipeline-fact')
    ok = 'Success!' in proc.stdout
    found = {}
    for kind, fname in [('preprocessing', 'Telemetry_PreProcessingLeak.csv'),
                        ('overlap', 'Telemetry_OverlapLeak.csv'),
                        ('overlap_final', 'FinalOverlapLeak.csv'),
                        ('multitest', 'Telemetry_MultiUseTestLeak.csv')]:
        p = os.path.join(fact, fname)
        n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        found[kind] = n
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(ok=ok, counts=found, flagged=ok and any(found.values()), seconds=secs,
                tail=proc.stdout.strip().splitlines()[-1:] if not ok else [])


if __name__ == '__main__':
    root, out = sys.argv[1:3]
    res = json.load(open(out)) if os.path.exists(out) else {}
    paths = sorted(glob.glob(os.path.join(root, 'temporal', '*', '*.py')) +
                   glob.glob(os.path.join(root, 'tabular', '*', '*.py')) +
                   glob.glob(os.path.join(root, 'library', '*.py')))
    for i, p in enumerate(paths):
        key = os.path.relpath(p, root)
        if key in res:
            continue
        src = open(p).read()
        toc = time_offset_check(src) if key.startswith('temporal') else None
        ya = yang_analyser(p)
        res[key] = dict(toc_findings=toc, toc_flagged=(bool(toc) if toc is not None else None),
                        yang=ya)
        json.dump(res, open(out, 'w'), indent=1)
        print(i + 1, len(paths), key, 'TOC', toc, 'YANG', ya['flagged'], ya['counts'],
              round(ya['seconds'], 1), flush=True)
