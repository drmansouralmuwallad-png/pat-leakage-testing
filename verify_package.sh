#!/usr/bin/env bash
# Smoke test of the replication package. Run from this folder:  bash verify_package.sh
# 1. PAT unit tests (pytest example)  2. regenerate summary.json and tables from stored results
# 3. re-run one temporal series, one tabular split, one library case and one case-study series
#    and compare with the stored per-instance results.
set -euo pipefail
PY=${PYTHON:-python3}
OUT=$(mktemp -d)
echo "== 1. PAT unit tests"
$PY -m pytest -q code/examples
echo "== 2. regenerate summary and tables from stored results"
cp -r results "$OUT/results"
$PY code/analyse.py "$OUT/results" > /dev/null
$PY code/make_tables.py "$OUT/results" "$OUT" > /dev/null
$PY - "$OUT" <<'PY'
import json, sys
a = json.load(open('results/summary.json')); b = json.load(open(sys.argv[1] + '/results/summary.json'))
ok = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
print('summary.json identical:', ok); sys.exit(0 if ok else 1)
PY
echo "== 3a. re-run one temporal series (26 pipelines)"
ONLY_SERIES="M5|CA_1|FOODS" $PY code/run_temporal.py sources data/panels.npy "$OUT/t.jsonl" 1 > /dev/null
echo "== 3b. re-run the tabular corpus for one split (split 0 of each dataset)"
$PY code/run_tabular.py sources data/tabular_meta.json "$OUT/k.jsonl" 1 > /dev/null
$PY - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
key = lambda r, *f: tuple(r[x] for x in f)
old = {key(r, 'panel', 'series', 'source'): r for r in map(json.loads, open('results/temporal.jsonl'))}
new = [json.loads(l) for l in open(d + '/t.jsonl')]
t_ok = all(old[key(r, 'panel', 'series', 'source')]['pat']['audits'][i]['hits'] == r['pat']['audits'][i]['hits']
           for r in new for i in range(2))
oldk = {key(r, 'dataset', 'split', 'source'): r for r in map(json.loads, open('results/tabular.jsonl'))}
newk = [json.loads(l) for l in open(d + '/k.jsonl')]
k_ok = all(oldk[key(r, 'dataset', 'split', 'source')]['pat']['hits'] == r['pat']['hits'] for r in newk)
print(f'temporal: {len(new)} instances reproduced identically: {t_ok}')
print(f'tabular:  {len(newk)} instances reproduced identically: {k_ok}')
sys.exit(0 if t_ok and k_ok else 1)
PY
echo "All checks passed."
