"""Compute every quantity reported for RQ1-RQ5 from the result files and write
results/summary.json. Nothing in the manuscript is typed by hand: tables are produced by
make_tables.py from summary.json.

usage: python analyse.py <results_dir>
"""
import collections
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from corpus_temporal import OPERATORS as T_OPS, MULTI_STAGE_CONFIGS   # noqa: E402
from corpus_tabular import OPERATORS as K_OPS                         # noqa: E402

R = sys.argv[1]


def jl(name):
    return [json.loads(l) for l in open(os.path.join(R, name))]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def rate(flags):
    flags = [bool(f) for f in flags]
    p, lo, hi = wilson(sum(flags), len(flags))
    return dict(k=sum(flags), n=len(flags), rate=p, lo=lo, hi=hi)


def mcnemar(a, b):
    """Exact McNemar test on paired detections a (PAT) and b (baseline)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return dict(pat_only=only_a, baseline_only=only_b, p=p)


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


out = {}
# ------------------------------------------------------------------ temporal corpus
T = jl('temporal.jsonl')
S = json.load(open(os.path.join(R, 'static.json')))
DCt = {d['file']: d for d in jl('deepchecks_temporal.jsonl')}
subset = set(open(os.path.join(R, 'dc_subset_series.txt')).read().split())

clean_shift = [r['shift_stat'] for r in T if r['admissible'] and r['shift_stat'] is not None]
tau_star = float(np.quantile(clean_shift, 0.95))
out['shift_tau_star'] = tau_star

for r in T:
    key = f"temporal/{r['panel']}/{r['source']}.py"
    r['toc'] = bool(S[key]['toc_flagged'])
    r['yang'] = bool(S[key]['yang']['flagged'])
    r['yang_ok'] = bool(S[key]['yang']['ok'])
    r['shift1'] = r['shift_stat'] is not None and r['shift_stat'] >= 0.01
    r['shiftc'] = r['shift_stat'] is not None and r['shift_stat'] > tau_star
    r['pat_flag'] = r['pat']['flagged']
    r['equivalent'] = (r['op'] is not None and not r['pat_flag']
                       and not r.get('oracle_dependent', True))
    r['in_subset'] = f"{r['panel']}|{r['series']}" in subset
    dc = DCt.get(f"{r['panel']}__{r['series']}__{r['source']}.npz")
    r['dc'] = None if dc is None or dc.get('flagged') is None else bool(dc['flagged'])
    r['dc_seconds'] = None if dc is None else dc.get('seconds')
    op = r['op']
    r['mechanism'] = 'admissible' if op is None else (
        'multi-stage' if r['config'] in MULTI_STAGE_CONFIGS else T_OPS[op]['mechanism'])

TOOLS_T = [('pat_flag', 'PAT'), ('toc', 'Time-offset check'), ('shift1', 'Shift check (1%)'),
           ('shiftc', 'Shift check (calibrated)'), ('yang', 'Static analyser (Yang et al.)'),
           ('dc', 'Deepchecks')]
out['yang_failures'] = {k: v['yang']['tail'] for k, v in S.items() if not v['yang']['ok']}
out['n_sources'] = dict(temporal=len({(r['panel'], r['source']) for r in T}))


def summarise(rows, tools):
    clean = [r for r in rows if r['admissible']]
    mut = [r for r in rows if not r['admissible'] and not r['equivalent']]
    eq = [r for r in rows if r['equivalent']]
    res = dict(n_clean=len(clean), n_mutants=len(mut), n_equivalent=len(eq), tools={})
    for k, name in tools:
        c = [r[k] for r in clean if r[k] is not None]
        m = [r[k] for r in mut if r[k] is not None]
        e = [r[k] for r in eq if r[k] is not None]
        res['tools'][name] = dict(tpr=rate(m), fpr=rate(c), flags_on_equivalent=rate(e))
        if k != 'pat_flag':
            pairs = [(r['pat_flag'], r[k]) for r in mut if r[k] is not None]
            mc = mcnemar([a for a, _ in pairs], [b for _, b in pairs])
            pa = np.mean([a for a, _ in pairs]); pb = np.mean([b for _, b in pairs])
            mc['cohen_h'] = cohen_h(pa, pb)
            res['tools'][name]['vs_pat'] = mc
    return res


out['rq1_temporal_full'] = summarise(T, [t for t in TOOLS_T if t[0] != 'dc'])
out['rq1_temporal_subset'] = summarise([r for r in T if r['in_subset']], TOOLS_T)

# per operator (RQ2)
ops = sorted({(r['config'], r['op']) for r in T if r['op']})
per_op = []
for cfg, op in ops:
    rows = [r for r in T if r['op'] == op and r['config'] == cfg]
    mut = [r for r in rows if not r['equivalent']]
    sub = [r for r in mut if r['in_subset']]
    rec = dict(config=cfg, op=op, taxonomy=T_OPS[op]['taxonomy'], mechanism=rows[0]['mechanism'],
               n=len(rows), n_equivalent=sum(r['equivalent'] for r in rows),
               pat=rate([r['pat_flag'] for r in mut]),
               pat_b1_s025=rate([r['pat_b1_s025'] for r in mut]),
               toc=bool(rows[0]['toc']) if len({r['toc'] for r in rows}) == 1 else 'mixed',
               yang=rate([r['yang'] for r in mut]),
               shiftc=rate([r['shiftc'] for r in mut]),
               dc=rate([r['dc'] for r in sub if r['dc'] is not None]))
    per_op.append(rec)
out['rq2_per_operator'] = per_op

by_mech = collections.defaultdict(list)
for r in T:
    if r['op'] and not r['equivalent']:
        by_mech[r['mechanism']].append(r)
out['rq2_by_mechanism'] = {m: {name: rate([r[k] for r in rows if r[k] is not None])
                               for k, name in TOOLS_T if k != 'dc'}
                           for m, rows in by_mech.items()}
out['rq2_by_mechanism_subset'] = {m: rate([r['dc'] for r in rows if r['in_subset'] and r['dc'] is not None])
                                  for m, rows in by_mech.items()}

# per-feature localisation (PAT), where the defective columns are defined by construction
loc = collections.defaultdict(lambda: [0, 0, 0])   # tp, fp, fn over columns
for r in T:
    if not r['op'] or r['equivalent'] or not r['pat_flag']:
        continue
    gt = T_OPS[r['op']]['cols']
    if gt is None or gt == 'ALL':
        continue
    flagged = {c for a in r['pat']['audits'] for c in a['flagged']}
    names = {c.split(':', 1)[1] if not c.startswith('y_train') else 'y_train:target' for c in flagged}
    gtset = set(gt)
    loc[r['op']][0] += len(names & gtset)
    loc[r['op']][1] += len(names - gtset)
    loc[r['op']][2] += len(gtset - names)
out['rq2_localisation'] = {op: dict(tp=v[0], fp=v[1], fn=v[2],
                                    precision=v[0] / max(1, v[0] + v[1]),
                                    recall=v[0] / max(1, v[0] + v[2])) for op, v in loc.items()}
tp = sum(v[0] for v in loc.values()); fp = sum(v[1] for v in loc.values()); fn = sum(v[2] for v in loc.values())
out['rq2_localisation_total'] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp), recall=tp / (tp + fn))

# per pipeline source: a dynamic tool detects a source if it flags at least one instance
src = collections.defaultdict(list)
for r in T:
    src[(r['panel'], r['config'], r['op'])].append(r)
per_src = {'clean': 0, 'mutant': 0, 'pat_any': 0, 'pat_all': 0, 'pat_clean_any': 0}
for (panel, cfg, op), rows in src.items():
    if op is None:
        per_src['clean'] += 1
        per_src['pat_clean_any'] += any(r['pat_flag'] for r in rows)
    else:
        nonq = [r for r in rows if not r['equivalent']]
        if not nonq:
            continue
        per_src['mutant'] += 1
        per_src['pat_any'] += any(r['pat_flag'] for r in nonq)
        per_src['pat_all'] += all(r['pat_flag'] for r in nonq)
out['rq1_per_source_temporal'] = per_src

# RQ3: sensitivity for piecewise defects (grid) -- detection with the first b draws
grid = collections.defaultdict(lambda: collections.defaultdict(list))
for r in T:
    if 'grid' not in r or r['equivalent']:
        continue
    for s, audits in r['grid'].items():
        for b in (1, 2, 5, 10):
            grid[(r['op'], s)][b].append(any(any(h[:b]) for h in audits))
out['rq3_grid'] = {f'{op}|{s}': {str(b): rate(v) for b, v in d.items()} for (op, s), d in grid.items()}
# margin for tolerance: smallest maximal relative change among detected audits, per mechanism
marg = collections.defaultdict(list)
for r in T:
    if r['op'] and r['pat_flag']:
        m = max(a['mrc'] for a in r['pat']['audits'])
        marg[r['mechanism']].append(m)
out['rq3_margin'] = {m: dict(min=float(np.min(v)), p05=float(np.quantile(v, 0.05)),
                             median=float(np.median(v))) for m, v in marg.items()}
# cost per instance
out['rq3_cost_temporal'] = dict(
    pat_mean_s=float(np.mean([r['pat']['seconds'] for r in T])),
    pat_median_s=float(np.median([r['pat']['seconds'] for r in T])),
    pat_runs=int(np.median([r['pat']['n_runs'] for r in T])),
    pipeline_mean_s=float(np.mean([r['t_base'] for r in T])),
    shift_mean_s=float(np.mean([r['t_shift'] for r in T])),
    dc_mean_s=float(np.mean([r['dc_seconds'] for r in T if r['dc_seconds']])),
    yang_mean_s=float(np.mean([v['yang']['seconds'] for k, v in S.items() if k.startswith('temporal')])))

# ------------------------------------------------------------------ tabular corpus
K = jl('tabular.jsonl')
DCk = {d['file']: d for d in jl('deepchecks_tabular.jsonl')}
for r in K:
    key = f"tabular/{r['dataset']}/{r['source']}.py"
    r['yang'] = bool(S[key]['yang']['flagged'])
    r['pat_flag'] = r['pat']['flagged']
    r['equivalent'] = r['op'] is not None and not r['pat_flag'] and not r.get('oracle_dependent', True)
    dc = DCk.get(f"{r['dataset']}__{r['split']}__{r['source']}.npz")
    r['dc'] = None if dc is None or dc.get('flagged') is None else bool(dc['flagged'])
    r['dc_seconds'] = None if dc is None else dc.get('seconds')
TOOLS_K = [('pat_flag', 'PAT'), ('yang', 'Static analyser (Yang et al.)'), ('dc', 'Deepchecks')]
out['rq1_tabular'] = summarise(K, TOOLS_K)
per_op_k = []
for op in sorted({r['op'] for r in K if r['op']}):
    rows = [r for r in K if r['op'] == op]
    mut = [r for r in rows if not r['equivalent']]
    per_op_k.append(dict(op=op, taxonomy=K_OPS[op]['taxonomy'], mechanism=K_OPS[op]['mechanism'],
                         label_free=K_OPS[op]['label_free'], n=len(rows),
                         n_equivalent=sum(r['equivalent'] for r in rows),
                         pat=rate([r['pat_flag'] for r in mut]), yang=rate([r['yang'] for r in mut]),
                         dc=rate([r['dc'] for r in mut if r['dc'] is not None])))
out['rq2_per_operator_tabular'] = per_op_k
out['rq3_cost_tabular'] = dict(
    pat_mean_s=float(np.mean([r['pat']['seconds'] for r in K])),
    pipeline_mean_s=float(np.mean([r['t_base'] for r in K])),
    dc_mean_s=float(np.mean([r['dc_seconds'] for r in K if r['dc_seconds']])),
    yang_mean_s=float(np.mean([v['yang']['seconds'] for k, v in S.items() if k.startswith('tabular')])))

# ------------------------------------------------------------------ library audit (RQ4)
L = jl('library.jsonl')
cases = collections.defaultdict(list)
for r in L:
    cases[(r['kind'], r['case'], r.get('pattern', ''))].append(r)
lib = []
for (kind, case, pat), rows in cases.items():
    ok = [r for r in rows if 'error' not in r]
    key = f"library/T__{case}.py" if kind == 'temporal' else f"library/K__{case}__{pat}.py"
    lib.append(dict(kind=kind, case=case, pattern=pat, library=rows[0]['library'],
                    expected_admissible=rows[0]['expected_admissible'],
                    reason=rows[0].get('reason', ''), uses_labels=rows[0].get('uses_labels'),
                    n=len(ok), n_error=len(rows) - len(ok),
                    pat_flagged=sum(r['flagged'] for r in ok), yang=bool(S[key]['yang']['flagged'])))
out['rq4_library'] = lib
adm = [x for x in lib if x['expected_admissible']]
inad = [x for x in lib if not x['expected_admissible']]
out['rq4_summary'] = dict(
    n_cases=len(lib), n_temporal=sum(x['kind'] == 'temporal' for x in lib),
    n_tabular_cases=len({x['case'] for x in lib if x['kind'] == 'tabular'}),
    adm_runs=sum(x['n'] for x in adm), adm_false_alarms=sum(x['pat_flagged'] for x in adm),
    inad_runs=sum(x['n'] for x in inad), inad_detected=sum(x['pat_flagged'] for x in inad),
    inad_cases=len(inad), inad_cases_all=sum(x['pat_flagged'] == x['n'] for x in inad),
    inad_cases_none=sum(x['pat_flagged'] == 0 for x in inad),
    yang_tp=sum(x['yang'] for x in inad), yang_fp=sum(x['yang'] for x in adm),
    n_adm_cases=len(adm), errors=sum(x['n_error'] for x in lib))

# ------------------------------------------------------------------ case study (RQ5)
C = json.load(open(os.path.join(R, 'case_study.json')))
V = ['v0_as_built', 'v1_rolling_fixed', 'v2_twin_fixed', 'v2u_unseeded']
cs = {}
for v in V:
    cols = collections.Counter(c for s in C for c in C[s][v]['pat_cols'])
    nd = collections.Counter(c.split(':', 1)[1] for s in C for c in C[s][v]['nondeterministic_cols'])
    rec = dict(flagged=sum(C[s][v]['pat_flagged'] for s in C), n=len(C), cols=dict(cols),
               nondet_series=sum(bool(C[s][v]['nondeterministic_cols']) for s in C),
               nondet_cols=sorted(nd), pat_seconds=float(np.mean([C[s][v]['pat_seconds'] for s in C])))
    if v != 'v2u_unseeded':
        for m in ('ridge', 'lgbm'):
            w = np.array([C[s][v]['rmse_with_twin'][m] for s in C])
            wo = np.array([C[s][v]['rmse_without_twin'][m] for s in C])
            eff = float(np.exp(np.mean(np.log(np.maximum(w, 1e-12) / np.maximum(wo, 1e-12)))) - 1)
            rec[m] = dict(rmse_with=float(w.mean()), rmse_without=float(wo.mean()),
                          twin_effect=eff, twin_better=int((w < wo).sum()),
                          p=float(wilcoxon(w, wo).pvalue) if np.any(w != wo) else None)
    cs[v] = rec
for m in ('ridge', 'lgbm'):
    a = np.array([C[s]['v0_as_built']['rmse_with_twin'][m] for s in C])
    b = np.array([C[s]['v2_twin_fixed']['rmse_with_twin'][m] for s in C])
    cs[f'inflation_{m}'] = dict(v0=float(a.mean()), v2=float(b.mean()),
                                ratio_geo=float(np.exp(np.mean(np.log(np.maximum(a, 1e-12) / b))) - 1))
out['rq5_case'] = cs

# ------------------------------------------------------------------ misses in controlled benchmark
M = json.load(open(os.path.join(R, 'misses_nv.json')))
out['rq2_nv_misses'] = dict(n=len(M), min_margin_sd=[v['min_margin_sd'] for v in M.values()],
                            all_rows_below=all(v['share_rows_all_future_below'] == 1.0 for v in M.values()),
                            b5_scale2=sum(v['b5_scale2'] for v in M.values()),
                            b5_scale3=sum(v['b5_scale3'] for v in M.values()),
                            b20_scale3=sum(v['b20_scale3'] for v in M.values()))
if os.path.exists(os.path.join(R, 'scalability.json')):
    out['rq3_scalability'] = json.load(open(os.path.join(R, 'scalability.json')))

json.dump(out, open(os.path.join(R, 'summary.json'), 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('rq1_temporal_full', 'rq1_tabular')}, indent=1, default=float)[:4000])
