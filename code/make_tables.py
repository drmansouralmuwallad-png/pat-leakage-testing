"""LaTeX tables for the IST manuscript, generated from results/summary.json.

usage: python make_tables.py <results_dir> <paper_dir>
"""
import collections
import json
import os
import sys

R, P = sys.argv[1:3]
S = json.load(open(os.path.join(R, 'summary.json')))


def pct(x, d=1):
    return '--' if x is None else f'{100 * x:.{d}f}'


def rci(r):
    if not r['n']:
        return '--'
    return f"{pct(r['rate'])} [{pct(r['lo'])}, {pct(r['hi'])}]"


def frac(r):
    return f"{r['k']}/{r['n']}"


def pval(p):
    if p is None:
        return '--'
    if p < 1e-4:
        return '$<10^{-4}$'
    return f'{p:.3f}'


def write(name, body):
    open(os.path.join(P, name), 'w').write(body)


# ---------------------------------------------------------------- RQ1
ORDER = ['PAT', 'Time-offset check', 'Shift check (1%)', 'Shift check (calibrated)',
         'Static analyser (Yang et al.)', 'Deepchecks']
LABEL = {'PAT': 'PAT (this paper)', 'Time-offset check': 'Time-offset check',
         'Shift check (1%)': 'Shift check, $\\tau=1\\%$',
         'Shift check (calibrated)': 'Shift check, calibrated $\\tau^{*}$',
         'Static analyser (Yang et al.)': 'Static analyser \\cite{yang2022notebooks}',
         'Deepchecks': 'Deepchecks \\cite{chorev2022deepchecks}'}
lines = []
for sec, title in [('rq1_temporal_subset', 'Temporal pipelines, common subset'),
                   ('rq1_temporal_full', 'Temporal pipelines, all instances'),
                   ('rq1_tabular', 'Tabular pipelines')]:
    d = S[sec]
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}: {d['n_mutants']} leaky and "
                 f"{d['n_clean']} clean instances, {d['n_equivalent']} equivalent mutants excluded}} \\\\")
    for t in ORDER:
        if t not in d['tools']:
            continue
        v = d['tools'][t]
        j = v['tpr']['rate'] - v['fpr']['rate']
        mc = v.get('vs_pat')
        mcs = '--' if mc is None else f"{mc['pat_only']}/{mc['baseline_only']}, {pval(mc['p'])}"
        lines.append(f"\\quad {LABEL[t]} & {rci(v['tpr'])} & {rci(v['fpr'])} & {j:.2f} & "
                     f"{frac(v['flags_on_equivalent'])} & {mcs} \\\\")
    lines.append('\\addlinespace[3pt]')
write('tab_rq1.tex', '\\begin{tabular}{@{}l c c c c c@{}}\n\\toprule\n'
      'Technique & \\makecell{Detection rate\\\\(\\%) [95\\% CI]} & \\makecell{False-alarm rate\\\\(\\%) [95\\% CI]} & '
      '$J$ & \\makecell{Flags on\\\\equivalent} & \\makecell{McNemar vs PAT\\\\(PAT only/other only, $p$)} \\\\\n'
      '\\midrule\n' + '\n'.join(lines[:-1]) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------------------------------------------------------- RQ2 temporal operators
MECH_ORDER = ['direct', 'hidden look-ahead', 'conditional', 'weak (global statistic)',
              'nonlinear (piecewise)', 'multi-stage']
rows = sorted(S['rq2_per_operator'], key=lambda r: (MECH_ORDER.index(r['mechanism']), r['op'], r['config']))
L = []
prev = None
for r in rows:
    if r['mechanism'] != prev:
        if prev is not None:
            L.append('\\addlinespace[2pt]')
        L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{r['mechanism'].capitalize()}}}}} \\\\")
        prev = r['mechanism']
    op = r['op'].replace('_', '\\_')
    cfg = r['config'].split('_')[0]
    toc = 'yes' if r['toc'] is True else ('no' if r['toc'] is False else 'mixed')
    L.append(f"\\quad {op} & {cfg} & {r['taxonomy']} & {r['n_equivalent']} & {pct(r['pat']['rate'])} & "
             f"{pct(r['pat_b1_s025']['rate'])} & {toc} & {pct(r['shiftc']['rate'])} & "
             f"{pct(r['yang']['rate'])} & {pct(r['dc']['rate'])} \\\\")
write('tab_ops.tex', '\\begin{tabular}{@{}l c c c c c c c c c@{}}\n\\toprule\n'
      '& & & & \\multicolumn{2}{c}{PAT} & & & & \\\\ \\cmidrule(lr){5-6}\n'
      'Operator & Config. & Taxon. & Equiv. & \\makecell{$B{=}5$,\\\\scale 2} & \\makecell{$B{=}1$,\\\\scale 0.25} & '
      'TOC & \\makecell{Shift,\\\\$\\tau^{*}$} & Static & \\makecell{Deep-\\\\checks} \\\\\n\\midrule\n'
      + '\n'.join(L).replace('\\multicolumn{9}', '\\multicolumn{10}') + '\n\\bottomrule\n\\end{tabular}\n')

# tabular operators
L = []
for r in sorted(S['rq2_per_operator_tabular'], key=lambda r: (r['taxonomy'], r['op'])):
    L.append(f"{r['op'].replace('_', chr(92) + '_')} & {r['taxonomy']} & {r['mechanism']} & "
             f"{'yes' if r['label_free'] else 'no'} & {r['pat']['n']} & {pct(r['pat']['rate'])} & "
             f"{pct(r['yang']['rate'])} & {pct(r['dc']['rate'])} \\\\")
write('tab_ops_tab.tex', '\\begin{tabular}{@{}l c l c c c c c@{}}\n\\toprule\n'
      'Operator & Taxon. & Mechanism & Label-free & $n$ & PAT & Static & Deepchecks \\\\\n\\midrule\n'
      + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------------------------------------------------------- RQ3 grid (supplement)
L = []
for op in ['FLAG_ALL', 'WINS_ALL', 'SEL_ALL']:
    for s in ['0.5', '1.0', '2.0', '3.0']:
        d = S['rq3_grid'][f'{op}|{s}']
        L.append(f"{op.replace('_', chr(92) + '_')} & {float(s):g} & " +
                 ' & '.join(pct(d[b]['rate']) for b in ['1', '2', '5', '10']) +
                 f" & {d['1']['n']} \\\\")
    L.append('\\addlinespace[2pt]')
write('tab_grid.tex', '\\begin{tabular}{@{}l c c c c c c@{}}\n\\toprule\n'
      'Operator & Scale & $B=1$ & $B=2$ & $B=5$ & $B=10$ & Instances \\\\\n\\midrule\n'
      + '\n'.join(L[:-1]) + '\n\\bottomrule\n\\end{tabular}\n')

# scalability
L = []
for r in S['rq3_scalability']:
    L.append(f"{r['T']:,} & {r['p']} & {1000 * r['t_pipeline']:.0f} & {r['t_pat']:.2f} & "
             f"{r['t_pat'] / r['t_pipeline']:.1f} & {r['t_shift']:.2f} \\\\".replace(',', '{,}'))
write('tab_scal.tex', '\\begin{tabular}{@{}r r r r r r@{}}\n\\toprule\n'
      '$T$ & $p$ & \\makecell{Pipeline run\\\\(ms)} & \\makecell{PAT audit\\\\(s)} & '
      '\\makecell{PAT / run} & \\makecell{Shift check\\\\(s)} \\\\\n\\midrule\n' + '\n'.join(L) +
      '\n\\bottomrule\n\\end{tabular}\n')

# ---------------------------------------------------------------- RQ4 library
lib = S['rq4_library']
by = collections.defaultdict(lambda: dict(cases=0, adm=0, adm_ok=0, inad=0, inad_all=0,
                                           inad_some=0, inad_none=0, yang_tp=0, yang_fp=0))
for x in lib:
    key = (x['kind'], x['library'])
    b = by[key]
    b['cases'] += 1
    if x['expected_admissible']:
        b['adm'] += 1
        b['adm_ok'] += x['pat_flagged'] == 0
        b['yang_fp'] += x['yang']
    else:
        b['inad'] += 1
        b['inad_all'] += x['pat_flagged'] == x['n']
        b['inad_some'] += 0 < x['pat_flagged'] < x['n']
        b['inad_none'] += x['pat_flagged'] == 0
        b['yang_tp'] += x['yang']
L = []
for (kind, libname), b in sorted(by.items()):
    L.append(f"{kind} & {libname.replace('_', chr(92) + '_')} & {b['cases']} & {b['adm_ok']}/{b['adm']} & "
             f"{b['inad_all']}/{b['inad_some']}/{b['inad_none']} & {b['yang_tp']}/{b['inad']} & "
             f"{b['yang_fp']}/{b['adm']} \\\\")
write('tab_lib.tex', '\\begin{tabular}{@{}l l c c c c c@{}}\n\\toprule\n'
      '& & & \\multicolumn{2}{c}{PAT} & \\multicolumn{2}{c}{Static analyser} \\\\ '
      '\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n'
      'Setting & Library & Cases & \\makecell{Admissible:\\\\never flagged} & '
      '\\makecell{Inadmissible: flagged\\\\in all/some/no runs} & \\makecell{Inadm.\\\\flagged} & '
      '\\makecell{Adm.\\\\flagged} \\\\\n\\midrule\n' + '\n'.join(L) + '\n\\bottomrule\n\\end{tabular}\n')
# full library table (supplement)
L = []
for x in sorted(lib, key=lambda x: (x['kind'], x['library'], x['case'], x['pattern'])):
    name = x['case'].replace('_', '\\_') + (f" ({x['pattern']})" if x['pattern'] else '')
    L.append(f"{x['kind']} & {x['library'].replace('_', chr(92) + '_')} & {name} & {'yes' if x['expected_admissible'] else 'no'} & "
             f"{x['pat_flagged']}/{x['n']} & {'yes' if x['yang'] else 'no'} & "
             f"{x['reason'] if x['kind'] == 'temporal' else ('fitted on training rows' if x['pattern'] == 'B' else 'fitted on all rows' + (', uses labels' if x['uses_labels'] else ''))} \\\\")
write('tab_lib_full.tex', '\\begin{longtable}{@{}l l l c c c p{5.2cm}@{}}\n\\caption{Every library-transformer case: documented behaviour, PAT verdicts and the static analyser.}\\\\\n\\toprule\n'
      'Setting & Library & Case & Adm. & PAT & Static & Documented behaviour \\\\\n\\midrule\n\\endhead\n'
      + '\n'.join(L) + '\n\\bottomrule\n\\end{longtable}\n')

# ---------------------------------------------------------------- RQ5 case study
C = S['rq5_case']
names = {'v0_as_built': 'v0: as built', 'v1_rolling_fixed': 'v1: rolling statistics shifted',
         'v2_twin_fixed': 'v2: twin state read at $t-1$', 'v2u_unseeded': 'v2u: v2, simulator unseeded'}
L = []
for v in ['v0_as_built', 'v1_rolling_fixed', 'v2_twin_fixed', 'v2u_unseeded']:
    c = C[v]
    cols = ', '.join(f"{k.replace('_', chr(92) + '_')} ({n})" for k, n in sorted(c['cols'].items(), key=lambda kv: -kv[1]))
    if not cols:
        cols = 'none'
    nd = f"{c['nondet_series']}/30"
    if v != 'v2u_unseeded':
        lg = c['lgbm']
        acc = (f"{lg['rmse_without']:.1f} & {lg['rmse_with']:.1f} & {100 * lg['twin_effect']:+.1f} & "
               f"{lg['twin_better']}/30 & {pval(lg['p'])}")
    else:
        acc = '-- & -- & -- & -- & --'
    L.append(f"{names[v]} & {c['flagged']}/30 & {nd} & {acc} \\\\")
    L.append(f"\\multicolumn{{8}}{{@{{}}p{{0.97\\textwidth}}}}{{\\footnotesize\\quad Flagged columns (series): {cols}}} \\\\")
    L.append('\\addlinespace[2pt]')
write('tab_case.tex', '\\begin{tabular}{@{}l c c c c c c c@{}}\n\\toprule\n'
      '& & & \\multicolumn{5}{c}{LightGBM, mean test RMSE over 30 series} \\\\ \\cmidrule(l){4-8}\n'
      'Version & \\makecell{PAT\\\\flagged} & \\makecell{Non-\\\\deterministic} & no twin & + twin & '
      '\\makecell{Twin\\\\effect (\\%)} & \\makecell{Twin\\\\better} & $p$ \\\\\n\\midrule\n'
      + '\n'.join(L[:-1]) + '\n\\bottomrule\n\\end{tabular}\n')
print('tables written')
