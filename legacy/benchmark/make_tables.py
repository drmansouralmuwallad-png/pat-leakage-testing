"""LaTeX tables and headline counts for the manuscript, read from results/*.json.
Writes ../paper/tab_bench.tex, ../paper/tab_diag.tex and results/headline.json."""
import json, numpy as np, collections
from gens import CELLS
S = json.load(open('results/summary.json')); P = json.load(open('results/pat.json'))
rec = json.load(open('results/recover.json'))
PANELS = ['M5','M4D','M4H','ELEC']
def mkcols(ls): return [(d, l) for d in PANELS for l in ls if f'{d}|{l}' in S]
cols = mkcols(['ridge','lgbm']); cols2 = mkcols(['xgb','mlp']); allcols = mkcols(['ridge','lgbm','xgb','mlp'])
NAME = {'M5':'M5','M4D':'M4-D','M4H':'M4-H','ELEC':'Elec.'}; LN = {'ridge':'Ridge','lgbm':'LGBM','xgb':'XGB','mlp':'MLP'}
def fmt(c):
    e = 100*c['eff']; s = f'{e:+.1f}' if abs(e) < 99.95 else f'{e:+.0f}'
    return (r'\textbf{' + s + '}') if (c['p_holm'] < 0.05 and c['eff'] <= -0.01) else (s + (r'$^{\dagger}$' if (c['p_holm'] < 0.05 and c['eff'] >= 0.01) else ''))
def bench_table(cols):
    L = [r'\begin{tabular}{@{}ll' + 'r'*len(cols) + '@{}}', r'\toprule',
         ' & & ' + ' & '.join(r'\multicolumn{1}{c}{' + NAME[d] + '}' for d, l in cols) + r' \\',
         'Cell & Adm. & ' + ' & '.join(r'\multicolumn{1}{c}{' + LN[l] + '}' for d, l in cols) + r' \\', r'\midrule']
    prev = None
    for g, p, a in CELLS:
        if prev and g != prev: L.append(r'\addlinespace[2pt]')
        prev = g
        L.append(f'{g}, {p} & {"yes" if a else "no"} & ' + ' & '.join(fmt(S[f"{d}|{l}"]["cells"][f"{g}|{p}"]) for d, l in cols) + r' \\')
    L += [r'\midrule', r'\multicolumn{2}{@{}l}{Base vs.\ seasonal naive} & ' +
          ' & '.join(f'{100*S[f"{d}|{l}"]["base_vs_snaive"]:+.1f}' for d, l in cols) + r' \\', r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)
open('../paper/tab_bench.tex', 'w').write(bench_table(cols))
open('../paper/tab_bench2.tex', 'w').write(bench_table(cols2))

# ---------- diagnostics
pat = collections.defaultdict(lambda: [0, 0, 0, 0])
for k, v in P.items():
    ds, *_, g, p = k.split('|'); a = pat[(g, p)]; a[0] += len(v['hits']); a[1] += sum(v['hits']); a[2] += 1; a[3] += any(v['hits'])
def shift_flag(c): return c['shift_med'] >= 0.01 and c['shift_p'] < 0.05
H_ALL = None
def psame(g, ver):
    return np.mean([v['P_same'] for k, v in rec.items() if k.endswith(f'|{g}|{ver}')]), np.mean([v['P_lag'] for k, v in rec.items() if k.endswith(f'|{g}|{ver}')])
D = [r'\begin{tabular}{@{}ll ccc cc@{}}', r'\toprule',
     r'& & \multicolumn{3}{c}{Cells flagged / audited} & \multicolumn{2}{c}{Partial $R^2$ of $y_t$} \\ \cmidrule(lr){3-5}\cmidrule(l){6-7}',
     r'Cell & Adm. & Index & Shift & PAT & row $t$ & row $t-1$ \\', r'\midrule']
conf = {k: dict(tp=0, fn=0, fp=0, tn=0) for k in ('index', 'shift', 'pat1', 'pat5')}
def upd(k, flag, adm):
    conf[k]['fp' if (flag and adm) else 'tp' if flag else 'tn' if adm else 'fn'] += 1
prev = None
for g, p, a in CELLS:
    if prev and g != prev: D.append(r'\addlinespace[2pt]')
    prev = g
    idx = (p == 'T'); sh = [shift_flag(S[f'{d}|{l}']['cells'][f'{g}|{p}']) for d, l in cols]
    for d, l in cols:
        upd('index', idx, a); upd('shift', shift_flag(S[f'{d}|{l}']['cells'][f'{g}|{p}']), a)
    n, h, u, an = pat[(g, p)]
    for k, v in P.items():
        if k.endswith(f'|{g}|{p}'):
            upd('pat5', any(v['hits']), a)
            for x in v['hits']: upd('pat1', x, a)
    ver = 'rt' if p == 'O*' else 'full'; ps, pl = psame(g, ver)
    D.append(f'{g}, {p} & {"yes" if a else "no"} & {len(cols) if idx else 0}/{len(cols)} & {sum(sh)}/{len(cols)} & {an}/{u} & {ps:.2f} & {pl:.2f}' + r' \\')
D += [r'\bottomrule', r'\end{tabular}']
open('../paper/tab_diag.tex', 'w').write('\n'.join(D))

# ---------- headline: conclusions supported / reversed
flips = []
for d, l in allcols:
    C = S[f'{d}|{l}']['cells']
    for g in ['INV', 'SES', 'NV', 'ROLL', 'CMA', 'HP']:
        adm = C[f'{g}|O*'] if f'{g}|O*' in C else C[f'{g}|O']
        sup = lambda c: c['eff'] <= -0.01 and c['p_holm'] < 0.05
        for p in (['T', 'O'] if f'{g}|O*' in C else ['T']):
            c = C[f'{g}|{p}']
            flips.append(dict(ds=d, learner=l, gen=g, proto=p, inadm_eff=c['eff'], adm_eff=adm['eff'],
                              inadm_supports=sup(c), adm_supports=sup(adm)))
H = dict(conf=conf, flips=flips,
         n_inadm=len(flips), n_inadm_support=sum(f['inadm_supports'] for f in flips),
         n_false_support=sum(f['inadm_supports'] and not f['adm_supports'] for f in flips),
         n_inflated_true=sum(f['inadm_supports'] and f['adm_supports'] for f in flips),
         pat_rates={f'{g}|{p}': dict(single=pat[(g, p)][1]/pat[(g, p)][0], any5=pat[(g, p)][3]/pat[(g, p)][2]) for g, p, _ in CELLS})
json.dump(H, open('results/headline.json', 'w'), indent=1)
for k, v in conf.items(): print(k, v, 'TPR %.3f FPR %.3f' % (v['tp']/(v['tp']+v['fn']), v['fp']/max(1, v['fp']+v['tn'])))
print('inadmissible evaluations', H['n_inadm'], 'supporting', H['n_inadm_support'], 'false support', H['n_false_support'], 'inflated true', H['n_inflated_true'])
for f in flips:
    if f['inadm_supports']: print(f"{f['ds']:4s} {f['learner']:5s} {f['gen']:4s} {f['proto']:2s} inadm {100*f['inadm_eff']:+7.1f} adm {100*f['adm_eff']:+6.1f} adm_sup {f['adm_supports']}")

# ---------- per-learner breakdown (exploitability varies, admissibility does not)
byL = {}
for f in flips:
    b = byL.setdefault(f['learner'], dict(n=0, support=0, false_support=0))
    b['n'] += 1; b['support'] += f['inadm_supports']; b['false_support'] += f['inadm_supports'] and not f['adm_supports']
H['by_learner'] = byL
json.dump(H, open('results/headline.json', 'w'), indent=1)
print('by learner', byL)
