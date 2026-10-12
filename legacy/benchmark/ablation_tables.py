"""PAT sensitivity (draws B, perturbation scale, tolerance eps) and cost, from
results/pat_ablation.json and results/timing_fit.json. Writes ../paper/tab_ablation.tex,
../paper/tab_cost.tex, ../paper/figs/fig4_pat_sensitivity.pdf and results/ablation_summary.json"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = json.load(open('results/pat_ablation.json')); TF = json.load(open('results/timing_fit.json'))
from run_bench import CFG
SC = ['0.25', '0.5', '1.0', '2.0', '3.0']; BS = [1, 2, 5, 10, 20]; EPS = [1e-12, 1e-9, 1e-6, 1e-3]
LIN = ['SES|T', 'ROLL|T', 'CMA|T', 'CMA|O', 'HP|T', 'HP|O']; PW = ['INV|T', 'NV|T']
def det(keys, s, B, eps=1e-9):
    return float(np.mean([any(x > eps for x in A[k]['maxrel'][s][:B]) for k in keys]))
def keys_of(cell): return [k for k in A if k.endswith('|' + cell)]
adm_keys = [k for k, v in A.items() if v['admissible']]
out = dict(units_per_cell=len(keys_of('INV|T')))
rows = []
for name, cells in [('Linear generators (6 cells)', LIN), ('INV, T', ['INV|T']), ('NV, T', ['NV|T'])]:
    ks = sum((keys_of(c) for c in cells), [])
    out[name] = {s: {B: det(ks, s, B) for B in BS} for s in SC}
    rows.append((name, ks))
fp = {s: float(np.mean([any(x > 1e-9 for x in A[k]['maxrel'][s]) for k in adm_keys])) for s in SC}
out['false_alarm_rate'] = fp; out['admissible_audits_per_scale'] = len(adm_keys) * 3
inad = [k for k, v in A.items() if not v['admissible']]
out['eps'] = {str(e): dict(det=det(inad, '1.0', 5, e), fa=float(np.mean([any(x > e for x in A[k]['maxrel']['1.0']) for k in adm_keys]))) for e in EPS}
out['max_admissible_change'] = float(max(max(max(v) for v in A[k]['maxrel'].values()) for k in adm_keys))
# minimal relative change seen on detected linear cells at scale 0.25
out['min_detected_change_linear_0.25'] = float(min(x for c in LIN for k in keys_of(c) for x in A[k]['maxrel']['0.25'] if x > 0))
# --- table: B x scale for the three groups
L = [r'\begin{tabular}{@{}l l ccccc@{}}', r'\toprule', r'& & \multicolumn{5}{c}{Draws $B$} \\ \cmidrule(l){3-7}',
     r'Cells & Scale & 1 & 2 & 5 & 10 & 20 \\', r'\midrule']
for name, _ in rows:
    for j, s in enumerate(SC):
        L.append((name if j == 0 else '') + f' & {float(s):g} & ' + ' & '.join(f'{100*out[name][s][B]:.1f}' for B in BS) + r' \\')
    L.append(r'\addlinespace[2pt]')
L[-1] = r'\midrule'
L.append(r'\multicolumn{2}{@{}l}{False alarms, admissible cells} & \multicolumn{5}{c}{' +
         ', '.join(f'{100*fp[s]:.1f}' for s in SC) + r' (scales 0.25--3)} \\')
L += [r'\bottomrule', r'\end{tabular}']
open('../paper/tab_ablation.tex', 'w').write('\n'.join(L))
# --- cost table
NM = {'M5': 'M5', 'M4D': 'M4-Daily', 'M4H': 'M4-Hourly', 'ELEC': 'Electricity'}
C = [r'\begin{tabular}{@{}l c c c c c@{}}', r'\toprule',
     r'Panel & \makecell{Pipeline run\\$C_\phi$ (ms)\\mean [max cell]} & \makecell{PAT per series\\$2(B{+}1)C_\phi$, $B=5$\\(s)} & \makecell{LightGBM fit\\$C_{\mathrm{fit}}$ (s)} & \makecell{Shift check per\\series and cell\\$R\,C_{\mathrm{fit}}$ (s)} & Ratio \\', r'\midrule']
cost = {}
for ds in ['M5', 'M4D', 'M4H', 'ELEC']:
    per_cell = {}
    for k, v in A.items():
        if k.startswith(ds + '|'): per_cell.setdefault(k.split('|')[-2] + '|' + k.split('|')[-1], []).append(v['sec_per_run'])
    means = {c: np.mean(x) for c, x in per_cell.items()}
    cphi = float(np.mean(list(means.values()))); cmax = float(max(means.values()))
    pat = 2 * 6 * cphi; shift = CFG[ds]['R'] * TF[ds]['lgbm_fit_sec']
    cost[ds] = dict(cphi_ms=1000*cphi, cphi_max_ms=1000*cmax, pat_s=pat, fit_s=TF[ds]['lgbm_fit_sec'], shift_s=shift, ratio=shift/pat)
    C.append(f"{NM[ds]} & {1000*cphi:.1f} [{1000*cmax:.1f}] & {pat:.2f} & {TF[ds]['lgbm_fit_sec']:.2f} & {shift:.2f} & {shift/pat:.1f}" + r' \\')
C += [r'\bottomrule', r'\end{tabular}']
open('../paper/tab_cost.tex', 'w').write('\n'.join(C)); out['cost'] = cost
json.dump(out, open('results/ablation_summary.json', 'w'), indent=1)
# --- figure
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4))
col = {'Linear generators (6 cells)': '#2e6fb0', 'INV, T': '#b03a2e', 'NV, T': '#d18b2c'}
for name, _ in rows:
    axs[0].plot([float(s) for s in SC], [100*out[name][s][5] for s in SC], 'o-', color=col[name], label=name)
    axs[1].plot(BS, [100*out[name]['1.0'][B] for B in BS], 'o-', color=col[name], label=name)
axs[0].plot([float(s) for s in SC], [100*fp[s] for s in SC], 's--', color='#4a5a6a', label='admissible cells (false alarms)')
axs[0].set_xlabel('perturbation scale (x s.d. of first differences)'); axs[0].set_ylabel('flagged audits (%)'); axs[0].set_title('$B=5$', fontsize=9.5)
axs[1].set_xscale('log'); axs[1].set_xticks(BS); axs[1].set_xticklabels(BS); axs[1].set_xlabel('number of draws $B$'); axs[1].set_title('scale 1', fontsize=9.5)
for a in axs: a.set_ylim(-3, 103)
axs[0].legend(fontsize=7.5, frameon=False, loc='center right')
fig.savefig('../paper/figs/fig4_pat_sensitivity.pdf'); print(json.dumps({k: v for k, v in out.items() if k != 'cost'}, indent=0)[:2500]); print(cost)
