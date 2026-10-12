"""Figures for the IST manuscript, drawn from results/summary.json.

usage: python make_figs.py <results_dir> <fig_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch         # noqa: E402

R, F = sys.argv[1:3]
S = json.load(open(os.path.join(R, 'summary.json')))
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ---------------------------------------------------------------- Fig. 1: PAT overview
fig, ax = plt.subplots(figsize=(8.4, 2.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis('off')


def box(x, y, w, h, text, fill='#f4f3f0', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc=fill, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6, color=INK,
            weight='bold' if bold else 'normal', linespacing=1.3)


def arrow(x0, y0, x1, y1, text=None):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.3, text, ha='center', fontsize=6.8, color=INK2)


box(0.5, 13, 13, 10, 'Data $D$\n(available $A$ +\nunavailable $U$)')
box(20, 25, 18, 8.5, 'Pipeline script $\\phi$\n(unmodified)')
box(20, 2, 18, 8.5, 'Pipeline script $\\phi$\n(same code)')
box(0.5, 0.5, 13, 9.5, 'Perturbation\nlaw $Q$ on $U$\n($B$ draws)', fill='#fdf0e9')
arrow(13.8, 21, 19.5, 28.5)
ax.text(14.5, 27.5, 'run', fontsize=6.8, color=INK2)
arrow(13.8, 15, 19.5, 7.5)
ax.text(17.6, 13.2, 'rerun with $U+\\delta^{(b)}$', fontsize=6.8, color=INK2)
arrow(7, 10.3, 7, 12.6)
box(44, 25, 21, 8.5, r'Learner-facing outputs' + '\n' + r'$X_{tr},\ y_{tr},\ X_{te}[A]$')
box(44, 2, 21, 8.5, r'Learner-facing outputs' + '\n' + r'$X^{(b)}_{tr},\ y^{(b)}_{tr},\ X^{(b)}_{te}[A]$')
arrow(38.4, 29, 43.6, 29)
arrow(38.4, 6, 43.6, 6)
box(70, 12.5, 12, 11, 'Compare\n(relative\ntolerance $\\varepsilon$)', fill='#eaf2fb')
arrow(65.4, 28, 69.6, 21)
arrow(65.4, 7, 69.6, 15)
box(86.5, 22, 13, 11, 'No change:\nno violation\ndetected')
box(86.5, 3, 13, 11, 'Change:\ninadmissible +\nflagged columns', fill='#fbe9e7')
arrow(82.4, 21, 86, 26)
arrow(82.4, 15, 86, 10)
fig.savefig(os.path.join(F, 'fig_pat_overview.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: TPR vs FPR
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
order = ['PAT', 'Time-offset check', 'Shift check (1%)', 'Shift check (calibrated)',
         'Static analyser (Yang et al.)', 'Deepchecks']
short = {'PAT': 'PAT', 'Time-offset check': 'Time-offset check', 'Shift check (1%)': 'Shift check, 1%',
         'Shift check (calibrated)': 'Shift check, calibrated',
         'Static analyser (Yang et al.)': 'Static analyser', 'Deepchecks': 'Deepchecks'}
offsets = {'PAT': (7, -10), 'Time-offset check': (8, 7), 'Shift check (1%)': (8, -11),
           'Shift check (calibrated)': (8, -11), 'Static analyser (Yang et al.)': (-8, 7),
           'Deepchecks': (-8, -12)}
for ax, sec, title in [(axes[0], 'rq1_temporal_subset', 'Temporal pipelines (common subset)'),
                       (axes[1], 'rq1_tabular', 'Tabular pipelines')]:
    tools = S[sec]['tools']
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--', zorder=0)
    for i, name in enumerate(order):
        if name not in tools:
            continue
        t = tools[name]
        x, y = t['fpr']['rate'], t['tpr']['rate']
        ax.errorbar(x, y, xerr=[[x - t['fpr']['lo']], [t['fpr']['hi'] - x]],
                    yerr=[[y - t['tpr']['lo']], [t['tpr']['hi'] - y]], fmt='o', ms=6,
                    color=CAT[i], mec='white', mew=1.2, elinewidth=1, capsize=0, zorder=3)
        dx, dy = offsets[name]
        if sec == 'rq1_tabular' and name.startswith('Static'):
            dx, dy = (8, -3)
        ax.annotate(short[name], (x, y), xytext=(dx, dy), textcoords='offset points',
                    ha='left' if dx > 0 else 'right', fontsize=7.2, color=INK)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel('False-alarm rate on clean pipelines')
    ax.set_title(title, fontsize=8.5, color=INK, loc='left')
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Detection rate on leaky pipelines')
fig.tight_layout()
fig.savefig(os.path.join(F, 'fig_tpr_fpr.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: sensitivity grid
seq = ['#b9d4f2', '#7fb0e8', '#3f86d9', '#1d5aa6']   # one hue, light to dark (scale 0.5 to 3)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
names = {'FLAG_ALL': 'Threshold flag (FLAG_ALL)', 'WINS_ALL': 'Winsorising (WINS_ALL)',
         'SEL_ALL': 'Feature selection (SEL_ALL)'}
for ax, op in zip(axes, ['FLAG_ALL', 'WINS_ALL', 'SEL_ALL']):
    for j, s in enumerate(['0.5', '1.0', '2.0', '3.0']):
        d = S['rq3_grid'][f'{op}|{s}']
        bs = [1, 2, 5, 10]
        ax.plot(bs, [d[str(b)]['rate'] for b in bs], marker='o', ms=4.5, lw=2, color=seq[j],
                mec='white', mew=0.8, label=f'scale {float(s):g}')
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1', '2', '5', '10'])
    ax.minorticks_off()
    ax.set_title(names[op], fontsize=8.2, color=INK, loc='left')
    ax.set_xlabel('Draws $B$'); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel('Detection rate')
axes[0].legend(frameon=False, fontsize=7, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(F, 'fig_sensitivity.pdf'), bbox_inches='tight')
plt.close(fig)
print('figures written')
