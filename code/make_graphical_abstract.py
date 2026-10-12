"""Graphical abstract at its print size, 13 x 5 cm (Elsevier: at least 1328 x 531 px, readable
at 5 x 13 cm). Numbers are read from summary.json.
usage: python make_graphical_abstract.py <results_dir> <out_stem>   -> .pdf, .tif, .png"""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
S = json.load(open(os.path.join(sys.argv[1], 'summary.json')))
INK, INK2, BLUE = '#0b0b0b', '#52514e', '#1d5aa6'
t = S['rq1_temporal_full']['tools']; k = S['rq1_tabular']['tools']
n_clean = S['rq1_temporal_full']['n_clean'] + S['rq1_tabular']['n_clean']
plt.rcParams['font.family'] = 'DejaVu Sans'
fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 50); ax.axis('off')
def box(x, y, w, h, txt, fc='#f4f3f0', fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6', fc=fc, ec=INK2, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.15)
def arr(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, mutation_scale=7))
ax.text(3, 44.5, 'PAT: black-box test for leakage in ML feature pipelines', fontsize=9, color=INK, weight='bold')
box(2, 23, 23, 11, 'Run pipeline\nscript as is')
box(2, 5, 23, 11, 'Rerun with only\nunavailable data\nperturbed', fc='#fdf0e9')
arr(13.5, 22.4, 13.5, 16.6)
box(31, 14, 20, 11, 'Learner inputs\nchanged?', fc='#eaf2fb')
arr(26, 28, 30.4, 22); arr(26, 11, 30.4, 17)
box(57, 26, 16, 8, 'No: pass')
box(57, 4, 16, 9, 'Yes: leak +\ncolumns', fc='#fbe9e7')
arr(51.6, 21.5, 56.4, 29); arr(51.6, 17.5, 56.4, 9)
ax.plot([78, 78], [4, 38], color='#d9d8d4', lw=0.7)
ax.text(82, 33, f"{100*t['PAT']['tpr']['rate']:.0f}% / {100*k['PAT']['tpr']['rate']:.0f}%", fontsize=14, color=BLUE, weight='bold')
ax.text(82, 27.5, 'temporal / tabular leaks found', fontsize=7.5, color=INK)
ax.text(82, 16, f"0 of {n_clean:,}", fontsize=14, color=BLUE, weight='bold')
ax.text(82, 10.5, 'clean pipelines flagged', fontsize=7.5, color=INK)
ax.text(82, 3.5, f"4,380 pipelines, 26 leak types", fontsize=6.8, color=INK2)
stem = sys.argv[2]
fig.savefig(stem + '.pdf'); fig.savefig(stem + '.png', dpi=600)
from PIL import Image
Image.open(stem + '.png').convert('RGB').save(stem + '.tif', compression='tiff_lzw', dpi=(600, 600))
print('saved', Image.open(stem + '.png').size)
