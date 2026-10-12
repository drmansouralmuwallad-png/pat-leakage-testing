"""Print every quantity quoted in the results text, from results/summary.json and headline.json."""
import json, numpy as np
S = json.load(open('results/summary.json')); H = json.load(open('results/headline.json'))
P = json.load(open('results/pat.json')); R = json.load(open('results/recover.json'))
keys = [k for k in S]
print('combos', keys)
print('base vs snaive', {k: round(100*v['base_vs_snaive'], 1) for k, v in S.items()})
adm = [(k, c, v) for k, b in S.items() for c, v in b['cells'].items() if v['adm']]
print('admissible n', len(adm), 'range', round(100*min(v['eff'] for *_, v in adm), 1), round(100*max(v['eff'] for *_, v in adm), 1))
print('adm supports', [(k, c, round(100*v['eff'], 1)) for k, c, v in adm if v['eff'] <= -0.01 and v['p_holm'] < .05])
print('adm worse', [(k, c, round(100*v['eff'], 1)) for k, c, v in adm if v['eff'] >= 0.01 and v['p_holm'] < .05])
for cell in ['SES|T', 'ROLL|T', 'NV|T', 'INV|T', 'CMA|T', 'HP|T', 'CMA|O', 'HP|O', 'CMA|O*', 'HP|O*']:
    e = {k: round(100*b['cells'][cell]['eff'], 1) for k, b in S.items()}
    print(cell, 'range', min(e.values()), max(e.values()), e)
print('flips n', H['n_inadm'], 'support', H['n_inadm_support'], 'false', H['n_false_support'], 'inflated', H['n_inflated_true'])
print('inflated', [(f['ds'], f['learner'], f['gen'], f['proto'], round(100*f['inadm_eff'], 1), round(100*f['adm_eff'], 1)) for f in H['flips'] if f['inadm_supports'] and f['adm_supports']])
print('nonsupport', [(f['ds'], f['learner'], f['gen'], f['proto'], round(100*f['inadm_eff'], 1)) for f in H['flips'] if not f['inadm_supports']])
print('by learner', H['by_learner'])
print('conf', H['conf'])
print('shift FP', [(k, c) for k, b in S.items() for c, v in b['cells'].items() if v['adm'] and 'shift_med' in v and v['shift_med'] >= .01 and v['shift_p'] < .05])
miss = [k for k, v in P.items() if not v['admissible'] and not any(v['hits'])]; print('PAT misses', len(miss), miss[:10])
print('PAT audits per cell', sum(1 for k in P if k.endswith('|INV|T')))
for g in ['INV', 'SES', 'NV', 'ROLL', 'CMA', 'HP']:
    for ver in ['full', 'rt']:
        v = [x for k, x in R.items() if k.endswith(f'|{g}|{ver}')]
        if v: print('partialR2', g, ver, round(np.mean([x['P_same'] for x in v]), 2), round(np.mean([x['P_lag'] for x in v]), 2))
