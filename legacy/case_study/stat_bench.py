"""Leak-free statistical benchmarks: seasonal naive (m=7) and ETS.
Both are fitted only on demand up to the forecast origin, so they are leak-free by
construction. Same origins and windows as the other runs."""
import sys, os, json, time, warnings
sys.path.insert(0,'./pipeline')
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import run_full_simulation as R
from statsmodels.tsa.holtwinters import ExponentialSmoothing
H=int(os.environ.get('H','7')); OUT=f'./stat_h{H}.json'
done=json.load(open(OUT)) if os.path.exists(OUT) else {}
cfg=R.ExperimentConfig()
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
for sid in ids:
    if sid in done: continue
    t0=time.time(); d=agg.loc[sid].values.astype(float)
    res={'SeasonalNaive':[], 'ETS':[]}
    for w in range((cfg.test_end_day-cfg.val_end_day)//H):
        ws=cfg.val_end_day+w*H; o=ws-1; y=d[ws:ws+H]
        sn=np.resize(d[ws-7:ws], H)
        res['SeasonalNaive'].append(float(np.sqrt(np.mean((y-sn)**2))))
        hist=d[max(0,o-728):o+1]
        try:
            m=ExponentialSmoothing(hist, trend=None, seasonal='add', seasonal_periods=7,
                                   initialization_method='estimated').fit(optimized=True)
            p=np.asarray(m.forecast(H))
        except Exception:
            p=sn
        res['ETS'].append(float(np.sqrt(np.mean((y-p)**2))))
    done[sid]={k:float(np.mean(v)) for k,v in res.items()}
    json.dump(done, open(OUT,'w'))
    print(f"{sid:16s} sNaive {done[sid]['SeasonalNaive']:7.1f}  ETS {done[sid]['ETS']:7.1f} ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
