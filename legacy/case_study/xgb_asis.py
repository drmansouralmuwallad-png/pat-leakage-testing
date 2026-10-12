"""Run the pipeline's own XGBoost / XGBoost+DT exactly as coded (h=7, all 30 series)."""
import sys, json, os, time
sys.path.insert(0, './pipeline')
import numpy as np, pandas as pd
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
H=int(__import__('os').environ.get('H','7'))
OUT=f'./asis_h{H}.json'; os.makedirs('/home/claude/verify', exist_ok=True)
done=json.load(open(OUT)) if os.path.exists(OUT) else {}
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
cfg=R.ExperimentConfig()
for sid in ids:
    if sid in done: continue
    t0=time.time()
    d=agg.loc[sid].values.astype(float)
    tw=DigitalTwinSimulator().simulate(d, cfg.train_end_day-1, avg_price=10.0, series_id=sid)
    conv=build_conventional_features(pd.Series(d))
    conv_arr=conv.fillna(0).values; comb=np.hstack([conv_arr, tw.fillna(0).values])
    res={'XGBoost':[], 'XGBoost+DT':[]}
    n_w=(cfg.test_end_day-cfg.val_end_day)//H
    for w in range(n_w):
        ws=cfg.val_end_day+w*H; we=min(ws+H, cfg.test_end_day); y=d[ws:we]
        for name,F in (('XGBoost',conv_arr),('XGBoost+DT',comb)):
            Xtr,ytr=F[:ws],d[:ws]; vs=int(len(Xtr)*0.9)
            m=R.XGBoostModel(cfg); m.fit(Xtr[:vs],ytr[:vs],Xtr[vs:],ytr[vs:])
            p=m.predict(F[ws:we])                      # features dated on the target days, as coded
            res[name].append(float(np.sqrt(np.mean((y-p)**2))))
    done[sid]={k:float(np.mean(v)) for k,v in res.items()}
    json.dump(done, open(OUT,'w'))
    print(f"{sid:18s} XGB {done[sid]['XGBoost']:8.2f}  XGB+DT {done[sid]['XGBoost+DT']:8.2f}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
