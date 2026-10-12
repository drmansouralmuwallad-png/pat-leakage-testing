"""Leak-free direct multi-step forecasting (h=7), matching the manuscript's stated protocol:
features are dated at the forecast origin o (last observed day, so d_o is known); the
target is d_{o+k}, k = 1..h. One pooled XGBoost per window with k as an input feature,
trained only on (feature at j, target at j+k) pairs whose targets fall before the origin."""
import sys, json, os, time
sys.path.insert(0, './pipeline')
import numpy as np, pandas as pd, xgboost as xgb
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
H=int(__import__('os').environ.get('H','7'))
OUT=f'./leakfree_h{H}.json'
done=json.load(open(OUT)) if os.path.exists(OUT) else {}
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
cfg=R.ExperimentConfig()
def pooled(F, d, o, H):
    """training pairs: X=[F[j], k], y=d[j+k], for all j with j+k <= o"""
    J=np.arange(28, o-H+1)                      # skip warm-up rows
    X=np.vstack([np.column_stack([F[J], np.full(len(J),k)]) for k in range(1,H+1)])
    y=np.concatenate([d[J+k] for k in range(1,H+1)])
    return X,y
for sid in ids:
    if sid in done: continue
    t0=time.time(); d=agg.loc[sid].values.astype(float)
    tw=DigitalTwinSimulator().simulate(d, cfg.train_end_day-1, avg_price=10.0, series_id=sid)
    conv=build_conventional_features(pd.Series(d)).fillna(0).values
    comb=np.hstack([conv, tw.fillna(0).values])
    res={'XGBoost':[], 'XGBoost+DT':[], 'naive_last_week':[]}
    for w in range((cfg.test_end_day-cfg.val_end_day)//H):
        ws=cfg.val_end_day+w*H; o=ws-1; y=d[ws:ws+H]
        res['naive_last_week'].append(float(np.sqrt(np.mean((y-np.resize(d[ws-7:ws],len(y)))**2))))
        for name,F in (('XGBoost',conv),('XGBoost+DT',comb)):
            X,t=pooled(F,d,o,H); vs=int(len(X)*0.9)
            m=xgb.XGBRegressor(n_estimators=cfg.xgb_n_rounds, max_depth=cfg.xgb_max_depth,
                  learning_rate=cfg.xgb_lr, subsample=cfg.xgb_subsample,
                  colsample_bytree=cfg.xgb_colsample, early_stopping_rounds=cfg.xgb_early_stop,
                  random_state=cfg.seed, n_jobs=1, verbosity=0)
            idx=np.random.default_rng(cfg.seed).permutation(len(X))   # val drawn across k
            tr,vl=idx[:vs],idx[vs:]
            m.fit(X[tr],t[tr],eval_set=[(X[vl],t[vl])],verbose=False)
            Xq=np.column_stack([np.repeat(F[o][None,:],H,0), np.arange(1,H+1)])  # origin-dated only
            p=m.predict(Xq); res[name].append(float(np.sqrt(np.mean((y-p)**2))))
    done[sid]={k:float(np.mean(v)) for k,v in res.items()}
    json.dump(done, open(OUT,'w'))
    print(f"{sid:18s} XGB {done[sid]['XGBoost']:8.2f}  XGB+DT {done[sid]['XGBoost+DT']:8.2f}  naive {done[sid]['naive_last_week']:8.2f} ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
