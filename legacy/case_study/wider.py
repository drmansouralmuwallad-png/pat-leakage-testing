"""Corrected protocol on a wider sample: store-DEPARTMENT series (10 stores x 7 depts = 70),
a finer aggregation than the 30 store-category series. XGBoost only, h=7."""
import sys, os, json, time
sys.path.insert(0,'./pipeline')
import numpy as np, pandas as pd, xgboost as xgb
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
OUT='./wider_h7.json'; done=json.load(open(OUT)) if os.path.exists(OUT) else {}
cfg=R.ExperimentConfig(); H=7
sales,_,_=R.load_m5_data('./m5_data')
dcols=[c for c in sales.columns if c.startswith('d_')]
sales['dept']=sales['item_id'].str.rsplit('_',n=1).str[0]
agg=sales.groupby(['store_id','dept'])[dcols].sum()
ids=[f"{a}|{b}" for a,b in agg.index]
agg.index=ids
print(f"{len(ids)} store-department series", flush=True)
def pooled(F,d,o):
    J=np.arange(28,o-H+1)
    X=np.vstack([np.column_stack([F[J],np.full(len(J),k)]) for k in range(1,H+1)])
    return X, np.concatenate([d[J+k] for k in range(1,H+1)])
for sid in ids:
    if sid in done: continue
    t0=time.time(); d=agg.loc[sid].values.astype(float)
    tw=DigitalTwinSimulator().simulate(d,cfg.train_end_day-1,avg_price=10.0,series_id=sid)
    conv=build_conventional_features(pd.Series(d)).fillna(0).values
    comb=np.hstack([conv,tw.fillna(0).values])
    res={'XGBoost':[], 'XGBoost+DT':[], 'naive':[]}
    for w in range((cfg.test_end_day-cfg.val_end_day)//H):
        ws=cfg.val_end_day+w*H; o=ws-1; y=d[ws:ws+H]
        res['naive'].append(float(np.sqrt(np.mean((y-d[ws-7:ws])**2))))
        for nm,F in (('XGBoost',conv),('XGBoost+DT',comb)):
            X,t=pooled(F,d,o); vs=int(len(X)*0.9)
            idx=np.random.default_rng(cfg.seed).permutation(len(X)); tr,vl=idx[:vs],idx[vs:]
            m=xgb.XGBRegressor(n_estimators=cfg.xgb_n_rounds,max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_lr,subsample=cfg.xgb_subsample,colsample_bytree=cfg.xgb_colsample,
                early_stopping_rounds=cfg.xgb_early_stop,random_state=cfg.seed,n_jobs=1,verbosity=0)
            m.fit(X[tr],t[tr],eval_set=[(X[vl],t[vl])],verbose=False)
            q=np.column_stack([np.repeat(F[o][None,:],H,0),np.arange(1,H+1)])
            res[nm].append(float(np.sqrt(np.mean((y-m.predict(q))**2))))
    done[sid]={k:float(np.mean(v)) for k,v in res.items()}
    json.dump(done,open(OUT,'w'))
    print(f"{sid:22s} XGB {done[sid]['XGBoost']:7.1f}  +DT {done[sid]['XGBoost+DT']:7.1f} ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
