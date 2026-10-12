"""Re-run the corrected protocol at h=7 storing PER-WINDOW errors, so that MASE and
Diebold-Mariano tests can be computed. Covers seasonal naive, ETS, XGBoost and XGBoost+DT.
MASE scaling uses the in-sample seasonal naive (m=7) MAE on the training period."""
import sys, os, json, time, warnings
sys.path.insert(0,'./pipeline'); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
from statsmodels.tsa.holtwinters import ExponentialSmoothing
OUT='./metrics_h7.json'; done=json.load(open(OUT)) if os.path.exists(OUT) else {}
cfg=R.ExperimentConfig(); H=7
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
def pooled(F,d,o):
    J=np.arange(28,o-H+1)
    X=np.vstack([np.column_stack([F[J],np.full(len(J),k)]) for k in range(1,H+1)])
    return X, np.concatenate([d[J+k] for k in range(1,H+1)])
for sid in ids:
    if sid in done: continue
    t0=time.time(); d=agg.loc[sid].values.astype(float)
    scale=float(np.mean(np.abs(d[7:cfg.train_end_day]-d[:cfg.train_end_day-7])))  # in-sample sNaive MAE
    tw=DigitalTwinSimulator().simulate(d,cfg.train_end_day-1,avg_price=10.0,series_id=sid)
    conv=build_conventional_features(pd.Series(d)).fillna(0).values
    comb=np.hstack([conv,tw.fillna(0).values])
    err={k:[] for k in ('SeasonalNaive','ETS','XGBoost','XGBoost+DT')}
    for w in range((cfg.test_end_day-cfg.val_end_day)//H):
        ws=cfg.val_end_day+w*H; o=ws-1; y=d[ws:ws+H]
        sn=np.resize(d[ws-7:ws],H); err['SeasonalNaive'].append((y-sn).tolist())
        try:
            m=ExponentialSmoothing(d[max(0,o-728):o+1],trend=None,seasonal='add',
                 seasonal_periods=7,initialization_method='estimated').fit(optimized=True)
            p=np.asarray(m.forecast(H))
        except Exception: p=sn
        err['ETS'].append((y-p).tolist())
        for nm,F in (('XGBoost',conv),('XGBoost+DT',comb)):
            X,t=pooled(F,d,o); vs=int(len(X)*0.9)
            idx=np.random.default_rng(cfg.seed).permutation(len(X)); tr,vl=idx[:vs],idx[vs:]
            mm=xgb.XGBRegressor(n_estimators=cfg.xgb_n_rounds,max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_lr,subsample=cfg.xgb_subsample,colsample_bytree=cfg.xgb_colsample,
                early_stopping_rounds=cfg.xgb_early_stop,random_state=cfg.seed,n_jobs=1,verbosity=0)
            mm.fit(X[tr],t[tr],eval_set=[(X[vl],t[vl])],verbose=False)
            q=np.column_stack([np.repeat(F[o][None,:],H,0),np.arange(1,H+1)])
            err[nm].append((y-mm.predict(q)).tolist())
    done[sid]={'scale':scale,'errors':err}
    json.dump(done,open(OUT,'w'))
    print(f"{sid:16s} done ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
