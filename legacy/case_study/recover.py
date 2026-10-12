"""How much of the target is recoverable from the features at the same index?
Fits an ordinary least-squares model of d_t on f_t over the training period and reports
in-sample R^2. High R^2 means the feature set encodes the value to be predicted."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'./pipeline')
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
TR=1577
def r2(X,y):
    X=np.column_stack([np.ones(len(X)),X])
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None); p=X@beta
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
res={}
for sid in ids:
    d=agg.loc[sid].values.astype(float)
    conv=build_conventional_features(pd.Series(d)).fillna(0)
    tw=DigitalTwinSimulator().simulate(d,TR-1,avg_price=10.0,series_id=sid).fillna(0)
    sl=slice(28,TR); y=d[sl]
    lag=conv[[c for c in conv.columns if c.startswith('sales_lag')]].values[sl]
    roll=conv[[c for c in conv.columns if c.startswith('rolling')]].values[sl]
    twv=tw.values[sl]
    res[sid]=dict(lags_only=r2(lag,y), rolling_same_day=r2(roll,y), twin_same_day=r2(twv,y),
                  all_same_day=r2(np.hstack([lag,roll,twv]),y),
                  twin_lagged_1=r2(tw.values[27:TR-1],y))
json.dump(res, open('recoverability.json','w'), indent=1)
m={k:float(np.mean([v[k] for v in res.values()])) for k in next(iter(res.values()))}
print("Mean in-sample R^2 for predicting d_t, 30 series, training period:")
for k,v in m.items(): print(f"  {k:20s} {v:6.3f}")
