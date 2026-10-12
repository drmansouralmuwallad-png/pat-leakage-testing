"""Recompute Figure 15's correlations from the real M5 data and the pipeline's features,
under several plausible methods, and test each against the 138 values recovered from the
original figure. Training period only (days 1-1,577)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'./pipeline')
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features
sales,_,_=R.load_m5_data('./m5_data'); agg,ids=R.aggregate_to_store_category(sales)
cal=pd.read_csv('./m5_data/calendar.csv')
TR=1577
# price per store-category-week: mean item price, mapped to days
pr=pd.read_csv('./m5_data/sell_prices.csv', usecols=['store_id','item_id','wm_yr_wk','sell_price'])
pr['cat']=pr.item_id.str.split('_').str[0]
wk=pr.groupby(['store_id','cat','wm_yr_wk']).sell_price.mean()
d2w=cal.wm_yr_wk.values[:1941]
names=["sales_lag_7","rolling_mean_7d","rolling_std_7d","sell_price","day_of_week","snap_flag",
       "inv_position","days_of_supply","stockout_flag","sc_stress","fill_rate_7d","pipeline_inv"]
rows=[]
for sid in ids:
    st,cat=sid.rsplit('_',1)[0], sid.rsplit('_',1)[1]
    d=agg.loc[sid].values.astype(float)
    conv=build_conventional_features(pd.Series(d))
    tw=DigitalTwinSimulator().simulate(d, TR-1, avg_price=10.0, series_id=sid)
    price=np.array([wk.get((st,cat,w),np.nan) for w in d2w])
    df=pd.DataFrame({
      "sales_lag_7":conv["sales_lag_7"].values, "rolling_mean_7d":conv["rolling_mean_7"].values,
      "rolling_std_7d":conv["rolling_std_7"].values, "sell_price":price,
      "day_of_week":cal.wday.values[:1941], "snap_flag":cal[f"snap_{st[:2]}"].values[:1941]})
    for n in names[6:]: df[n]=tw[n].values
    df["sid"]=sid; rows.append(df.iloc[28:TR])
D=pd.concat(rows)
recovered=json.load(open('/home/claude/dt/correlation_values.json'))
K=np.array([[np.nan if v is None else v for v in r] for r in recovered["M"]],float)
def score(M):
    m=~np.isnan(K); iu=np.triu_indices(12,1); mm=m[iu]
    e=np.abs(M[iu][mm]-K[iu][mm]); return e.mean(), e.max()
methods={}
methods["pooled raw"]=D[names].corr().values
methods["mean of per-series"]=np.nanmean(np.stack([g[names].corr().values for _,g in D.groupby("sid")]),0)
Z=D.groupby("sid")[names].transform(lambda s:(s-s.mean())/(s.std() if s.std()>0 else 1))
methods["pooled, standardised per series"]=Z.corr().values
for sid in ["CA_1_FOODS"]:
    methods[f"single series {sid}"]=D[D.sid==sid][names].corr().values
print("method                               mean|err|  max|err|   vs 138 recovered values")
for k,M in methods.items():
    a,b=score(M); print(f"  {k:36s} {a:8.3f}  {b:8.3f}")
best=min(methods,key=lambda k:score(methods[k])[0])
np.save('./corr_methods.npy', {k:v for k,v in methods.items()}, allow_pickle=True)
M=methods["mean of per-series"]
print("\nthe three unlabelled pairs under each method:")
for (i,j) in [(0,10),(1,8),(3,10)]:
    print(f"  {names[i]:16s} x {names[j]:14s} " + "  ".join(f"{k.split()[0]}:{methods[k][i,j]:+.2f}" for k in methods))
print("\nsign agreement with recovered values (pooled raw):",
      f"{np.mean(np.sign(methods['pooled raw'][np.triu_indices(12,1)])==np.sign(np.nan_to_num(K[np.triu_indices(12,1)]))):.2f}")
