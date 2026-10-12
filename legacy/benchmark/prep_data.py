"""Prepare the three benchmark panels as .npy arrays (one list of 1-d series per dataset).
M5: 30 store x category daily unit-sales series (sum over items), 1,941 days.
M4 Daily: 40 series with length >= 1,500, drawn with a fixed seed; last 1,500 obs kept.
M4 Hourly: 40 series of length 960 (the longest), drawn with a fixed seed."""
import numpy as np, pandas as pd, json
rng=np.random.default_rng(2026)
out={}
# M5
cols=pd.read_csv('../data/sales_train_evaluation.csv',nrows=0).columns
dcols=[c for c in cols if c.startswith('d_')]
agg=None
for ch in pd.read_csv('../data/sales_train_evaluation.csv',usecols=['store_id','cat_id']+dcols,chunksize=5000):
    g=ch.groupby(['store_id','cat_id'])[dcols].sum()
    agg=g if agg is None else agg.add(g,fill_value=0)
agg=agg.sort_index()
out['M5']={f'{s}|{c}':agg.loc[(s,c)].values.astype(float) for s,c in agg.index}
# M4 daily
d=pd.read_csv('../data/m4_Daily.csv',index_col=0)
lens=d.notna().sum(1); ok=lens[lens>=1500].index.tolist()
pick=sorted(rng.choice(ok,40,replace=False).tolist(),key=lambda s:int(s[1:]))
out['M4D']={s:d.loc[s].dropna().values.astype(float)[-1500:] for s in pick}
# M4 hourly
h=pd.read_csv('../data/m4_Hourly.csv',index_col=0)
lens=h.notna().sum(1); ok=lens[lens>=960].index.tolist()
pick=sorted(rng.choice(ok,40,replace=False).tolist(),key=lambda s:int(s[1:]))
out['M4H']={s:h.loc[s].dropna().values.astype(float)[-960:] for s in pick}
np.save('panels.npy',out,allow_pickle=True)
for k,v in out.items():
    L=[len(x) for x in v.values()]; print(k,len(v),'series, length',min(L),max(L),
          'min value',min(x.min() for x in v.values()))
