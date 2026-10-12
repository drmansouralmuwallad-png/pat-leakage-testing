import numpy as np; from gens import _hp_trend, g_hp_rt
rng=np.random.default_rng(1); y=np.cumsum(rng.normal(size=400))+rng.normal(size=400)*3
rt=g_hp_rt(y,100,7)[:,0]
err=[abs(_hp_trend(y[:t+1])[-1]-rt[t]) for t in (50,120,250,399)]
print('max |KF end-point - HP end-point| =',max(err), ' scale', y.std())
