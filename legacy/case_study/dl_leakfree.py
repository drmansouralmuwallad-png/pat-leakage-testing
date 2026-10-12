"""
dl_leakfree.py — leak-free LSTM / Transformer, 7-day horizon, all 30 series.

Same protocol as xgb_leakfree.py:
  - input  : the last 28 days of features up to and including the forecast origin o
             (demand channel + conventional features, + twin features for the DT variant)
  - output : the next 7 days of demand, predicted jointly (direct multi-step head)
  - train  : origins j with targets j+1..j+7 <= o only; last 10% (time-ordered) for early
             stopping
  - schedule: refit quarterly (windows 0 and 13 of 26); the model fitted at a refit window
             is reused, unchanged, for the windows after it. The manuscript refits every
             4th window; the quarterly schedule is needed for the run to finish on one CPU
             core, and it applies identically to the baseline and +DT models, so the paired
             comparison is unaffected
Architectures follow the pipeline config (LSTM 2x128; Transformer d=64, 4 heads, ff 256,
2 layers). Epoch cap reduced from 200 to 40 (patience 6), batch 128, so the run fits on one
CPU core; documented in the verification report.

Checkpoints per series to dl_leakfree_h7.json, so an interrupted run resumes.
"""
import sys, os, json, time
sys.path.insert(0, './pipeline')
import numpy as np, pandas as pd, torch, torch.nn as nn
import run_full_simulation as R
from digital_twin_simulation import DigitalTwinSimulator, build_conventional_features

torch.set_num_threads(1)
H_ENV = int(os.environ.get('H', '7'))
OUT = f'./dl_leakfree_h{H_ENV}.json'
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
H, L, EPOCHS, PATIENCE, RETRAIN = H_ENV, 28, 40, 6, 13
cfg = R.ExperimentConfig()


class LSTMNet(nn.Module):
    def __init__(s, f):
        super().__init__()
        s.rnn = nn.LSTM(f, 128, 2, batch_first=True, dropout=0.2)
        s.head = nn.Linear(128, H)
    def forward(s, x):
        o, _ = s.rnn(x); return s.head(o[:, -1])


class TransNet(nn.Module):
    def __init__(s, f):
        super().__init__()
        s.inp = nn.Linear(f, 64)
        s.pos = nn.Parameter(torch.zeros(1, L, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 256, 0.2, batch_first=True)
        s.enc = nn.TransformerEncoder(layer, 2)
        s.head = nn.Linear(64, H)
    def forward(s, x):
        z = s.enc(s.inp(x) + s.pos); return s.head(z[:, -1])


def windows(F, y, lo, hi):
    """origins j in [lo, hi]: X = F[j-L+1 .. j], Y = y[j+1 .. j+H]"""
    J = np.arange(lo, hi + 1)
    X = np.stack([F[j - L + 1:j + 1] for j in J]); Y = np.stack([y[j + 1:j + 1 + H] for j in J])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def train(Net, X, Y, seed):
    torch.manual_seed(seed)
    n = len(X); v = int(n * 0.9)
    net = Net(X.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, bad, state = 1e18, 0, None
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(v)
        for b in range(0, v, 128):
            i = perm[b:b + 128]; opt.zero_grad()
            loss = ((net(X[i]) - Y[i]) ** 2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(X[v:]) - Y[v:]) ** 2).mean().item()
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(state); net.eval(); return net


sales, _, _ = R.load_m5_data('./m5_data'); agg, ids = R.aggregate_to_store_category(sales)
for sid in ids:
    if sid in done: continue
    t0 = time.time()
    d = agg.loc[sid].values.astype(float)
    tw = DigitalTwinSimulator().simulate(d, cfg.train_end_day - 1, avg_price=10.0, series_id=sid)
    conv = build_conventional_features(pd.Series(d)).fillna(0).values
    twin = tw.fillna(0).values
    tr = slice(0, cfg.train_end_day)
    mu, sd = d[tr].mean(), d[tr].std() + 1e-9                     # training-period scaling only
    def z(A):
        m, s = A[tr].mean(0), A[tr].std(0); s[s == 0] = 1; return (A - m) / s
    base = np.column_stack([(d - mu) / sd, z(conv)])
    feats = {'LSTM': base, 'LSTM+DT': np.column_stack([base, z(twin)]),
             'Transformer': base, 'Transformer+DT': np.column_stack([base, z(twin)])}
    nets = {'LSTM': LSTMNet, 'LSTM+DT': LSTMNet, 'Transformer': TransNet, 'Transformer+DT': TransNet}
    ys = (d - mu) / sd
    res = {k: [] for k in feats}; model = {}
    for w in range((cfg.test_end_day - cfg.val_end_day) // H):
        ws = cfg.val_end_day + w * H; o = ws - 1; y_true = d[ws:ws + H]
        for k, F in feats.items():
            if w % RETRAIN == 0:
                X, Y = windows(F, ys, L - 1 + 28, o - H)
                model[k] = train(nets[k], X, Y, cfg.seed)
            with torch.no_grad():
                p = model[k](torch.tensor(F[o - L + 1:o + 1][None], dtype=torch.float32)).numpy()[0]
            res[k].append(float(np.sqrt(np.mean((y_true - (p * sd + mu)) ** 2))))
    done[sid] = {k: float(np.mean(v)) for k, v in res.items()}
    json.dump(done, open(OUT, 'w'))
    r = done[sid]
    print(f"{sid:16s} LSTM {r['LSTM']:7.1f}  +DT {r['LSTM+DT']:7.1f} | "
          f"Trans {r['Transformer']:7.1f}  +DT {r['Transformer+DT']:7.1f}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE")
