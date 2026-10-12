"""Target-driven feature generators, the forecast-origin protocols, and the perturbation
admissibility test (PAT). Every generator maps a (normalised) series y[0..T-1] to a T x p
array G whose row t is what the generator reports at time t. Parameters are calibrated on
y[:t0] only, t0 being one step after the first test origin, so calibration itself never
reads test-period data."""
import numpy as np, pandas as pd

# ---------------------------------------------------------------- causal generators
INV_MODE = 'seeded'   # 'seeded' (default), 'unseeded', 'crn'  -- see pat_crn.py
def g_inv(y, t0, m):
    """(s,S) inventory simulation, lost sales, integer lead time U{2..5}; state recorded
    after the period's demand is served (the ordinary simulator update order)."""
    mu, sd = y[:t0].mean(), y[:t0].std()
    Lbar = 3.5
    s = mu*Lbar + 1.65*sd*np.sqrt(Lbar); S = s + mu*m
    rng = np.random.default_rng(None if INV_MODE == 'unseeded' else 0)
    T = len(y); oh = S; pipe = {}                        # arrival time -> quantity
    Lcrn = rng.integers(2, 6, size=T) if INV_MODE == 'crn' else None  # one draw per period
    out = np.zeros((T, 4)); served = np.zeros(T)
    for t in range(T):
        oh += pipe.pop(t, 0.0)
        before = oh; sv = min(y[t], oh); oh -= sv; served[t] = sv
        ip = oh + sum(pipe.values())
        if ip <= s:
            L = int(Lcrn[t]) if Lcrn is not None else int(rng.integers(2, 6)); pipe[t+L] = pipe.get(t+L, 0.0) + (S-ip); ip = S
        out[t, 0] = ip; out[t, 1] = oh; out[t, 2] = float(y[t] > before + 1e-12)
    dem = pd.Series(y).rolling(7, min_periods=1).sum().values
    srv = pd.Series(served).rolling(7, min_periods=1).sum().values
    out[:, 3] = np.where(dem > 0, srv/np.maximum(dem, 1e-12), 1.0)
    return out

def g_ses(y, t0, m):
    """State-space filter: simple exponential smoothing (local-level Kalman filter in
    steady state). Reports the filtered level and the one-step innovation."""
    def run(a, x):
        l = np.empty(len(x)); e = np.empty(len(x)); prev = x[0]
        for t in range(len(x)):
            e[t] = x[t]-prev; prev = prev + a*e[t]; l[t] = prev
        return l, e
    grid = np.linspace(0.05, 0.95, 19)
    a = grid[np.argmin([np.sum(run(a, y[:t0])[1][1:]**2) for a in grid])]
    l, e = run(a, y)
    return np.column_stack([l, e])

def g_nv(y, t0, m, w=28, q=0.8):
    """Re-solved newsvendor: the critical-fractile order quantity re-optimised each period
    on the last w observations, and the realised shortage against the previous decision."""
    s = pd.Series(y)
    qt = s.rolling(w, min_periods=1).quantile(q).values
    short = np.maximum(y - np.r_[qt[0], qt[:-1]], 0.0)
    return np.column_stack([qt, short])

def g_roll(y, t0, m):
    """Rolling mean and standard deviation over one seasonal cycle, window ending at t."""
    s = pd.Series(y)
    return np.column_stack([s.rolling(m, min_periods=1).mean().values,
                            s.rolling(m, min_periods=2).std().fillna(0).values])

# ---------------------------------------------------------------- non-causal generators
def _cma_w(m):  # centred window of odd length covering one cycle
    return m+1 if m % 2 == 0 else m
def g_cma(y, t0, m):
    """Centred moving average over one seasonal cycle, computed on the full series."""
    return pd.Series(y).rolling(_cma_w(m), center=True, min_periods=1).mean().values[:, None]
def g_cma_rt(y, t0, m):
    """Real-time vintage of g_cma: at each t the centred filter is applied to y[:t+1]
    only, so its end-point value is an asymmetric (trailing) average."""
    h = _cma_w(m)//2
    return pd.Series(y).rolling(h+1, min_periods=1).mean().values[:, None]

LAM = 1e4
def _hp_trend(x, lam=LAM):
    import scipy.sparse as sp, scipy.sparse.linalg as sl
    n = len(x); D = sp.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
    return sl.spsolve((sp.eye(n) + lam*(D.T@D)).tocsc(), x)
def g_hp(y, t0, m):
    """Hodrick-Prescott trend fitted to the full series (two-sided smoother)."""
    return _hp_trend(y)[:, None]
def g_hp_rt(y, t0, m, lam=LAM):
    """Real-time vintage of g_hp: the end-point of the HP trend fitted to y[:t+1], for
    every t. Computed exactly with the Kalman filter of the integrated-random-walk model
    whose smoother is the HP filter (noise ratio 1/lam), in O(T)."""
    F = np.array([[2., -1.], [1., 0.]]); Q = np.array([[1./lam, 0.], [0., 0.]])
    x = np.array([y[0], y[0]]); P = np.eye(2)*1e7; out = np.empty(len(y))
    for t in range(len(y)):
        if t > 0: x = F@x; P = F@P@F.T + Q
        S = P[0, 0] + 1.0; K = P[:, 0]/S; x = x + K*(y[t]-x[0])
        P = P - np.outer(K, P[0, :]); out[t] = x[0]
    return out[:, None]

GENS = {  # name: (full-sample generator, real-time generator or None if causal)
    'INV':  (g_inv,  None),
    'SES':  (g_ses,  None),
    'NV':   (g_nv,   None),
    'ROLL': (g_roll, None),
    'CMA':  (g_cma,  g_cma_rt),
    'HP':   (g_hp,   g_hp_rt),
}
CELLS = []   # (generator, protocol, admissible?)
for gname, (g, grt) in GENS.items():
    CELLS.append((gname, 'T', False))
    CELLS.append((gname, 'O', grt is None))
    if grt is not None: CELLS.append((gname, 'O*', True))

def gen_matrix(gname, proto, y, t0, m):
    g, grt = GENS[gname]
    return (grt if proto == 'O*' else g)(y, t0, m)

def gen_index(proto, j, k, shift=0):
    """Row of the generator output paired with target y[j+k] when the origin is j."""
    return (j+k if proto == 'T' else j) - shift
