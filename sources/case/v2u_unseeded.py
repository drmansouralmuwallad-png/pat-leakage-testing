import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('m5_store_category.csv')
N_TEST = 56
sales = df['y']


def simulate_twin(demand, seed):
    """Digital twin of a lost-sales (s, S) inventory with random lead times of 2-5 days.
    The state of day t is recorded after day t's demand has been served."""
    rng = np.random.default_rng(seed)
    d = demand.to_numpy(dtype=float)
    mu, sd = d[:-N_TEST].mean(), d[:-N_TEST].std()
    s = mu * 3.5 + 1.65 * sd * np.sqrt(3.5)
    S = s + mu * 7
    on_hand, pipeline = S, {}
    rows = []
    for t in range(len(d)):
        on_hand += pipeline.pop(t, 0.0)
        before = on_hand
        served = min(d[t], on_hand)
        on_hand -= served
        position = on_hand + sum(pipeline.values())
        if position <= s:
            lead = int(rng.integers(2, 6))
            pipeline[t + lead] = pipeline.get(t + lead, 0.0) + (S - position)
            position = S
        rows.append((position, on_hand, float(d[t] > before), served))
    tw = pd.DataFrame(rows, columns=['inv_position', 'on_hand', 'stockout', 'served'],
                      index=demand.index)
    tw['fill_rate_7d'] = (tw['served'].rolling(7, min_periods=1).sum()
                          / demand.rolling(7, min_periods=1).sum().clip(lower=1e-9))
    return tw.drop(columns='served')


feat = pd.DataFrame(index=df.index)
for k in range(1, 8):
    feat['lag_%d' % k] = sales.shift(k)
feat['lag_14'] = sales.shift(14)
feat['dow'] = np.arange(len(df)) % 7
for w in (7, 28):
    feat['roll_mean_%d' % w] = sales.shift(1).rolling(w).mean()
feat['roll_std_7'] = sales.shift(1).rolling(7).std()
feat['roll_min_7'] = sales.shift(1).rolling(7).min()
feat['roll_max_7'] = sales.shift(1).rolling(7).max()
twin = simulate_twin(sales, seed=None)
for c in twin.columns:
    feat['dt_' + c] = twin[c].shift(1)
feat = feat.iloc[35:]
target = sales.loc[feat.index]
X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=N_TEST, shuffle=False)
