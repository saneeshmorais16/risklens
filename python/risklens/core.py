import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ['amount', 'hour', 'velocity_24h', 'distance_km', 'new_device']

def generate_transactions(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    customer = rng.integers(1, 151, n)
    amount = np.round(rng.lognormal(3.6, 0.9, n), 2)
    hour = rng.integers(0, 24, n)
    velocity = rng.poisson(2, n) + 1
    distance = rng.exponential(12, n)
    new_device = rng.binomial(1, .08, n)
    injected = np.zeros(n, dtype=int)
    ix = rng.choice(n, max(1, n // 25), replace=False)
    injected[ix] = 1
    amount[ix] *= rng.uniform(8, 20, len(ix))
    velocity[ix] += rng.integers(8, 22, len(ix))
    distance[ix] += rng.uniform(100, 1200, len(ix))
    new_device[ix] = 1
    return pd.DataFrame({'transaction_id': [f'TX-{i:06d}' for i in range(n)], 'customer_id': [f'C-{x:04d}' for x in customer], 'amount': np.round(amount, 2), 'hour': hour, 'velocity_24h': velocity, 'distance_km': np.round(distance, 1), 'new_device': new_device, 'injected_scenario': injected})

def analyze(df, contamination=.05):
    if not 0 < contamination <= .5:
        raise ValueError('contamination must be between 0 and 0.5')
    if len(df) < 3:
        raise ValueError('at least three transactions required')
    if df[FEATURES].isna().any().any():
        raise ValueError('missing feature values')
    model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    values = df[FEATURES].copy()
    values['amount'] = np.log1p(values['amount'].clip(lower=0))
    values['distance_km'] = np.log1p(values['distance_km'].clip(lower=0))
    model.fit(values)
    result = df.copy()
    result['anomaly_score'] = np.round(-model.decision_function(values), 4)
    result['flagged'] = model.predict(values) == -1
    def explain(row):
        reasons = []
        if row['amount'] > df['amount'].quantile(.95): reasons.append('Unusually large amount')
        if row['velocity_24h'] >= 8: reasons.append('High 24-hour transaction velocity')
        if row['distance_km'] >= 100: reasons.append('Large distance from usual location')
        if row['new_device']: reasons.append('New device')
        if row['hour'] < 5: reasons.append('Overnight activity')
        return '; '.join(reasons) or 'Multivariate pattern outside typical activity'
    result['review_reasons'] = result.apply(explain, axis=1)
    return result.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
