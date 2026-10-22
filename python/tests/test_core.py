import pytest
from risklens.core import generate_transactions, analyze

def test_reproducible_data():
    assert generate_transactions(100, 12).equals(generate_transactions(100, 12))

def test_unique_ids():
    assert generate_transactions(100).transaction_id.is_unique

def test_anomaly_outputs():
    out = analyze(generate_transactions(400))
    assert {'flagged', 'anomaly_score', 'review_reasons'} <= set(out.columns)
    assert out.flagged.sum() == 20
    assert out.anomaly_score.is_monotonic_decreasing

def test_bad_contamination():
    with pytest.raises(ValueError): analyze(generate_transactions(100), 0)

def test_missing_data():
    df = generate_transactions(100)
    df.loc[0, 'amount'] = float('nan')
    with pytest.raises(ValueError): analyze(df)
