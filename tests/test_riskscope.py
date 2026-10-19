import time

import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import brier_score_loss

from app.main import app
from riskscope.data import TARGET, feature_columns, load_data, split_data

SAMPLE = dict(HighBP=1, HighChol=1, CholCheck=1, BMI=34.0, Smoker=0, Stroke=0, HeartDiseaseorAttack=0, PhysActivity=0,
              Fruits=1, Veggies=1, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=4, MentHlth=5,
              PhysHlth=10, DiffWalk=1, Sex=0, Age=10, Education=4, Income=3)


@pytest.fixture(scope="module")
def splits():
    return split_data(load_data())


def test_split_has_no_leakage(splits):
    train, val, test = splits
    ids = [set(s.index) for s in splits]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(ids[0] | ids[1] | ids[2]) == len(load_data())
    rates = [s[TARGET].mean() for s in splits]
    assert max(rates) - min(rates) < 0.005  # stratified


def test_split_is_deterministic(splits):
    again = split_data(load_data())
    assert list(again[2].index) == list(splits[2].index)


def test_calibration_on_test_set(splits):
    import joblib
    import numpy as np
    from app.main import ARTIFACT
    art = joblib.load(ARTIFACT)
    test = splits[2]
    raw = art["model"].predict_proba(test[feature_columns(test)])[:, 1]
    cal = art["calibrator"].predict(raw)

    brier = brier_score_loss(test[TARGET], cal)
    assert brier < 0.098

    n_bins = 10
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(cal, bins) - 1
    ece = 0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_pred = np.mean(cal[mask])
            prob_true = np.mean(test[TARGET].values[mask])
            ece += np.abs(prob_pred - prob_true) * np.sum(mask) / len(test)

    assert ece < 0.005


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_schema():
    r = TestClient(app).post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert isinstance(body["flagged_for_screening"], bool)
    assert len(body["top_features"]) == 3
    assert set(body["top_features"][0]) == {"feature", "value", "contribution"}


def test_predict_rejects_bad_input():
    client = TestClient(app)
    assert client.post("/predict", json={**SAMPLE, "GenHlth": 9}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in SAMPLE.items() if k != "BMI"}).status_code == 422


def test_higher_risk_profile_scores_higher():
    client = TestClient(app)
    healthy = {**SAMPLE, "HighBP": 0, "HighChol": 0, "BMI": 22.0, "GenHlth": 1, "DiffWalk": 0, "Age": 3}
    assert client.post("/predict", json=SAMPLE).json()["probability"] > client.post("/predict", json=healthy).json()["probability"]


def test_latency_100_requests():
    client = TestClient(app)
    client.post("/predict", json=SAMPLE)  # warm up
    lat = []
    for _ in range(100):
        t = time.perf_counter()
        assert client.post("/predict", json=SAMPLE).status_code == 200
        lat.append(time.perf_counter() - t)
    lat.sort()
    assert sum(lat) / 100 < 0.05 and lat[94] < 0.1  # mean < 50 ms, p95 < 100 ms
