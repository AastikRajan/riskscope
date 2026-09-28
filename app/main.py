"""FastAPI service: calibrated risk probability, risk band and top contributing features."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "riskscope.joblib"
_art = joblib.load(ARTIFACT)
FEATURES: list[str] = _art["features"]
_booster = _art["model"].booster_


def _flag() -> "int":
    return Field(ge=0, le=1)


class Person(BaseModel):
    HighBP: int = _flag()
    HighChol: int = _flag()
    CholCheck: int = _flag()
    BMI: float = Field(ge=10, le=100)
    Smoker: int = _flag()
    Stroke: int = _flag()
    HeartDiseaseorAttack: int = _flag()
    PhysActivity: int = _flag()
    Fruits: int = _flag()
    Veggies: int = _flag()
    HvyAlcoholConsump: int = _flag()
    AnyHealthcare: int = _flag()
    NoDocbcCost: int = _flag()
    GenHlth: int = Field(ge=1, le=5, description="1 excellent .. 5 poor")
    MentHlth: int = Field(ge=0, le=30, description="bad mental health days, last 30")
    PhysHlth: int = Field(ge=0, le=30, description="bad physical health days, last 30")
    DiffWalk: int = _flag()
    Sex: int = _flag()
    Age: int = Field(ge=1, le=13, description="BRFSS age category, 1 = 18-24 .. 13 = 80+")
    Education: int = Field(ge=1, le=6)
    Income: int = Field(ge=1, le=8)


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float  # SHAP value, log-odds


class Prediction(BaseModel):
    probability: float
    risk_band: str
    flagged_for_screening: bool
    top_features: list[Contribution]


app = FastAPI(title="RiskScope", description="Calibrated diabetes risk from BRFSS 2015 survey features. Not medical advice.")


def band(p: float) -> str:
    if p >= _art["bands"]["high_min"]:
        return "high"
    return "low" if p < _art["bands"]["low_max"] else "moderate"


@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    X = pd.DataFrame([person.model_dump()])[FEATURES]
    raw = float(_art["model"].predict_proba(X)[0, 1])
    p = float(_art["calibrator"].predict([raw])[0])
    contrib = _booster.predict(X, pred_contrib=True)[0][:-1]  # LightGBM native tree SHAP; last column is the bias
    top = np.argsort(-np.abs(contrib))[:3]
    return Prediction(
        probability=round(p, 4),
        risk_band=band(p),
        flagged_for_screening=p >= _art["threshold"],
        top_features=[Contribution(feature=FEATURES[i], value=float(X.iloc[0, i]), contribution=round(float(contrib[i]), 4)) for i in top],
    )
