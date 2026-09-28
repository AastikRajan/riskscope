# RiskScope: build spec

A calibrated, explainable health-risk model served as a small API, built on public CDC
survey data. It should read as the work of a junior data scientist who knows
imbalanced classification, calibration, fairness checks, explainability and serving.

## Data (already in the repo; no download needed)
`data/diabetes_binary_health_indicators_BRFSS2015.csv` and the other CSVs from the
Kaggle dataset "Diabetes Health Indicators Dataset" (alexteboul, CC0), built from the
CDC's Behavioral Risk Factor Surveillance System 2015 (253,680 survey responses, 21
features). Use the binary diabetes file as the main target.
Cite: CDC BRFSS 2015; Kaggle dataset alexteboul/diabetes-health-indicators-dataset.

## Build
1. `riskscope/data.py`: load the data and make a stratified train / validation / test split
   with a fixed seed. Describe the class balance.
2. `riskscope/train.py`: a logistic-regression baseline and a gradient-boosted model
   (LightGBM or sklearn HistGradientBoosting; CPU only). Tune lightly on validation.
   Report ROC-AUC, PR-AUC, Brier score and recall at a chosen screening threshold on
   the test set.
3. Calibration: isotonic or Platt on validation. A reliability plot saved to
   `reports/`.
4. Explainability: SHAP (TreeExplainer) global importance and one local explanation.
   If SHAP fails to install, use permutation importance and say so.
5. Fairness slice: metrics by sex and age group. Note the gaps plainly.
6. `app/main.py`: a FastAPI `/predict` that returns the probability, the risk band and
   the top 3 contributing features. `/health` too.
7. `tests/`: split has no leakage (ids are disjoint), API schema, calibration improves
   Brier on validation, and a latency check on 100 requests.

## README
The problem and why calibration matters for screening. How to run it. A results table
and the fairness table with the run's real numbers. A clear "not medical advice, survey
data, limits" section. Plain, confident wording. Don't mention how the code was written.

## Done means
`pytest -q` passes, `reports/metrics.md` exists, and the README shows it. Commit and push
to a new branch `build-1`.
