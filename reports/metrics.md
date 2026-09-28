# RiskScope metrics

All numbers below come from `python -m riskscope.train` (seed 42), evaluated on the held-out test set.

**Data.** 253,680 rows, 35,346 positive (diabetes or prediabetes) (13.9%); ratio about 1:6.2. Split sizes (train/val/test): [152208, 50736, 50736].

**Screening threshold** (calibrated probability 0.1304) chosen on validation for 80% recall. LightGBM params: {'num_leaves': 31, 'learning_rate': 0.03, 'min_child_samples': 200}, best iteration 269.

## Test-set results

| Model | ROC-AUC | PR-AUC | Brier | Recall @ threshold | Precision @ threshold | FPR @ threshold |
|---|---|---|---|---|---|---|
| Logistic regression (balanced) | 0.8206 | 0.3997 | 0.1773 | 0.797 | 0.297 | 0.305 |
| LightGBM (raw) | 0.8274 | 0.4277 | 0.0970 | 0.794 | 0.303 | 0.295 |
| LightGBM + isotonic | 0.8272 | 0.4191 | 0.0971 | 0.795 | 0.302 | 0.297 |

Validation Brier: raw 0.0966 -> calibrated 0.0964.

Each model uses its own validation-chosen threshold for the recall column.

## Fairness slice (calibrated LightGBM, test set, screening threshold)

| Slice | Group | n | Prevalence | Mean predicted | ROC-AUC | Brier | Recall | Precision | FPR |
|---|---|---|---|---|---|---|---|---|---|
| sex | female | 28430 | 0.129 | 0.131 | 0.837 | 0.0903 | 0.789 | 0.301 | 0.272 |
| sex | male | 22306 | 0.152 | 0.150 | 0.814 | 0.1057 | 0.802 | 0.304 | 0.330 |
| age_group | 18-34 | 4944 | 0.022 | 0.022 | 0.795 | 0.0198 | 0.250 | 0.208 | 0.021 |
| age_group | 35-49 | 9948 | 0.071 | 0.068 | 0.843 | 0.0556 | 0.583 | 0.259 | 0.127 |
| age_group | 50-64 | 17940 | 0.144 | 0.147 | 0.826 | 0.0992 | 0.794 | 0.309 | 0.299 |
| age_group | 65+ | 17904 | 0.205 | 0.205 | 0.762 | 0.1394 | 0.853 | 0.306 | 0.499 |

## Global importance (mean |SHAP|, log-odds, 3,000 test rows)

| Feature | mean abs SHAP |
|---|---|
| GenHlth | 0.603 |
| HighBP | 0.525 |
| Age | 0.404 |
| BMI | 0.393 |
| HighChol | 0.332 |
| Income | 0.122 |
| Sex | 0.115 |
| CholCheck | 0.085 |

## Local explanation (highest-risk row in that sample)

| Feature | Value | SHAP |
|---|---|---|
| GenHlth | 5 | +1.065 |
| BMI | 43 | +0.811 |
| HighBP | 1 | +0.679 |
| HeartDiseaseorAttack | 1 | +0.268 |
| HighChol | 1 | +0.256 |
| Age | 9 | +0.209 |

Figures: `reports/reliability.png`, `reports/shap_summary.png`.
