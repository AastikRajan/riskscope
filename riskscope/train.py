"""Train baseline + gradient-boosted models, calibrate, explain, and run the fairness slice."""
import itertools
import json

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from riskscope.data import AGE_GROUPS, SEED, TARGET, describe_balance, feature_columns, load_data, split_data

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
REPORTS, MODELS = ROOT / "reports", ROOT / "models"
TARGET_RECALL = 0.80  # screening: catch ~80% of true cases on validation


def threshold_for_recall(y, p, recall):
    """Highest threshold whose recall on (y, p) is still >= `recall`."""
    order = np.argsort(-p)
    tp = np.cumsum(y.to_numpy()[order])
    k = int(np.searchsorted(tp / y.sum(), recall))
    return float(p[order][k])


def threshold_metrics(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def score(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def fit_lgbm(params, Xtr, ytr, Xva, yva):
    # No class reweighting: the screening threshold handles imbalance, and unweighted training keeps probabilities usable
    m = lgb.LGBMClassifier(n_estimators=600, random_state=SEED, verbose=-1, n_jobs=-1, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_data()
    feats = feature_columns(df)
    train, val, test = split_data(df)
    Xtr, ytr, Xva, yva, Xte, yte = train[feats], train[TARGET], val[feats], val[TARGET], test[feats], test[TARGET]
    balance = describe_balance(df)
    print(balance)

    # --- baseline
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    lr_val, lr_test = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]

    # --- light tuning of LightGBM on validation PR-AUC
    grid = [dict(num_leaves=nl, learning_rate=lrate, min_child_samples=mcs)
            for nl, lrate, mcs in itertools.product([15, 31], [0.03, 0.08], [50, 200])]
    best, best_pr = None, -1
    for params in grid:
        m = fit_lgbm(params, Xtr, ytr, Xva, yva)
        pr = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        print(params, m.best_iteration_, round(pr, 4))
        if pr > best_pr:
            best, best_pr, best_params = m, pr, params
    gbm = best
    raw_val, raw_test = gbm.predict_proba(Xva)[:, 1], gbm.predict_proba(Xte)[:, 1]

    # --- calibration (isotonic, fit on validation)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_val, yva)
    cal_val, cal_test = iso.predict(raw_val), iso.predict(raw_test)

    # screening threshold: on calibrated validation probabilities, recall >= TARGET_RECALL
    thr = threshold_for_recall(yva, cal_val, TARGET_RECALL)
    prev = float(ytr.mean())
    bands = {"low_max": round(prev / 2, 4), "high_min": thr}  # low < half the base rate; high = flagged by screen

    results = {
        "logistic_regression": {**score(yte, lr_test)},
        "lightgbm_raw": {**score(yte, raw_test), **threshold_metrics(yte, raw_test, threshold_for_recall(yva, raw_val, TARGET_RECALL))},
        "lightgbm_calibrated": {**score(yte, cal_test), **threshold_metrics(yte, cal_test, thr)},
    }
    lr_thr = threshold_for_recall(yva, lr_val, TARGET_RECALL)
    results["logistic_regression"].update(threshold_metrics(yte, lr_test, lr_thr))
    val_brier = {"raw": brier_score_loss(yva, raw_val), "calibrated": brier_score_loss(yva, cal_val)}

    # --- reliability plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, p in [("LightGBM raw", raw_test), ("LightGBM isotonic", cal_test), ("Logistic (balanced)", lr_test)]:
        frac, mean = calibration_curve(yte, p, n_bins=15, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Reliability (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "reliability.png", dpi=130)
    plt.close(fig)

    # --- explainability: SHAP on the (uncalibrated) tree model, raw log-odds space
    explainer = shap.TreeExplainer(gbm)
    sample = Xte.sample(3000, random_state=SEED)
    sv = explainer.shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[..., 1] if sv.ndim == 3 else sv
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    fig = plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=130)
    plt.close("all")
    # one local explanation: highest-risk person in the test sample
    i = int(np.argmax(gbm.predict_proba(sample)[:, 1]))
    local = pd.Series(sv[i], index=feats)
    local = local.reindex(local.abs().sort_values(ascending=False).index)[:6]
    local_txt = "\n".join(f"| {k} | {sample.iloc[i][k]:g} | {v:+.3f} |" for k, v in local.items())

    # --- fairness slice (calibrated model, test set, screening threshold)
    t = test.copy()
    t["p"] = cal_test
    t["sex"] = t["Sex"].map({0: "female", 1: "male"})
    t["age_group"] = t["Age"].map(AGE_GROUPS)
    rows = []
    for col in ("sex", "age_group"):
        for g, sub in sorted(t.groupby(col)):
            tm = threshold_metrics(sub[TARGET], sub["p"].to_numpy(), thr)
            rows.append({"slice": col, "group": g, "n": len(sub), "prevalence": sub[TARGET].mean(),
                         "mean_predicted": sub["p"].mean(), "roc_auc": roc_auc_score(sub[TARGET], sub["p"]),
                         "brier": brier_score_loss(sub[TARGET], sub["p"]), **tm})
    fair = pd.DataFrame(rows)

    # --- persist
    joblib.dump({"model": gbm, "calibrator": iso, "features": feats, "threshold": thr, "bands": bands}, MODELS / "riskscope.joblib")
    out = {"balance": balance, "best_params": best_params, "best_iteration": int(gbm.best_iteration_),
           "threshold": thr, "bands": bands, "target_recall": TARGET_RECALL, "results": results,
           "val_brier": val_brier, "split_sizes": [len(train), len(val), len(test)]}
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    fair.to_csv(REPORTS / "fairness.csv", index=False)
    (REPORTS / "metrics.md").write_text(render_md(out, fair, imp, local_txt))
    print((REPORTS / "metrics.md").read_text())


def render_md(o, fair, imp, local_txt):
    r = o["results"]
    lines = ["# RiskScope metrics", "", "All numbers below come from `python -m riskscope.train` (seed 42), evaluated on the held-out test set.", "",
             f"**Data.** {o['balance']}. Split sizes (train/val/test): {o['split_sizes']}.", "",
             f"**Screening threshold** (calibrated probability {o['threshold']:.4f}) chosen on validation for {o['target_recall']:.0%} recall. "
             f"LightGBM params: {o['best_params']}, best iteration {o['best_iteration']}.", "",
             "## Test-set results", "",
             "| Model | ROC-AUC | PR-AUC | Brier | Recall @ threshold | Precision @ threshold | FPR @ threshold |", "|---|---|---|---|---|---|---|"]
    for k, name in [("logistic_regression", "Logistic regression (balanced)"), ("lightgbm_raw", "LightGBM (raw)"), ("lightgbm_calibrated", "LightGBM + isotonic")]:
        m = r[k]
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['brier']:.4f} | {m['recall']:.3f} | {m['precision']:.3f} | {m['false_positive_rate']:.3f} |")
    lines += ["", f"Validation Brier: raw {o['val_brier']['raw']:.4f} -> calibrated {o['val_brier']['calibrated']:.4f}.", "",
              "Each model uses its own validation-chosen threshold for the recall column.", "",
              "## Fairness slice (calibrated LightGBM, test set, screening threshold)", "",
              "| Slice | Group | n | Prevalence | Mean predicted | ROC-AUC | Brier | Recall | Precision | FPR |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in fair.iterrows():
        lines.append(f"| {x.slice} | {x.group} | {x.n} | {x.prevalence:.3f} | {x.mean_predicted:.3f} | {x.roc_auc:.3f} | {x.brier:.4f} | {x.recall:.3f} | {x.precision:.3f} | {x.false_positive_rate:.3f} |")
    lines += ["", "## Global importance (mean |SHAP|, log-odds, 3,000 test rows)", "", "| Feature | mean abs SHAP |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in imp.head(8).items()]
    lines += ["", "## Local explanation (highest-risk row in that sample)", "", "| Feature | Value | SHAP |", "|---|---|---|", local_txt, "",
              "Figures: `reports/reliability.png`, `reports/shap_summary.png`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
