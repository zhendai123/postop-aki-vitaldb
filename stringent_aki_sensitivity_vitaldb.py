# FINAL v1 - stringent AKI sensitivity, VitalDB
# Primary: KDIGO; >=1.5x baseline within 7d, OR >=0.3 rise vs any earlier
# postop value within 48h. Stringent: two concordant qualifying measurements.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_BOOT = 500

PATHS = {"VitalDB": dict(creat="vitaldb_creat.csv", cohort="vitaldb_cohort.csv")}

CONTINUOUS = ["age", "weight", "height", "albumin", "urea", "sodium", "potassium",
              "haemoglobin", "platelets", "glucose", "pt_inr", "aptt", "ast",
              "alt", "bilirubin", "crp"]
CATEGORICAL = ["sex", "asa_class", "emergency", "department"]
PREDICTORS = CONTINUOUS + CATEGORICAL


def prep(df):
    df = df[(df["creat_mgdl"] >= 0.2) & (df["creat_mgdl"] <= 25.0)]
    return df[(df["t_hours"] > 0) & (df["t_hours"] <= 168)]


def is_aki_primary(creat_long):
    df = prep(creat_long)
    out = {}
    for op, g in df.groupby("op_id"):
        g = g.sort_values("t_hours")
        base = g["baseline_creat_mgdl"].iloc[0]
        v = g["creat_mgdl"].to_numpy()
        t = g["t_hours"].to_numpy()
        hit = bool(np.any(v >= 1.5 * base))
        if not hit:
            for i in range(1, len(v)):
                dt = t[i] - t[:i]
                if np.any((dt <= 48) & (v[i] >= v[:i] + 0.3)):
                    hit = True
                    break
        out[op] = hit
    return pd.Series(out, name="aki_primary").astype(int)


def is_aki_stringent(creat_long):
    df = prep(creat_long)
    out = {}
    for op, g in df.groupby("op_id"):
        g = g.sort_values("t_hours")
        base = g["baseline_creat_mgdl"].iloc[0]
        v = g["creat_mgdl"].to_numpy()
        t = g["t_hours"].to_numpy()
        hit = bool(np.sum(v >= 1.5 * base) >= 2)
        if not hit:
            for i in range(1, len(v)):
                dt = t[i] - t[:i]
                risen = np.where((dt <= 48) & (v[i] >= v[:i] + 0.3))[0]
                for j in risen:
                    hit = bool(np.any((t[i:] > t[i]) & (v[i:] >= v[j] + 0.3)))
                    if hit:
                        break
                if hit:
                    break
        out[op] = hit
    return pd.Series(out, name="aki_stringent").astype(int)


def build_models():
    lr = Pipeline([
        ("imp", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), CONTINUOUS),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             CATEGORICAL)])),
        ("clf", LogisticRegression(penalty="l2", max_iter=5000))])
    lr_grid = GridSearchCV(lr, {"clf__C": np.logspace(-3, 2, 20)},
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    xgb = Pipeline([
        ("imp", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), CONTINUOUS),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             CATEGORICAL)])),
        ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", random_state=SEED, n_jobs=-1))])
    return {"LR": lr_grid, "XGB": xgb}


def calibration(p, y):
    eps = 1e-7
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    Xc = np.column_stack([np.ones_like(lg), lg])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return float(beta[0]), float(beta[1])


def main():
    report = []
    for cohort, files in PATHS.items():
        creat = pd.read_csv(files["creat"])
        coh = pd.read_csv(files["cohort"]).set_index("op_id")
        y_pri = is_aki_primary(creat).reindex(coh.index).fillna(0).astype(int)
        y_str = is_aki_stringent(creat).reindex(coh.index).fillna(0).astype(int)

        print("")
        print("=== " + cohort + " ===")
        print("events  primary " + str(y_pri.sum()) + " (" + format(y_pri.mean(), ".2%") +
              ")  |  stringent " + str(y_str.sum()) + " (" + format(y_str.mean(), ".2%") +
              ")  |  reclassified " + str((y_pri - y_str).clip(lower=0).sum()))

        for defn, y in [("primary", y_pri), ("stringent", y_str)]:
            X_tr, X_te, y_tr, y_te = train_test_split(
                coh[PREDICTORS], y, test_size=0.3, random_state=SEED, stratify=y)
            res = {}
            for label, est in build_models().items():
                est.fit(X_tr, y_tr)
                p = est.predict_proba(X_te)[:, 1]
                inter, slope = calibration(p, y_te.to_numpy())
                res[label] = dict(p=p,
                                  auroc=roc_auc_score(y_te, p),
                                  auprc=average_precision_score(y_te, p),
                                  brier=brier_score_loss(y_te, p),
                                  cal_intercept=inter, cal_slope=slope)
            rng = np.random.default_rng(SEED)
            yv = y_te.to_numpy()
            lp = res["LR"]["p"]
            lx = res["XGB"]["p"]
            diff = []
            for _ in range(N_BOOT):
                b = rng.integers(0, len(yv), len(yv))
                diff.append(roc_auc_score(yv[b], lp[b]) - roc_auc_score(yv[b], lx[b]))
            ci = np.percentile(diff, [2.5, 97.5])
            for label in ("LR", "XGB"):
                r = {k: v for k, v in res[label].items() if k != "p"}
                report.append(dict(cohort=cohort, definition=defn, model=label, **r))
                print("  [" + defn + "] " + label +
                      "  AUROC " + format(r["auroc"], ".3f") +
                      "  AUPRC " + format(r["auprc"], ".3f") +
                      "  Brier " + format(r["brier"], ".4f") +
                      "  cal slope " + format(r["cal_slope"], ".2f") +
                      "  intercept " + format(r["cal_intercept"], ".2f"))
            print("  [" + defn + "] LR-XGB paired bootstrap 95% CI: [" +
                  format(ci[0], ".3f") + ", " + format(ci[1], ".3f") + "]")

    pd.DataFrame(report).to_csv("stringent_sensitivity_results_vitaldb.csv", index=False)
    print("")
    print("Saved stringent_sensitivity_results_vitaldb.csv")


main()
