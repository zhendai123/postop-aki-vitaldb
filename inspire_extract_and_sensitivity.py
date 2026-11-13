# INSPIRE extraction + stringent AKI sensitivity (FINAL v2, 30d baseline window)
# Reproduces: cohort 106,775 of 130,960 ops; primary 4,309 events; stringent 2,545.
# Outputs: inspire_cohort.csv, inspire_creat.csv, stringent_sensitivity_results_inspire.csv
import pandas as pd
import numpy as np

W_MIN = 10080      # postop window: 7 days in minutes
PRE_MIN = 43200    # preop baseline window: 30 days in minutes
NEEDED = ['creatinine', 'albumin', 'bun', 'sodium', 'potassium', 'hb',
          'platelet', 'glucose', 'ptinr', 'aptt', 'ast', 'alt',
          'total_bilirubin', 'crp']
NEEDED_SET = set(NEEDED)

ops = pd.read_csv('operations.csv.gz')
for c in ['anstart_time', 'opend_time', 'discharge_time', 'age', 'asa', 'weight', 'height']:
    ops[c] = pd.to_numeric(ops[c], errors='coerce')
print('raw operations:', len(ops), '(target 130960)')

n_cardiac = int(ops['cpboff_time'].notna().sum())
ops = ops[ops['cpboff_time'].isna()]
n_agetime = int(((ops['age'] < 18) | ops['anstart_time'].isna() | ops['discharge_time'].isna()).sum())
ops = ops[(ops['age'] >= 18) & ops['anstart_time'].notna() & ops['discharge_time'].notna()]
print('excluded cardiac (target 2522):', n_cardiac, '| age/time (target 14):', n_agetime,
      '| remaining:', len(ops))

ops_small = ops[['op_id', 'subject_id', 'anstart_time', 'discharge_time']]

pre_vals = {i: {} for i in NEEDED if i != 'creatinine'}
pre_cr = {}
post_cr = {}

k = 0
for chunk in pd.read_csv('labs.csv.gz',
                         usecols=['subject_id', 'chart_time', 'item_name', 'value'],
                         chunksize=1000000):
    k += 1
    chunk = chunk[chunk['item_name'].isin(NEEDED_SET)]
    if chunk.empty:
        continue
    chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
    chunk = chunk.dropna(subset=['value'])
    m = chunk.merge(ops_small, on='subject_id', how='inner')
    if m.empty:
        continue
    dt = m['chart_time'] - m['anstart_time']
    keep = (dt >= -PRE_MIN) & (dt <= W_MIN)
    m = m[keep].copy()
    m['dt'] = dt[keep]
    if m.empty:
        continue

    cr = m[m['item_name'] == 'creatinine']
    post = cr[cr['dt'] > 0]
    for op, g in post.groupby('op_id'):
        post_cr.setdefault(op, []).extend(list(zip(g['dt'] / 60.0, g['value'])))
    pre = cr[cr['dt'] <= 0]
    for op, g in pre.groupby('op_id'):
        g = g.sort_values('dt')
        pre_cr[op] = (float(g['dt'].iloc[-1]), float(g['value'].iloc[-1]))

    other = m[(m['item_name'] != 'creatinine') & (m['dt'] <= 0)]
    for item, g in other.groupby('item_name'):
        g = g.sort_values('dt')
        d = pre_vals.setdefault(item, {})
        for op, gg in g.groupby('op_id'):
            d[op] = float(gg['value'].iloc[-1])
    if k % 5 == 0:
        print('labs chunks processed:', k)

coh = ops.set_index('op_id').copy()
coh['baseline_creat_mgdl'] = pd.Series({op: v[1] for op, v in pre_cr.items()})
n_nobase = int(coh['baseline_creat_mgdl'].isna().sum())
coh = coh[coh['baseline_creat_mgdl'].notna()]
n_ckd = int((coh['baseline_creat_mgdl'] >= 4.0).sum())
coh = coh[coh['baseline_creat_mgdl'] < 4.0]
n_asa6 = int((coh['asa'] == 6).sum())
coh = coh[coh['asa'] != 6]
print('no baseline (target 1305):', n_nobase,
      '| CKD4-5 (target 1476):', n_ckd,
      '| ASA6 (target 56):', n_asa6,
      '| FINAL (target 125587):', len(coh))

ren = {'asa': 'asa_class', 'emop': 'emergency', 'bun': 'urea', 'hb': 'haemoglobin',
       'platelet': 'platelets', 'ptinr': 'pt_inr', 'total_bilirubin': 'bilirubin'}
for item, d in pre_vals.items():
    coh[item] = pd.Series(d)
coh = coh.rename(columns=ren)
KEEP = ['age', 'sex', 'weight', 'height', 'asa_class', 'emergency', 'department',
        'albumin', 'urea', 'sodium', 'potassium', 'haemoglobin', 'platelets',
        'glucose', 'pt_inr', 'aptt', 'ast', 'alt', 'bilirubin', 'crp',
        'baseline_creat_mgdl']
coh[KEEP].to_csv('inspire_cohort.csv')
print('wrote inspire_cohort.csv:', len(coh))

anstart = ops.set_index('op_id')['anstart_time']
disch = ops.set_index('op_id')['discharge_time']
rows = []
for op, vals in post_cr.items():
    if op not in coh.index:
        continue
    cap = (disch.get(op, np.nan) - anstart.get(op, np.nan)) / 60.0
    base = pre_cr.get(op, (np.nan, np.nan))[1]
    for t, v in vals:
        if not np.isnan(cap) and t > cap:
            continue
        rows.append((op, t, v, base))
pd.DataFrame(rows, columns=['op_id', 't_hours', 'creat_mgdl', 'baseline_creat_mgdl']).to_csv(
    'inspire_creat.csv', index=False)
print('wrote inspire_creat.csv:', len(rows))

SEED = 42
N_BOOT = 500
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


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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


report = []
creat = pd.read_csv("inspire_creat.csv")
ch = pd.read_csv("inspire_cohort.csv").set_index("op_id")
y_pri = is_aki_primary(creat).reindex(ch.index).fillna(0).astype(int)
y_str = is_aki_stringent(creat).reindex(ch.index).fillna(0).astype(int)
print("")
print("=== INSPIRE ===")
print("events  primary", y_pri.sum(), "(" + format(y_pri.mean(), ".2%") + ")",
      " |  stringent", y_str.sum(), "(" + format(y_str.mean(), ".2%") + ")",
      " |  reclassified", int((y_pri - y_str).clip(lower=0).sum()))

for defn, y in [("primary", y_pri), ("stringent", y_str)]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        ch[PREDICTORS], y, test_size=0.3, random_state=SEED, stratify=y)
    res = {}
    for label, est in build_models().items():
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X_te)[:, 1]
        inter, slope = calibration(p, y_te.to_numpy())
        res[label] = dict(p=p, auroc=roc_auc_score(y_te, p),
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
        r = {kk: vv for kk, vv in res[label].items() if kk != "p"}
        report.append(dict(cohort="INSPIRE", definition=defn, model=label, **r))
        print("  [" + defn + "] " + label +
              "  AUROC " + format(r["auroc"], ".3f") +
              "  AUPRC " + format(r["auprc"], ".3f") +
              "  Brier " + format(r["brier"], ".4f") +
              "  cal slope " + format(r["cal_slope"], ".2f") +
              "  intercept " + format(r["cal_intercept"], ".2f"))
    print("  [" + defn + "] LR-XGB paired bootstrap 95% CI: [" +
          format(ci[0], ".3f") + ", " + format(ci[1], ".3f") + "]")

pd.DataFrame(report).to_csv("stringent_sensitivity_results_inspire.csv", index=False)
print("")
print("Saved stringent_sensitivity_results_inspire.csv")
