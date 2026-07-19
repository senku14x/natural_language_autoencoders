"""Position-artifact diagnostics + corrected null for the nonlinear
blindness re-check (companion to recheck_blindness_nonlinear.py).

1. Metadata-only baseline: edit_pos alone / edit_pos+stratum+value classify
   target-vs-distractor under query_first (position shifts with the upstream
   query text) — reproducing the GBM-on-delta "dissociation" without any
   activations.
2. Corrected null: exact per-position class matching (50/50 within every
   edit_pos), same GBM pipeline — the delta signal collapses to chance in
   both query orders.

Run on cached position_sweep artifacts; no GPU. Report:
temporary_artifacts/2026-07-19_blindness_recheck_report.md
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

R = "research/data/artifacts/v1/position_sweep/"
rows = pd.read_parquet(R + "rows.parquet").reset_index(drop=True)
delta = np.load(R + "h_edit_cf.npy") - np.load(R + "h_edit_base.npy")
D = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8)
y = rows.cell.isin(["TARGET_EDIT", "REVERSE"]).astype(int).to_numpy()
g = rows.family_id.to_numpy()
ok = set(rows[y == 1].value_new) & set(rows[y == 0].value_new)
base = ((rows.prompt_format == "raw") & rows.preamble & rows.value_new.isin(ok)).to_numpy()

GBM = lambda s: Pipeline([("sc", StandardScaler()), ("pca", PCA(128, random_state=s)),
                          ("gb", HistGradientBoostingClassifier(max_iter=200, random_state=s))])


def cv_auc(X, yy, gg, model_fn, seed):
    out = []
    for tr, te in GroupKFold(5).split(X, yy, gg):
        if len(np.unique(yy[te])) < 2:
            continue
        out.append(roc_auc_score(yy[te], model_fn(seed).fit(X[tr], yy[tr])
                                 .predict_proba(X[te])[:, 1]))
    return float(np.mean(out))


for qo in ["query_first", "query_last"]:
    m = base & (rows.query_order == qo).to_numpy()
    sub, yy, gg = rows[m], y[m], g[m]
    print(f"=== {qo} (n={m.sum()}) ===")
    print(f"  edit_pos by class: target mean {sub[yy==1].edit_pos.mean():.1f} "
          f"| distractor mean {sub[yy==0].edit_pos.mean():.1f}")

    # metadata-only baselines (no activations)
    X1 = sub.edit_pos.to_numpy().reshape(-1, 1).astype(float)
    hgb = lambda s: HistGradientBoostingClassifier(max_iter=100, random_state=s)
    print(f"  AUC edit_pos alone:        {cv_auc(X1, yy, gg, hgb, 0):.3f}")
    Xm = pd.get_dummies(sub[["stratum", "value_new"]],
                        columns=["stratum", "value_new"]).astype(float)
    Xm["edit_pos"] = sub.edit_pos.to_numpy()
    print(f"  AUC metadata only:         {cv_auc(Xm.to_numpy(), yy, gg, hgb, 0):.3f}")

    # corrected null: exact per-position 50/50 matching, 5 seeds
    idx = np.where(m)[0]
    res = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        keep = []
        for pos, grp in rows.iloc[idx].groupby("edit_pos"):
            gi = grp.index.to_numpy()
            t, d0 = gi[y[gi] == 1], gi[y[gi] == 0]
            k = min(len(t), len(d0))
            if k:
                keep += list(r.choice(t, k, replace=False)) + list(r.choice(d0, k, replace=False))
        mm = np.array(keep)
        res.append(cv_auc(D[mm], y[mm], g[mm], GBM, seed))
    print(f"  GBM on delta, POSITION-MATCHED: {np.mean(res):.3f} "
          f"[{min(res):.3f}-{max(res):.3f}] (n≈{len(mm)})")
