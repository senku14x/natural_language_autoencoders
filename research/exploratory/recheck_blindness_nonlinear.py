"""Re-check of the 'edit-site delta is behaviorally blind' null.
USER-SUPPLIED script (2026-07-19), run verbatim against cached position_sweep
artifacts. Claim under test: the AUC~0.5 null is a property of the linear
probe; a GBM finds ~0.63 under query_first with query_last at chance.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

LIN = lambda s: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))])
GBM = lambda s: Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=128, random_state=s)),
                          ("gb", HistGradientBoostingClassifier(max_iter=200, random_state=s))])


def auc(m, mk, seed, permute=False):
    X, yy, gg = D[m], y[m].copy(), g[m]
    if permute:
        r = np.random.default_rng(seed)
        fams = pd.unique(gg)
        mp = dict(zip(fams, r.permutation([yy[gg == f][0] for f in fams])))
        yy = np.array([mp[f] for f in gg])
    out = []
    for tr, te in GroupKFold(n_splits=5).split(X, yy, gg):
        if len(np.unique(yy[te])) < 2:
            continue
        out.append(roc_auc_score(yy[te], mk(seed).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]))
    return float(np.mean(out))


ok = set(rows[y == 1].value_new) & set(rows[y == 0].value_new)
base = ((rows.prompt_format == "raw") & rows.preamble & rows.value_new.isin(ok)).to_numpy()

print(f"{'cell':>34} | {'n':>5} | {'linear':>7} | {'GBM (5 seeds)':>22} | {'perm null':>10}")
for qo in ["query_first", "query_last"]:
    m = base & (rows.query_order == qo).to_numpy()
    lin = auc(m, LIN, 0)
    real = [auc(m, GBM, s) for s in range(5)]
    null = [auc(m, GBM, s, permute=True) for s in range(5)]
    tag = "  <- causal negative control" if qo == "query_last" else ""
    print(f"{'raw/pre value-matched ' + qo:>34} | {m.sum():5d} | {lin:7.3f} | "
          f"{np.mean(real):.3f} [{min(real):.3f}-{max(real):.3f}] | {np.mean(null):10.3f}{tag}")
