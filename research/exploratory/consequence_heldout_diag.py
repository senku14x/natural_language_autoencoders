"""Diagnostics to disambiguate the two surprising results before concluding.

D1 (EXP-1): the main sweep unit-NORMALIZED the final delta, discarding
    magnitude. But target rows change the answer (big final delta) and
    distractor rows do not (small). Report per-layer delta-NORM separation
    (target vs distractor) via single-feature AUC — the magnitude signal.

D2 (EXP-2): held-out retrieval against the INPUT token embedding was 0.000,
    but that may be the wrong target space, not proof of non-generalization.
    Test whether unseen-value deltas are internally CONSISTENT: leave-one-out
    1-NN among held-out rows (predict the color from other held-out deltas).
    If separable, unseen values ARE consistently represented (a seen-trained
    decoder just can't NAME them); if at chance, the unseen-value rep is not
    consistent. Compare to seen-color 1-NN (dev) as the reference ceiling.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CH = REPO / "research/data/artifacts/v1/consequence_heldout"
PS = REPO / "research/data/artifacts/v1/position_sweep"
unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

# ── reconstruct dev row order (same filter/order as the main script) ─────────
df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
w = df.set_index("pair_id")
sc = pd.read_parquet(REPO / "research/data/artifacts/v1/stage_a/screen.parquet")
dev = sc[(sc.split == "dev") & sc.eligible
         & sc.cell.isin(["TARGET_EDIT", "REVERSE", "DISTRACTOR_EDIT"])].copy().reset_index(drop=True)
is_target = dev.cell.isin(["TARGET_EDIT", "REVERSE"]).to_numpy()
fdelta = np.load(CH / "final_delta_alllayers.npy").astype(np.float32)  # [N, L+1, d]
assert len(dev) == len(fdelta)

# ── D1: per-layer delta-norm separation (target vs distractor) ───────────────
print("=== D1: final-position delta NORM, target vs distractor, by layer ===")
d1 = {}
for L in range(fdelta.shape[1]):
    nrm = np.linalg.norm(fdelta[:, L, :], axis=-1)
    t, dd = nrm[is_target], nrm[~is_target]
    # single-feature AUC = P(norm_target > norm_distractor)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(is_target.astype(int), nrm)
    d1[L] = {"auc_norm": float(auc), "median_target": float(np.median(t)),
             "median_distractor": float(np.median(dd))}
    if L in (0, 10, 20, 24, 28):
        print(f"  L{L:2d}  norm-AUC {auc:.3f}  median target {np.median(t):.1f} vs distractor {np.median(dd):.1f}")
json.dump(d1, open(CH / "diag_norm_by_layer.json", "w"), indent=1)
best = max(d1, key=lambda k: d1[k]["auc_norm"])
print(f"  best norm-AUC at L{best}: {d1[best]['auc_norm']:.3f}")

# ── D2: held-out-value internal consistency (1-NN among held-out) ────────────
print("\n=== D2: are UNSEEN-value deltas internally consistent? (1-NN) ===")
tv = pd.read_parquet(CH / "testvalue_rows.parquet")
tvd = unit(np.load(CH / "testvalue_edit_delta.npy").astype(np.float32))
ps = pd.read_parquet(PS / "rows.parquet")
psd = unit((np.load(PS / "h_edit_cf.npy") - np.load(PS / "h_edit_base.npy")).astype(np.float32))
Schange = (ps.stratum == "S").to_numpy() & ps.cell.isin(["TARGET_EDIT", "REVERSE"]).to_numpy()
dev_seen = set(ps[Schange].value_old) | set(ps[Schange].value_new)
held = sorted((set(tv.value_old) | set(tv.value_new)) - dev_seen)

def loo_1nn(X, labels, groups):
    """Leave-one-family-out 1-NN accuracy predicting label from nearest other-family row."""
    labels = np.asarray(labels); groups = np.asarray(groups)
    ok = []
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    for i in range(len(X)):
        mask = groups != groups[i]
        if mask.sum() == 0:
            continue
        j = np.where(mask)[0][np.argmax(S[i, mask])]
        ok.append(labels[i] == labels[j])
    return float(np.mean(ok)), len(ok)

# held-out: rows whose new value is a held-out color; label = that color
for tgt in ["value_new"]:
    ho = tv[tgt].isin(held).to_numpy()
    acc, n = loo_1nn(tvd[ho], tv[tgt].to_numpy()[ho], tv.family_id.to_numpy()[ho])
    nclass = len(set(tv[tgt].to_numpy()[ho]))
    print(f"  HELD-OUT colors 1-NN (predict unseen color from other held-out deltas): "
          f"{acc:.3f}  (n={n}, {nclass} colors, chance {1/nclass:.3f})")
    # seen reference: same on dev seen colors
    sacc, sn = loo_1nn(psd[Schange], ps[Schange].value_new.to_numpy(), ps[Schange].family_id.to_numpy())
    sclass = len(set(ps[Schange].value_new))
    print(f"  SEEN colors 1-NN (reference): {sacc:.3f}  (n={sn}, {sclass} colors, chance {1/sclass:.3f})")
    json.dump({"heldout_1nn_acc": acc, "heldout_n": n, "heldout_nclass": nclass,
               "heldout_chance": 1 / nclass, "seen_1nn_acc": sacc, "seen_nclass": sclass,
               "held_out_colors": held},
              open(CH / "diag_heldout_1nn.json", "w"), indent=1)
print("DONE")
