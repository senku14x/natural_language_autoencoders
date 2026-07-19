"""Decisive cheap test: is `value_old` LINEARLY recoverable from the edit-site
DELTA (d = h_cf - h_base), or does the delta surface only the new endpoint?

If old ~ delta is at chance while new ~ delta is well above → the delta carries
only the new token linearly, so any "old -> new" caption's "old" would be
confabulated/looked-up, not read (caution against "verbalize the diff").
If old ~ delta is well above chance → both endpoints are linearly present and
the AV's 0% zero-shot old-mention is a decoding limit, not information absence.

Linear probe = Pipeline(StandardScaler, PCA(64), LogisticRegression), fit
per-fold inside GroupKFold(by family) so nothing leaks. Direction-only
features (unit-normalized), matching the AV interface. Per stratum (colors S,
names N). Controls: positive (old~h_base, new~h_cf), negative (old~h_cf,
new~h_base), and a label-permutation sanity that must collapse to chance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/zeroshot_av"
SEED = 20260719
rng = np.random.default_rng(SEED)

pool = pd.read_parquet(OUT / "pool.parquet").reset_index(drop=True)
if "family_id" not in pool.columns:
    fam = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet",
                          columns=["pair_id", "family_id"])
    pool = pool.merge(fam, on="pair_id", how="left")
    assert pool.family_id.notna().all()
h_base = np.load(OUT / "h_edit_base.npy")
h_cf = np.load(OUT / "h_edit_cf.npy")
delta = h_cf - h_base
assert len(pool) == len(delta)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


FEATS = {"delta": unit(delta), "h_base": unit(h_base), "h_cf": unit(h_cf)}


def probe(X, y, groups, permute=False):
    y = np.asarray(y)
    if permute:
        y = rng.permutation(y)
    classes, ycodes = np.unique(y, return_inverse=True)
    K = len(classes)
    # need enough per class for CV; drop classes with <2 members
    keep = np.isin(ycodes, [c for c in range(K) if (ycodes == c).sum() >= 2])
    Xk, yk, gk = X[keep], ycodes[keep], np.asarray(groups)[keep]
    classes2, yk = np.unique(yk, return_inverse=True)
    K = len(classes2)
    n_splits = min(5, len(np.unique(gk)))
    gkf = GroupKFold(n_splits=n_splits)
    top1, top5 = [], []
    for tr, te in gkf.split(Xk, yk, gk):
        # only evaluate test classes seen in train
        seen = np.isin(yk[te], np.unique(yk[tr]))
        if seen.sum() == 0:
            continue
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=min(64, Xk[tr].shape[0] - 1, Xk.shape[1]))),
            ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ])
        pipe.fit(Xk[tr], yk[tr])
        proba = pipe.predict_proba(Xk[te][seen])
        cls = pipe.named_steps["lr"].classes_
        truth = yk[te][seen]
        pred1 = cls[proba.argmax(1)]
        top1.append((pred1 == truth).mean())
        order = cls[np.argsort(-proba, 1)[:, :5]]
        top5.append(np.mean([t in row for t, row in zip(truth, order)]))
    return {"top1": float(np.mean(top1)), "top1_std": float(np.std(top1)),
            "top5": float(np.mean(top5)), "n": int(keep.sum()),
            "n_classes": int(K), "chance": float(1.0 / K)}


ARMS = [
    ("old~delta", "delta", "value_old"),
    ("new~delta", "delta", "value_new"),
    ("old~h_base (pos ctrl)", "h_base", "value_old"),
    ("new~h_cf (pos ctrl)", "h_cf", "value_new"),
    ("old~h_cf (neg ctrl)", "h_cf", "value_old"),
    ("new~h_base (neg ctrl)", "h_base", "value_new"),
]

results = {}
for stratum in ["S", "N"]:
    m = (pool.stratum == stratum).to_numpy()
    sub = pool[m]
    g = sub.family_id.to_numpy()
    print(f"\n===== stratum {stratum}  (n={m.sum()}, "
          f"{sub.value_old.nunique()} old / {sub.value_new.nunique()} new classes) =====")
    results[stratum] = {}
    for name, feat, tgt in ARMS:
        r = probe(FEATS[feat][m], sub[tgt].to_numpy(), g)
        results[stratum][name] = r
        print(f"  {name:24s} top1 {r['top1']:.3f}±{r['top1_std']:.3f}  "
              f"top5 {r['top5']:.3f}  chance {r['chance']:.3f}  (n={r['n']}, K={r['n_classes']})")
    # permutation sanity on the key arm
    perm = probe(FEATS["delta"][m], sub.value_old.to_numpy(), g, permute=True)
    results[stratum]["old~delta (label-perm)"] = perm
    print(f"  {'old~delta (label-perm)':24s} top1 {perm['top1']:.3f}  chance {perm['chance']:.3f}")

json.dump(results, open(OUT / "oldvalue_probe.json", "w"), indent=2)

# ── plot ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PLOTS = REPO / "research/plots"
BLUE, GREEN, GRAY, RED, INK, MUTED = "#2a78d6", "#008300", "#9a9a94", "#e34948", "#0b0b0b", "#52514e"
plot_arms = ["new~delta", "old~delta", "old~h_base (pos ctrl)", "new~h_cf (pos ctrl)",
             "old~h_cf (neg ctrl)", "new~h_base (neg ctrl)", "old~delta (label-perm)"]
colors = {"new~delta": BLUE, "old~delta": RED, "old~h_base (pos ctrl)": GREEN,
          "new~h_cf (pos ctrl)": GREEN, "old~h_cf (neg ctrl)": GRAY,
          "new~h_base (neg ctrl)": GRAY, "old~delta (label-perm)": GRAY}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, stratum in zip(axes, ["S", "N"]):
    R = results[stratum]
    y = np.arange(len(plot_arms))
    vals = [R[a]["top1"] for a in plot_arms]
    ax.barh(y, vals, color=[colors[a] for a in plot_arms])
    for yi, a in enumerate(plot_arms):
        ax.plot([R[a]["chance"]] * 2, [yi - 0.4, yi + 0.4], color=INK, lw=1.2, ls=":")
        ax.annotate(f"{R[a]['top1']:.2f}", (R[a]["top1"], yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(plot_arms if stratum == "S" else [])
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("linear-probe top-1 accuracy (GroupKFold by family)")
    ax.set_title(f"stratum {stratum} ({'colors' if stratum=='S' else 'names'})", fontsize=10)
axes[0].annotate("dotted = chance (1/K)", (0.02, len(plot_arms) - 0.6), fontsize=7.5, color=INK)
fig.suptitle("Is value_old linearly recoverable from the edit-site delta?  "
             "(dev raw/preamble eligible pairs)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(PLOTS / "2026-07-19_oldvalue_probe.png", dpi=170)
print("\nwrote", PLOTS / "2026-07-19_oldvalue_probe.png")
print("DONE")
