"""Position sweep + scaled probe + target/distractor discriminator (probe-only).

(A) Scaled old~delta / new~delta per cell + pooled, family-bootstrap CIs.
(B) Cross-position transfer (colors): train probe on one position bucket, test
    on another; new~delta AND new~h_cf (state positive control) — separates a
    rotated basis from position underinformativeness.
(C) target-vs-distractor discriminability at the edit site, AUC, GroupKFold by
    family, split by query_order (causal-availability test); plus decode the
    edited value from distractor deltas (identity-present sanity).

Registered predictions (before seeing results):
 - (A) names CI tightens vs the n=50 run; point estimates ~unchanged.
 - (B) transfer high among mid/late buckets; EARLY (pos 3-26) degrades for BOTH
   delta and state → underinformativeness, not a rotated basis.
 - (C) target/distractor AUC ~0.5 under query_last (edit token can't see the
   query); possibly >0.5 under query_first. Edited value decodable from
   distractor deltas regardless (identity is present, relevance is not).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/position_sweep"
PLOTS = REPO / "research/plots"
SEED = 20260719
rng = np.random.default_rng(SEED)

rows = pd.read_parquet(OUT / "rows.parquet").reset_index(drop=True)
hb = np.load(OUT / "h_edit_base.npy")
hc = np.load(OUT / "h_edit_cf.npy")
delta = hc - hb
unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
D, HC, HB = unit(delta), unit(hc), unit(hb)
change = rows.cell.isin(["TARGET_EDIT", "REVERSE"]).to_numpy()

BUCKET = np.where(rows.edit_pos < 30, "early(3-26)",
                  np.where(rows.edit_pos < 98, "late(74-97)", "later(98-122)"))
rows["bucket"] = BUCKET


def make_pipe(ntrain, nfeat):
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=min(64, ntrain - 1, nfeat))),
                     ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])


def cv_top1(X, y, groups, nboot=1000):
    y = np.asarray(y); groups = np.asarray(groups)
    _, yc = np.unique(y, return_inverse=True)
    keep = np.isin(yc, [c for c in range(yc.max() + 1) if (yc == c).sum() >= 2])
    X, yc, g = X[keep], yc[keep], groups[keep]
    _, yc = np.unique(yc, return_inverse=True)
    K = yc.max() + 1
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
    fam_acc = {}
    accs = []
    for tr, te in gkf.split(X, yc, g):
        seen = np.isin(yc[te], np.unique(yc[tr]))
        if seen.sum() == 0:
            continue
        pipe = make_pipe(len(tr), X.shape[1]).fit(X[tr], yc[tr])
        pred = pipe.predict(X[te][seen])
        truth = yc[te][seen]
        correct = (pred == truth).astype(float)
        accs.append(correct.mean())
        for fam, ok in zip(g[te][seen], correct):
            fam_acc.setdefault(fam, []).append(ok)
    # family-bootstrap CI
    per_fam = np.array([np.mean(v) for v in fam_acc.values()])
    boots = [np.mean(rng.choice(per_fam, len(per_fam), replace=True)) for _ in range(nboot)]
    return {"top1": float(np.mean(accs)), "ci": [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], "n": int(keep.sum()), "K": int(K)}


def transfer(Xtr, ytr, Xte, yte):
    """Train on all of (Xtr,ytr), eval top1 on test classes seen in train.
    Labels stay as their original string values throughout."""
    cls_tr = np.unique(ytr)
    seen = np.isin(yte, cls_tr)
    if seen.sum() == 0:
        return np.nan
    pipe = make_pipe(len(Xtr), Xtr.shape[1]).fit(Xtr, ytr)
    pred = pipe.predict(Xte[seen])  # returns original string labels
    return float((pred == yte[seen]).mean())


results = {}

# ── (A) scaled probe ─────────────────────────────────────────────────────────
print("=== (A) scaled old/new ~ delta (family-bootstrap CI) ===")
results["A"] = {}
for stratum in ["S", "N"]:
    m = change & (rows.stratum == stratum).to_numpy()
    sub = rows[m]
    for tgt in ["value_old", "value_new"]:
        r = cv_top1(D[m], sub[tgt].to_numpy(), sub.family_id.to_numpy())
        results["A"][f"{stratum}:{tgt}"] = r
        print(f"  {stratum} {tgt:10s}  top1 {r['top1']:.3f}  CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  "
              f"n={r['n']} K={r['K']} chance={1/r['K']:.3f}")

# ── (B) cross-position transfer (colors) ─────────────────────────────────────
print("\n=== (B) cross-position transfer, colors, new value ===")
Sm = change & (rows.stratum == "S").to_numpy()
buckets = ["early(3-26)", "late(74-97)", "later(98-122)"]
results["B"] = {"delta": {}, "h_cf": {}}
for feat_name, F in [("delta", D), ("h_cf", HC)]:
    print(f"  -- feature: {feat_name} (rows: new value) --")
    for bi in buckets:
        mi = Sm & (rows.bucket == bi).to_numpy()
        if mi.sum() < 10:
            continue
        line = []
        for bj in buckets:
            mj = Sm & (rows.bucket == bj).to_numpy()
            if mj.sum() < 5:
                line.append(np.nan); continue
            if bi == bj:
                acc = cv_top1(F[mi], rows[mi].value_new.to_numpy(),
                              rows[mi].family_id.to_numpy())["top1"]
            else:
                acc = transfer(F[mi], rows[mi].value_new.to_numpy(),
                               F[mj], rows[mj].value_new.to_numpy())
            line.append(acc)
            results["B"][feat_name][f"{bi}->{bj}"] = None if acc != acc else round(acc, 3)
        print(f"    train {bi:14s} -> " + "  ".join(
            f"{bj.split('(')[0]}:{('nan' if a!=a else f'{a:.2f}')}" for bj, a in zip(buckets, line)))

# clean within-raw/pre split (fixed format, pos 74-97, colors)
rp = change & (rows.prompt_format == "raw").to_numpy() & rows.preamble.to_numpy() & (rows.stratum == "S").to_numpy()
lo = rp & (rows.edit_pos < 86).to_numpy()
hi = rp & (rows.edit_pos >= 86).to_numpy()
results["B"]["clean_rawpre_lo->hi_delta"] = transfer(D[lo], rows[lo].value_new.to_numpy(), D[hi], rows[hi].value_new.to_numpy())
results["B"]["clean_rawpre_hi->lo_delta"] = transfer(D[hi], rows[hi].value_new.to_numpy(), D[lo], rows[lo].value_new.to_numpy())
print(f"  clean within-raw/pre (fixed format): lo->hi {results['B']['clean_rawpre_lo->hi_delta']:.3f}  "
      f"hi->lo {results['B']['clean_rawpre_hi->lo_delta']:.3f}  (n_lo={lo.sum()}, n_hi={hi.sum()})")

# ── (C) target vs distractor discriminability, by query order ────────────────
print("\n=== (C) target-vs-distractor at edit site (AUC, GroupKFold by family) ===")
results["C"] = {}
is_target = rows.cell.isin(["TARGET_EDIT", "REVERSE"]).astype(int).to_numpy()
for qo in ["query_last", "query_first", "both"]:
    m = np.ones(len(rows), bool) if qo == "both" else (rows.query_order == qo).to_numpy()
    X, y, g = D[m], is_target[m], rows[m].family_id.to_numpy()
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe = make_pipe(len(tr), X.shape[1]).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    results["C"][qo] = {"auc": float(np.mean(aucs)), "std": float(np.std(aucs)),
                        "n": int(m.sum()), "n_target": int(y.sum()), "n_distractor": int((~y.astype(bool)).sum())}
    print(f"  {qo:12s}  AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"(n={m.sum()}, tgt={y.sum()}, dist={(~y.astype(bool)).sum()})")

# edited value decodable from distractor deltas? (identity present regardless)
dist = (rows.cell == "DISTRACTOR_EDIT").to_numpy()
for stratum in ["S", "N"]:
    m = dist & (rows.stratum == stratum).to_numpy()
    if m.sum() < 20:
        continue
    r = cv_top1(D[m], rows[m].value_new.to_numpy(), rows[m].family_id.to_numpy())
    results["C"][f"edited_value_from_distractor_{stratum}"] = r
    print(f"  edited new-value from distractor delta [{stratum}]: top1 {r['top1']:.3f} "
          f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] chance {1/r['K']:.3f}")

json.dump(results, open(OUT / "probe_position_sweep.json", "w"), indent=2)

# ── plots ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, GREEN, RED, GRAY, INK, MUTED = "#2a78d6", "#008300", "#e34948", "#9a9a94", "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
# transfer heatmaps
for ax, feat in zip(axes, ["delta", "h_cf"]):
    M = np.full((3, 3), np.nan)
    for i, bi in enumerate(buckets):
        for j, bj in enumerate(buckets):
            k = f"{bi}->{bj}"
            if k in results["B"][feat] and results["B"][feat][k] is not None:
                M[i, j] = results["B"][feat][k]
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color=INK if M[i, j] < 0.6 else "white", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels([b.split("(")[0] for b in buckets], fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([b.split("(")[0] for b in buckets], fontsize=8)
    ax.set_xlabel("test position bucket"); ax.set_ylabel("train position bucket")
    ax.set_title(f"new-value transfer — {feat}", fontsize=10)
fig.suptitle("Cross-position transfer (colors): does the value basis stay the same across edit positions?\n"
             "delta vs state (h_cf) — if both drop at 'early', it's underinformativeness, not a rotated basis",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(PLOTS / "2026-07-19_position_transfer.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_position_transfer.png")

# discriminator bars
fig, ax = plt.subplots(figsize=(7.2, 3.2))
qos = ["query_last", "query_first", "both"]
aucs = [results["C"][q]["auc"] for q in qos]
stds = [results["C"][q]["std"] for q in qos]
y = np.arange(len(qos))
ax.barh(y, aucs, xerr=stds, color=[BLUE, RED, GRAY],
        error_kw=dict(ecolor=MUTED, lw=0.8, capsize=3))
ax.axvline(0.5, color=INK, ls=":", lw=1.2)
ax.annotate("chance (0.5)", (0.5, 2.4), fontsize=8, color=INK)
for yi, a in enumerate(aucs):
    ax.annotate(f"{a:.2f}", (a, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(qos); ax.invert_yaxis(); ax.set_xlim(0.4, 1.0)
ax.set_xlabel("target-vs-distractor discriminability at edit site (ROC AUC)")
ax.set_title("Can the edit-site delta tell a behaviorally-relevant edit from an irrelevant one?", fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS / "2026-07-19_target_distractor_auc.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_target_distractor_auc.png")
print("DONE")
