"""Plots for work items 1a (-delta AV arm) and 1b (trivial-decoder baseline).

Outputs:
  plots/2026-07-19_negdelta_av_mention_rates.png   (1a: all AV arms incl. -delta)
  plots/2026-07-19_trivial_decoder_vs_av.png       (1b: AV vs W_U readout, floors, probe)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/zeroshot_av"
PLOTS = REPO / "research/plots"
BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#0b0b0b", "#52514e"
SEED = 20260719
rng = np.random.default_rng(SEED)

prior = pd.read_parquet(OUT / "scored.parquet")
neg = pd.read_parquet(OUT / "negdelta_scored.parquet")
res = pd.concat([prior, neg], ignore_index=True)
lens = pd.read_parquet(OUT / "trivial_decoder.parquet")


def pair_bootstrap(d, col, nboot=2000):
    per_pair = d.groupby("semantic_id")[col].mean()
    vals = per_pair.to_numpy()
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(nboot)]
    return per_pair.mean(), np.percentile(boots, [2.5, 97.5])


# ── Fig 1: mention rates by arm, now including -delta ───────────────────────
ARM_LABEL = {"h_cf": "h_cf state (pos. control)", "real": "real Δ (own pair)",
             "neg_delta": "−Δ (negated own pair)", "transition_mean": "E[Δ|old,new] mean",
             "h_base": "h_base state", "shuffled": "shuffled Δ (diff. pair)",
             "random": "random Gaussian"}
order = ["h_cf", "real", "neg_delta", "transition_mean", "h_base", "shuffled", "random"]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
floor = float(res[res.arm.isin(["shuffled", "random"])]
              .groupby("semantic_id").new_hit.mean().mean())
y = np.arange(len(order))
for metric, color, off, lab in [("new_hit", BLUE, -0.22, "new value"),
                                ("old_hit", GRAY, 0.0, "old value"),
                                ("entity_hit", MUTED, 0.22, "edited entity")]:
    rates, los, his = [], [], []
    for a in order:
        d = res[res.arm == a]
        m, (lo, hi) = pair_bootstrap(d, metric)
        rates.append(m); los.append(m - lo); his.append(hi - m)
    ax.barh(y + off, rates, height=0.2, color=color, label=lab,
            xerr=[los, his], error_kw=dict(ecolor=MUTED, lw=0.7, capsize=2))
ax.axvline(floor, color=RED, ls="--", lw=1.2)
ax.annotate(f"shuffled/random new-value floor = {floor:.2f}",
            (floor, len(order) - 0.5), xytext=(5, 0), textcoords="offset points",
            color=RED, fontsize=8, va="center")
ax.set_yticks(y); ax.set_yticklabels([ARM_LABEL[a] for a in order])
ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xlabel("mention rate (pair-bootstrap 95% CI)")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Zero-shot AV read of edit-site deltas — with the −Δ arm (work item 1a)\n"
             "released kitft Qwen2.5-7B-L20 AV; 50 pairs × 5 samples @ T=1; −Δ names the OLD value",
             fontsize=10.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "2026-07-19_negdelta_av_mention_rates.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_negdelta_av_mention_rates.png")


# ── Fig 2: AV vs trivial decoder, same axes, floors + probe ceiling ─────────
def L(conv, arm, tgt, k, col="hit_rate"):
    r = lens[(lens.convention == conv) & (lens.arm == arm)
             & (lens.target == tgt) & (lens.topk == k)]
    return float(r[col].iloc[0]), float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])

av_real, (arl, arh) = pair_bootstrap(res[res.arm == "real"], "new_hit")
av_neg, (anl, anh) = pair_bootstrap(res[res.arm == "neg_delta"], "old_hit")
av_shuf, _ = pair_bootstrap(res[res.arm == "shuffled"], "new_hit")
av_real_old, _ = pair_bootstrap(res[res.arm == "real"], "old_hit")

probe = json.load(open(OUT / "oldvalue_probe.json"))

panels = [
    ("NEW value from +Δ", [
        ("released AV (string match,\n~220-token explanation)", av_real, arl, arh, BLUE),
        ("W_U·RMSNorm(Δ)  top-5", *L("rmsnorm", "pos_delta", "new", 5)[:1],
         *L("rmsnorm", "pos_delta", "new", 5)[1:], MUTED),
        ("W_U·RMSNorm(Δ)  top-1", *L("rmsnorm", "pos_delta", "new", 1)[:1],
         *L("rmsnorm", "pos_delta", "new", 1)[1:], MUTED),
        ("W_U·Δ (raw)  top-5", *L("raw", "pos_delta", "new", 5)[:1],
         *L("raw", "pos_delta", "new", 5)[1:], GRAY),
        ("W_U·RMSNorm(h_cf state)  top-5\n(instrument pos. control)",
         *L("rmsnorm", "h_cf", "new", 5)[:1], *L("rmsnorm", "h_cf", "new", 5)[1:], GRAY),
        ("shuffled-donor Δ, lens top-5", *L("rmsnorm", "shuffled_donor", "new", 5)[:1],
         *L("rmsnorm", "shuffled_donor", "new", 5)[1:], GRAY),
        ("shuffled Δ, AV", av_shuf, av_shuf, av_shuf, GRAY),
    ], ("linear probe (new~Δ)", None)),
    ("OLD value from −Δ", [
        ("released AV, −Δ (string match)", av_neg, anl, anh, BLUE),
        ("W_U·RMSNorm(−Δ)  top-5", *L("rmsnorm", "neg_delta", "old", 5)[:1],
         *L("rmsnorm", "neg_delta", "old", 5)[1:], MUTED),
        ("W_U·RMSNorm(−Δ)  top-1", *L("rmsnorm", "neg_delta", "old", 1)[:1],
         *L("rmsnorm", "neg_delta", "old", 1)[1:], MUTED),
        ("W_U·(−Δ) (raw)  top-5", *L("raw", "neg_delta", "old", 5)[:1],
         *L("raw", "neg_delta", "old", 5)[1:], GRAY),
        ("W_U·RMSNorm(h_base state)  top-5\n(instrument pos. control)",
         *L("rmsnorm", "h_base", "old", 5)[:1], *L("rmsnorm", "h_base", "old", 5)[1:], GRAY),
        ("+Δ, AV old-mention\n(sign convention: 0)", av_real_old, av_real_old,
         av_real_old, GRAY),
    ], ("linear probe (old~Δ)", None)),
]
# probe references: pooled across strata (colors 0.96 / names 0.76) — weight by sample
probe_new = 26/50*probe["S"]["new~delta"]["top1"] + 24/50*probe["N"]["new~delta"]["top1"]
probe_old = 26/50*probe["S"]["old~delta"]["top1"] + 24/50*probe["N"]["old~delta"]["top1"]
probe_refs = [probe_new, probe_old]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharex=True)
for ax, (title, bars, _), pref in zip(axes, panels, probe_refs):
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    lo = [max(0.0, b[1] - b[2]) for b in bars]
    hi = [max(0.0, b[3] - b[1]) for b in bars]
    colors = [b[4] for b in bars]
    yy = np.arange(len(bars))
    ax.barh(yy, vals, height=0.55, color=colors,
            xerr=[lo, hi], error_kw=dict(ecolor=MUTED, lw=0.7, capsize=2))
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axvline(pref, color=INK, ls=":", lw=1.2)
    ax.annotate(f"linear-probe decodability {pref:.2f}", (pref, len(bars) - 0.7),
                xytext=(-4, 0), textcoords="offset points", color=INK, fontsize=8,
                va="center", ha="right")
    ax.axvline(0.0, color=RED, ls="--", lw=1.0, alpha=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_xlabel("hit rate (50 pairs, bootstrap 95% CI)")
axes[1].set_xlabel("hit rate (50 pairs, bootstrap 95% CI)")
fig.suptitle("Trivial-decoder (unembedding) baseline vs released AV — edit-site L20 deltas (work item 1b)\n"
             "the W_U readout cannot name the value from Δ OR from the raw states; the AV can",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(PLOTS / "2026-07-19_trivial_decoder_vs_av.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_trivial_decoder_vs_av.png")
