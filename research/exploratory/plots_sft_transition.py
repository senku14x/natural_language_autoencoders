"""Plot for work item 2: SFT accuracy real vs shuffled by stratum, with the
analytic delta-ignoring floor, trivial-decoder and zero-shot references on the
same axes; plus training-loss curves for both runs.
Output: plots/2026-07-19_sft_transition_accuracy.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/sft_transition"
PLOTS = REPO / "research/plots"
BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#0b0b0b", "#52514e"

s = json.load(open(OUT / "sft_summary.json"))
loss_r = np.load(OUT / "loss_real.npy")
loss_s = np.load(OUT / "loss_shuffled.npy")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4),
                              gridspec_kw={"width_ratios": [1.35, 1]})

# ── panel A: pair-exact by arm × stratum ────────────────────────────────────
arms = [("real_x_real", "real-SFT × real Δ", BLUE),
        ("shuf_x_real", "shuffled-SFT × real Δ\n(Δ-ignoring floor, empirical)", GRAY),
        ("real_x_perm", "real-SFT × permuted Δ\n(own labels)", MUTED)]
groups = [("pooled", ""), ("S", "_S"), ("N", "_N")]
W = 0.26
xs = np.arange(len(groups))
for j, (arm, lab, col) in enumerate(arms):
    m = s[arm]
    vals, lo, hi = [], [], []
    for _, suf in groups:
        v = m["pair_acc" + suf] if suf else m["pair_acc"]
        ci = m[f"pair_ok{suf}_ci"] if suf else m["pair_ok_ci"]
        vals.append(v); lo.append(max(0, v - ci[0])); hi.append(max(0, ci[1] - v))
    ax.bar(xs + (j - 1) * W, vals, W * 0.92, color=col, label=lab,
           yerr=[lo, hi], error_kw=dict(ecolor=INK, lw=0.7, capsize=2))
    for x, v in zip(xs + (j - 1) * W, vals):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
fl = s["analytic_floor"]
ax.axhline(fl["pair"], color=RED, ls="--", lw=1.1)
ax.annotate(f"analytic Δ-ignoring floor {fl['pair']:.3f} "
            f"(mode {fl['mode_pair'][0]}→{fl['mode_pair'][1]})",
            (-0.42, 0.10), color=RED, fontsize=7.5, ha="left")
ax.axhline(0.04, color=MUTED, ls=":", lw=1.1)
ax.annotate("trivial decoder W_U·RMSNorm(Δ) top-5 = 0.04 (1b)", (2.62, 0.04),
            xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=7.5,
            ha="right")
ax.axhline(0.74, color=BLUE, ls=":", lw=1.1, alpha=0.7)
ax.annotate("released AV zero-shot\nnew-value string match 0.74", (0.5, 0.74),
            xytext=(0, -6), textcoords="offset points", color=BLUE, fontsize=7.5,
            ha="center", va="top", alpha=0.9)
ax.set_xticks(xs)
ax.set_xticklabels([f"pooled\n(n={s['dev_eligible']})", "colors (S)", "names (N)"])
ax.set_xlim(-0.45, 2.65)
ax.set_ylim(0, 1.12)
ax.set_ylabel("pair-exact accuracy (dev)")
ax.legend(loc="center right", frameon=False, fontsize=7.5)
ax.set_title("`old -> new` ordered-pair exact accuracy by arm and stratum\n"
             "(family-bootstrap 95% CI)", fontsize=10)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# ── panel B: training loss, real vs shuffled ────────────────────────────────
ax2.plot(np.arange(len(loss_r)), loss_r, color=BLUE, lw=1.0, label="real Δ run")
ax2.plot(np.arange(len(loss_s)), loss_s, color=GRAY, lw=1.0, label="shuffled Δ run")
ax2.set_yscale("log")
ax2.axhline(np.mean(loss_s[-50:]), color=RED, ls="--", lw=0.9)
ax2.annotate(f"shuffled plateau ≈ {np.mean(loss_s[-50:]):.2f} nats\n"
             "(≈ caption-prior entropy)", (len(loss_s) * 0.35, np.mean(loss_s[-50:])),
             xytext=(0, -22), textcoords="offset points", color=RED, fontsize=7.5)
spe = len(loss_r) // 3
for ep, c in enumerate(s["real_curve"]):
    x = (ep + 1) * spe - 1
    ax2.annotate(f"dev pair {c['pair_acc']:.3f}", (x, loss_r[min(x, len(loss_r)-1)]),
                 xytext=(-6, -14), textcoords="offset points", color=BLUE, fontsize=7)
    ax2.axvline(x, color=GRAY, ls=":", lw=0.5, alpha=0.5)
ax2.set_xlabel("step")
ax2.set_ylabel("train loss (nats, log scale)")
ax2.legend(loc="upper right", frameon=False, fontsize=8)
ax2.set_title("training loss — real converges, shuffled\nis pinned at the label prior",
              fontsize=10)
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)

fig.suptitle("AV SFT on edit-site deltas (work item 2) — LoRA on released kitft L20 AV, "
             "arrow-transition caption, dev raw/preamble eligible rows", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PLOTS / "2026-07-19_sft_transition_accuracy.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_sft_transition_accuracy.png")
