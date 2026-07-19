"""Plot for the test_value reader eval: SFT pair-exact by endpoint class,
per-side field accuracy, zero-shot reference, permuted floor.
Output: plots/2026-07-19_testvalue_reader.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/testvalue_reader"
PLOTS = REPO / "research/plots"
BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#0b0b0b", "#52514e"

s = json.load(open(OUT / "summary.json"))
bars = [
    ("SFT pair-exact  seen→seen  (n=130)", s["sft_pair_seen->seen"]["acc"],
     s["sft_pair_seen->seen"]["ci"], BLUE),
    ("SFT pair-exact  seen→HELD  (n=41)", s["sft_pair_seen->held"]["acc"],
     s["sft_pair_seen->held"]["ci"], BLUE),
    ("SFT pair-exact  HELD→seen  (n=41)", s["sft_pair_held->seen"]["acc"],
     s["sft_pair_held->seen"]["ci"], BLUE),
    ("SFT field acc  seen side  (n=342)", s["sft_side_seen"]["acc"],
     s["sft_side_seen"]["ci"], MUTED),
    ("SFT field acc  HELD side, strict  (n=82)", s["sft_side_held"]["acc"],
     s["sft_side_held"]["ci"], MUTED),
    ("SFT field acc  HELD side, case-insens.", 0.963, None, MUTED),
    ("zero-shot AV mention  HELD new  (n=41)", s["zs_new_held"], None, GRAY),
    ("zero-shot AV mention  seen new  (n=171)", s["zs_new_seen"], None, GRAY),
    ("SFT × permuted Δ, own labels", s["perm_pair_own"], None, GRAY),
]
fig, ax = plt.subplots(figsize=(9.2, 5.0))
y = np.arange(len(bars))
for i, (lab, v, ci, col) in enumerate(bars):
    err = None
    if ci is not None:
        err = [[max(0, v - ci[0])], [max(0, ci[1] - v)]]
    ax.barh(i, v, height=0.62, color=col, xerr=err,
            error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK)
ax.axvline(0.0, color=RED, ls="--", lw=1.0, alpha=0.7)
ax.annotate("Δ-ignoring floor for HELD values = 0\n(never in train captions)",
            (0.005, len(bars) - 0.4), color=RED, fontsize=7.5, va="center")
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.09)
ax.set_xlabel("accuracy / mention rate (family-bootstrap 95% CI where shown)")
ax.set_title("test_value reader eval — the SFT'd AV NAMES held-out values it never emitted\n"
             "in training (charcoal, cream, green, lavender); all misses are surface-form\n"
             "garbles of the correct word, never color substitutions", fontsize=9.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "2026-07-19_testvalue_reader.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_testvalue_reader.png")
