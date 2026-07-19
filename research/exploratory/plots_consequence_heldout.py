"""Honest figures for EXP-1/EXP-2 (raw main-script plots were misleading alone)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CH = REPO / "research/data/artifacts/v1/consequence_heldout"
PLOTS = REPO / "research/plots"
BLUE, RED, GRAY, GREEN, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#008300", "#0b0b0b", "#52514e"

dirauc = json.load(open(CH / "layer_sweep_auc.json"))
d1 = json.load(open(CH / "diag_norm_by_layer.json"))
Ls = sorted(int(k) for k in d1)
normauc = [d1[str(L)]["auc_norm"] for L in Ls]

# EXP-1: direction (flat) vs norm (rising)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(Ls, dirauc["both"], "-o", ms=3, color=GRAY, label="direction (unit-normalized delta)")
ax.plot(Ls, normauc, "-o", ms=3, color=BLUE, label="magnitude (delta norm)")
ax.axhline(0.5, color=INK, ls=":", lw=1)
ax.axvline(20, color=GREEN, ls="--", lw=1)
ax.annotate("L20 read layer", (20, 0.46), color=GREEN, fontsize=8, rotation=90, va="bottom")
ax.set_xlabel("layer (hidden_states index; final prompt position)")
ax.set_ylabel("target-vs-distractor AUC")
ax.set_ylim(0.42, 1.02); ax.legend(frameon=False, fontsize=9, loc="center left")
ax.set_title("EXP-1: relevance at the final position is MAGNITUDE, and LATE\n"
             "no readable 'relevance direction' at any layer (grey~0.5); the answer-change norm\n"
             "separates target/distractor only as the answer forms (L24-28)", fontsize=9.5)
fig.tight_layout(); fig.savefig(PLOTS / "2026-07-19_final_layer_sweep.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_final_layer_sweep.png")

# EXP-2: held-out consistency (D2) + the wrong-target artifact noted
nn = json.load(open(CH / "diag_heldout_1nn.json"))
ret = json.load(open(CH / "heldout_retrieval.json"))
fig, ax = plt.subplots(figsize=(8.2, 3.6))
bars = [
    ("unseen-value 1-NN consistency\n(same color from other families)", nn["heldout_1nn_acc"], nn["heldout_chance"], BLUE),
    ("seen-value 1-NN (reference)", nn["seen_1nn_acc"], 1 / nn["seen_nclass"], GRAY),
    ("unseen-value → input-embedding\nretrieval (WRONG target)", ret["value_new"]["heldout_value_top1"], ret["chance_top1"], RED),
    ("seen-value → input-embedding\nretrieval (same wrong target)", ret["value_new"]["dev_seenvalue_heldfamily_top1"], ret["chance_top1"], "#f4b7b6"),
]
y = np.arange(len(bars))
ax.barh(y, [b[1] for b in bars], color=[b[3] for b in bars])
for yi, b in enumerate(bars):
    ax.plot([b[2]] * 2, [yi - 0.4, yi + 0.4], color=INK, ls=":", lw=1.1)
    ax.annotate(f"{b[1]:.2f}", (b[1], yi), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("accuracy (dotted = chance)")
ax.set_title("EXP-2: unseen VALUES are consistently represented (top 2); the 0.00 retrieval (bottom)\n"
             "was a wrong-target artifact, not non-generalization", fontsize=9.5)
fig.tight_layout(); fig.savefig(PLOTS / "2026-07-19_heldout_value_retrieval.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_heldout_value_retrieval.png")
print("DONE")
