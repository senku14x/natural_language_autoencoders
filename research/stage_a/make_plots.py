"""Stage A figures. Every panel draws the trivial baselines on the same axes
as the method (project plotting norm #10).

Fig 1  margin-recovery distributions per condition, per eligible cell
Fig 2  behavioral-screen eligibility funnel per format x preamble cell
Fig 3  JS-recovery distributions per condition (same layout as Fig 1)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
PLOTS = REPO / "research/plots"
DATE = "2026-07-19"

# palette (dataviz reference instance, light mode)
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
AQUA, ORANGE, VIOLET, RED = "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"
GRAY, INK, MUTED = "#9a9a94", "#0b0b0b", "#52514e"

COND_ORDER = [
    ("real", "real Δ (own pair)", BLUE),
    ("mean_entity_transition", "E[Δ | entity, transition] (LOFO)", GREEN),
    ("mean_transition", "E[Δ | old→new] (LOFO)", AQUA),
    ("same_transition_other", "same-transition other pair", YELLOW),
    ("mean_global", "global mean Δ (LOFO)", ORANGE),
    ("output_token_direction", "W_U[new]−W_U[old], norm-matched", GRAY),
    ("unrelated_diff_transition", "unrelated Δ (diff. transition)", GRAY),
    ("matched_norm_random", "matched-norm random", GRAY),
    ("null_zero", "zero patch", GRAY),
    ("reverse_neg", "−Δ (reverse)", VIOLET),
]

pm = pd.read_parquet(OUT / "patch_metrics.parquet")
sc = pd.read_parquet(OUT / "screen.parquet")
chg = pm[pm.cell.isin(["TARGET_EDIT", "REVERSE"])].copy()
chg["cellkey"] = chg.prompt_format + " / " + np.where(chg.preamble, "preamble", "no preamble")
cells = [c for c in ["raw / preamble", "raw / no preamble", "chat / preamble",
                     "chat / no preamble"] if (chg.cellkey == c).any()]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "font.size": 9,
})


def strip_panel(ax, data, metric, xlim, gate=None):
    rng = np.random.default_rng(0)
    for yi, (cond, label, color) in enumerate(COND_ORDER):
        v = data.loc[data.condition == cond, metric].dropna().to_numpy()
        if len(v) == 0:
            continue
        vc = np.clip(v, xlim[0], xlim[1])
        jitter = rng.uniform(-0.22, 0.22, len(vc))
        ax.scatter(vc, yi + jitter, s=5, color=color, alpha=0.35, linewidths=0,
                   rasterized=True)
        med = float(np.median(v))
        ax.plot([np.clip(med, *xlim)] * 2, [yi - 0.32, yi + 0.32],
                color=color, lw=2.2, solid_capstyle="round")
        n_clip = int((v < xlim[0]).sum() + (v > xlim[1]).sum())
        note = f"{med:+.2f}" + (f"  (n={len(v)}, {n_clip} clipped)" if n_clip else f"  (n={len(v)})")
        ax.annotate(note, (xlim[1], yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.2, color=MUTED, annotation_clip=False)
    ax.axvline(0.0, color=MUTED, lw=0.9, ls=":")
    ax.axvline(1.0, color=MUTED, lw=0.9, ls=":")
    if gate is not None:
        ax.axvline(gate, color=RED, lw=1.1, ls="--")
    ax.set_yticks(range(len(COND_ORDER)))
    ax.set_yticklabels([l for _, l, _ in COND_ORDER])
    ax.set_ylim(len(COND_ORDER) - 0.5, -0.5)
    ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.02)


def condition_figure(metric, gate, xlabel, fname, xlim=(-1.5, 2.0)):
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells):
        d = chg[chg.cellkey == cell]
        n_pairs = d[d.condition == "real"].pair_id.nunique()
        strip_panel(ax, d, metric, xlim, gate)
        ax.set_title(f"{cell}   ({n_pairs} eligible pairs)", fontsize=9.5)
        ax.set_xlabel(xlabel)
    axes[0].annotate("gate 0.50", (gate, len(COND_ORDER) - 0.55), xytext=(3, 0),
                     textcoords="offset points", color=RED, fontsize=7.5)
    fig.suptitle(
        "Stage A patch-control matrix — dev split, Qwen2.5-7B-Instruct L20 final position\n"
        "median tick per condition; dotted 0 = no effect, dotted 1 = full counterfactual recovery",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(PLOTS / fname, dpi=170)
    print("wrote", PLOTS / fname)


condition_figure("margin_recovery", 0.50,
                 "normalized counterfactual-margin recovery",
                 f"{DATE}_stage_a_margin_recovery_matrix.png")

# ── Fig 4: site decomposition ───────────────────────────────────────────────
sd_path = OUT / "site_decomposition.parquet"
if sd_path.exists():
    sd = pd.read_parquet(sd_path)
    fin = chg[chg.condition == "real"][["pair_id", "margin_recovery"]].rename(
        columns={"margin_recovery": "v"})
    fin["condition"] = "final_only_real"
    both = pd.concat([sd.rename(columns={"margin_recovery": "v"})[["pair_id", "condition", "v"]], fin])

    SCOPES = [
        ("all_positions", "ALL positions (positive control)", INK),
        ("edit_and_final", "edit token + final", BLUE),
        ("edit_only_real", "edit token only", BLUE),
        ("edit_only_same_transition", "edit site, same-transition other pair", YELLOW),
        ("edit_only_unrelated", "edit site, unrelated Δ (diff. transition)", GRAY),
        ("edit_only_random", "edit site, matched-norm random", GRAY),
        ("edit_only_reverse", "edit site, −Δ (reverse)", VIOLET),
        ("post_edit_excl_edit", "post-edit positions (excl. edit)", MAGENTA),
        ("final_only_real", "final position only (the v1 primary site)", RED),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    rng = np.random.default_rng(0)
    xlim = (-0.35, 1.25)
    for yi, (cond, label, color) in enumerate(SCOPES):
        v = both.loc[both.condition == cond, "v"].dropna().to_numpy()
        if len(v) == 0:
            continue
        vc = np.clip(v, *xlim)
        ax.scatter(vc, yi + rng.uniform(-0.22, 0.22, len(vc)), s=5, color=color,
                   alpha=0.35, linewidths=0, rasterized=True)
        med = float(np.median(v))
        ax.plot([np.clip(med, *xlim)] * 2, [yi - 0.32, yi + 0.32], color=color,
                lw=2.4, solid_capstyle="round")
        ax.annotate(f"{med:+.3f}  (n={len(v)})", (xlim[1], yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=MUTED, annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=0.9, ls=":")
    ax.axvline(1, color=MUTED, lw=0.9, ls=":")
    ax.axvline(0.5, color=RED, lw=1.1, ls="--")
    ax.annotate("gate 0.50", (0.5, len(SCOPES) - 0.55), xytext=(3, 0),
                textcoords="offset points", color=RED, fontsize=7.5)
    ax.set_yticks(range(len(SCOPES)))
    ax.set_yticklabels([l for _, l, _ in SCOPES])
    ax.set_ylim(len(SCOPES) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("normalized counterfactual-margin recovery")
    ax.set_title(
        "Where does the counterfactual live at L20? — patch scope decomposition\n"
        "dev eligible pairs, Qwen2.5-7B-Instruct, block-20 output; median tick per scope",
        fontsize=10.5)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{DATE}_stage_a_site_decomposition.png", dpi=170)
    print("wrote", PLOTS / f"{DATE}_stage_a_site_decomposition.png")
condition_figure("js_recovery", 0.50,
                 "JS-to-counterfactual recovery (1 − JS_patch/JS_base)",
                 f"{DATE}_stage_a_js_recovery_matrix.png", xlim=(-1.0, 1.05))

# ── Fig 2: screen funnel ────────────────────────────────────────────────────
ch = sc[sc.cell.isin(["TARGET_EDIT", "REVERSE"])].copy()
ch["cellkey"] = ch.prompt_format + " / " + np.where(ch.preamble, "preamble", "no preamble")
fun = ch.groupby("cellkey").agg(base=("base_correct", "mean"),
                                both=("eligible", "mean"),
                                n=("eligible", "size"))
fun = fun.reindex(["raw / preamble", "raw / no preamble",
                   "chat / preamble", "chat / no preamble"])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(fun))
ax.barh(y - 0.19, fun.base * 100, height=0.34, color=GRAY,
        label="base prompt answered correctly")
ax.barh(y + 0.19, fun.both * 100, height=0.34, color=BLUE,
        label="eligible (base ∧ cf correct ∧ margin moves)")
for yi, (b, e) in enumerate(zip(fun.base, fun.both)):
    ax.annotate(f"{b:.0%}", (b * 100, yi - 0.19), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.annotate(f"{e:.0%}", (e * 100, yi + 0.19), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(fun.index)
ax.invert_yaxis()
ax.set_xlabel("% of dev TARGET+REVERSE rows (n=800 per cell)")
ax.set_xlim(0, 100)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Stage A behavioral screen — eligibility by format × preamble cell", fontsize=10.5)
fig.tight_layout()
fig.savefig(PLOTS / f"{DATE}_stage_a_screen_eligibility.png", dpi=170)
print("wrote", PLOTS / f"{DATE}_stage_a_screen_eligibility.png")
