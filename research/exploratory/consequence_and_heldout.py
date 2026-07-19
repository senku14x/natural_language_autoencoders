"""Two experiments in one job:

EXP-1  Final-position LAYER SWEEP. At the final prompt position, extract the
       delta at every layer (hidden_states[0..28]) for all eligible dev change
       + distractor rows. Metric: target-vs-distractor discriminability (ROC
       AUC, GroupKFold by family) per layer, split by query order. Answers:
       "where does the behavioral consequence / relevance become readable, and
       is the edit-site blindness specific to the edit site?"
       Prediction: AUC ~0.5 early, rises in late layers at the final position
       (the query has attended back) in BOTH query orders — the consequence is
       a late/final phenomenon = answer formation, dissociated from the edit
       site where it is never readable.

EXP-2  HELD-OUT VALUE generalization (lookup vs general token basis). Fit a
       ridge map delta -> new-value TOKEN EMBEDDING on dev colors, then test
       top-1 retrieval on test_value rows whose colors were NEVER in training
       (charcoal, cream, green, lavender). Answers: "is the transition reading
       a per-value lookup, or a value-general token basis that survives unseen
       values?" A closed-set classifier can't score held-out values; an
       embedding-regression can iff the basis is general.
       Prediction: given this is token identity in an embedding-like basis,
       held-out retrieval ~= seen retrieval, >> chance -> general token basis,
       generalizes to unseen values (confirms token identity, not a lookup).

Probe-only. Nothing here trains the AV. Writes results JSON + plots.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/stage_a"))
from stage_a_lib import CANON_B, CANON_L, load_model_and_tokenizer, left_pad_batch
from run_site_decomposition_lib import seq_forward_factory

OUT = REPO / "research/data/artifacts/v1/consequence_heldout"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS = REPO / "research/plots"
MODEL = str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
            "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
SEED = 20260719
rng = np.random.default_rng(SEED)
unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
w = df.set_index("pair_id")
sc = pd.read_parquet(REPO / "research/data/artifacts/v1/stage_a/screen.parquet")
audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
# value -> leading_space single-token id (raw-format answer form)
ls_id = {x["value"]: x["token_ids"][0] for x in audit
         if x["kept"] and x["form"] == "leading_space" and x["n_tokens"] == 1}
color_vals = sorted({x["value"] for x in audit if x["kind"] == "color" and x["value"] in ls_id})

model, tok = load_model_and_tokenizer(MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
seq_forward = seq_forward_factory(model, pad_id)  # returns (final logits, L20 all-pos)
E = model.get_input_embeddings().weight.detach().float().cpu().numpy()
n_layers = model.config.num_hidden_layers  # 28


@torch.no_grad()
def alllayer_final(id_lists):
    """Final-position hidden states at every layer. Returns [N, n_layers+1, d]."""
    outs = []
    for lo in range(0, len(id_lists), CANON_B):
        chunk = list(id_lists[lo:lo + CANON_B]); n = len(chunk)
        while len(chunk) < CANON_B:
            chunk.append(chunk[-1])
        ids, mask, pos = left_pad_batch(chunk, pad_id, pad_to=CANON_L)
        hs = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                   output_hidden_states=True).hidden_states
        stk = torch.stack([h[:n, -1, :].float().cpu() for h in hs], dim=1)  # [n, L+1, d]
        outs.append(stk)
    return torch.cat(outs).numpy()


# ══ EXP-1 extraction: all-layer final-position delta, dev change+distractor ══
dev = sc[(sc.split == "dev") & sc.eligible
         & sc.cell.isin(["TARGET_EDIT", "REVERSE", "DISTRACTOR_EDIT"])].copy()
for c in ["family_id", "query_order"]:
    dev[c] = w.loc[dev.pair_id, c].to_numpy()
dev = dev.reset_index(drop=True)
print(f"EXP-1 rows: {len(dev)} ({dev.cell.value_counts().to_dict()})")
fb = alllayer_final([list(w.loc[p, "base_input_ids"]) for p in dev.pair_id])
fc = alllayer_final([list(w.loc[p, "cf_input_ids"]) for p in dev.pair_id])
fdelta = fc - fb  # [N, L+1, d]
np.save(OUT / "final_delta_alllayers.npy", fdelta.astype(np.float16))
is_target = dev.cell.isin(["TARGET_EDIT", "REVERSE"]).astype(int).to_numpy()

# ══ EXP-2 extraction: edit-site L20 delta for test_value (held-out colors) ══
tv = df[(df.split == "test_value") & (df.prompt_format == "raw") & (df.preamble)
        & df.cell.isin(["TARGET_EDIT", "REVERSE"])].copy().reset_index(drop=True)
print(f"EXP-2 test_value rows (raw/pre change): {len(tv)}  "
      f"held-out colors: {sorted(set(tv.value_new) | set(tv.value_old))}")
_, hb20 = seq_forward([list(x) for x in tv.base_input_ids])
_, hc20 = seq_forward([list(x) for x in tv.cf_input_ids])
lens = np.array([len(x) for x in tv.base_input_ids])
p_edit = (CANON_L - lens) + tv.edit_pos.to_numpy()
ar = np.arange(len(tv))
tv_delta = (hc20[ar, p_edit] - hb20[ar, p_edit]).numpy()
np.save(OUT / "testvalue_edit_delta.npy", tv_delta.astype(np.float16))
tv[["pair_id", "semantic_id", "family_id", "value_old", "value_new"]].to_parquet(OUT / "testvalue_rows.parquet")

del model
torch.cuda.empty_cache()

# ══ EXP-1 analysis: target-vs-distractor AUC per layer, by query order ══
print("\n=== EXP-1: target-vs-distractor AUC by layer ===")
def make_pipe(ntr, nf):
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=min(64, ntr - 1, nf))),
                     ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])

layer_auc = {"query_last": [], "query_first": [], "both": []}
for L in range(n_layers + 1):
    X = unit(fdelta[:, L, :])
    for qo in ["query_last", "query_first", "both"]:
        m = np.ones(len(dev), bool) if qo == "both" else (dev.query_order == qo).to_numpy()
        Xm, ym, gm = X[m], is_target[m], dev.family_id.to_numpy()[m]
        aucs = []
        for tr, te in GroupKFold(5).split(Xm, ym, gm):
            if len(np.unique(ym[tr])) < 2 or len(np.unique(ym[te])) < 2:
                continue
            pipe = make_pipe(len(tr), Xm.shape[1]).fit(Xm[tr], ym[tr])
            aucs.append(roc_auc_score(ym[te], pipe.predict_proba(Xm[te])[:, 1]))
        layer_auc[qo].append(float(np.mean(aucs)))
    print(f"  L{L:2d}  last {layer_auc['query_last'][-1]:.3f}  "
          f"first {layer_auc['query_first'][-1]:.3f}  both {layer_auc['both'][-1]:.3f}")
json.dump(layer_auc, open(OUT / "layer_sweep_auc.json", "w"), indent=1)

# ══ EXP-2 analysis: embedding-regression retrieval, seen vs held-out ══
print("\n=== EXP-2: held-out-value retrieval (delta -> token embedding) ===")
ps = pd.read_parquet(REPO / "research/data/artifacts/v1/position_sweep/rows.parquet")
ps_hb = np.load(REPO / "research/data/artifacts/v1/position_sweep/h_edit_base.npy")
ps_hc = np.load(REPO / "research/data/artifacts/v1/position_sweep/h_edit_cf.npy")
ps_delta = ps_hc - ps_hb
S_change = (ps.stratum == "S").to_numpy() & ps.cell.isin(["TARGET_EDIT", "REVERSE"]).to_numpy()
Xdev, dev_rows = unit(ps_delta[S_change]), ps[S_change].reset_index(drop=True)
Xtv = unit(tv_delta)

cand_vals = [v for v in color_vals]  # candidate set = all 42 colors (seen + held-out)
cand_emb = unit(np.stack([E[ls_id[v]] for v in cand_vals]))
cand_idx = {v: i for i, v in enumerate(cand_vals)}

# TRUE held-out colors = appear in test_value endpoints but NEVER in the dev
# training rows (charcoal/cream/green/lavender). Retrieval must be scored ONLY
# on rows whose TARGET is a held-out color, else it isn't a generalization test.
dev_seen = set(dev_rows.value_old) | set(dev_rows.value_new)
held_out = sorted((set(tv.value_old) | set(tv.value_new)) - dev_seen)
print(f"  TRUE held-out colors (in test_value, never in dev-train): {held_out}")
assert all(v not in dev_seen for v in held_out)

def retrieval_topk(pred_emb, true_vals, k=(1, 3)):
    P = unit(pred_emb)
    order = np.argsort(-(P @ cand_emb.T), 1)
    return {f"top{kk}": float(np.mean([cand_idx.get(v, -1) in order[i, :kk]
            for i, v in enumerate(true_vals)])) for kk in k}

results2 = {"chance_top1": 1.0 / len(cand_vals), "n_candidates": len(cand_vals),
            "held_out_colors": held_out}
for tgt in ["value_new", "value_old"]:
    Ydev = np.stack([E[ls_id[v]] for v in dev_rows[tgt]])
    # dev generalization via GroupKFold by family (held-out FAMILIES, seen values)
    dev_hits1 = []
    for tr, te in GroupKFold(5).split(Xdev, np.zeros(len(Xdev)), dev_rows.family_id.to_numpy()):
        M = Ridge(alpha=100.0).fit(Xdev[tr], Ydev[tr])
        dev_hits1.append(retrieval_topk(M.predict(Xdev[te]), dev_rows[tgt].to_numpy()[te])["top1"])
    # held-out VALUES: train on ALL dev, retrieve on test_value rows whose TARGET is held-out
    M = Ridge(alpha=100.0).fit(Xdev, Ydev)
    ho = tv[tgt].isin(held_out).to_numpy()
    ho_ret = retrieval_topk(M.predict(Xtv[ho]), tv[tgt].to_numpy()[ho])
    # all test_value rows (target may be a seen color) — reference only
    all_ret = retrieval_topk(M.predict(Xtv), tv[tgt].to_numpy())
    results2[tgt] = {"dev_seenvalue_heldfamily_top1": float(np.mean(dev_hits1)),
                     "heldout_value_top1": ho_ret["top1"], "heldout_value_top3": ho_ret["top3"],
                     "n_heldout_rows": int(ho.sum()),
                     "all_testvalue_top1": all_ret["top1"]}
    print(f"  {tgt}: dev(seen-value,held-family) top1 {np.mean(dev_hits1):.3f} | "
          f"HELD-OUT-value top1 {ho_ret['top1']:.3f} top3 {ho_ret['top3']:.3f} "
          f"(n={ho.sum()}, chance {results2['chance_top1']:.3f})")
json.dump(results2, open(OUT / "heldout_retrieval.json", "w"), indent=2)

# ══ plots ══
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, RED, GRAY, GREEN, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#008300", "#0b0b0b", "#52514e"

# EXP-1 layer sweep line
fig, ax = plt.subplots(figsize=(8.4, 4.2))
xs = list(range(n_layers + 1))
ax.plot(xs, layer_auc["query_last"], "-o", ms=3, color=BLUE, label="query_last")
ax.plot(xs, layer_auc["query_first"], "-o", ms=3, color=RED, label="query_first")
ax.axhline(0.5, color=INK, ls=":", lw=1)
ax.axvline(20, color=GREEN, ls="--", lw=1)
ax.annotate("L20 (edit-site read layer)", (20, 0.52), color=GREEN, fontsize=8, rotation=90, va="bottom")
ax.set_xlabel("layer (hidden_states index; final prompt position)")
ax.set_ylabel("target-vs-distractor AUC")
ax.set_ylim(0.45, 1.0); ax.legend(frameon=False, fontsize=9)
ax.set_title("EXP-1: where does behavioral relevance become readable? (final position, by layer)\n"
             "edit-site L20 was blind (AUC~0.5); does the final position know later?", fontsize=10)
fig.tight_layout(); fig.savefig(PLOTS / "2026-07-19_final_layer_sweep.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_final_layer_sweep.png")

# EXP-2 bars
fig, ax = plt.subplots(figsize=(7.4, 3.2))
labels, vals, cols = [], [], []
for tgt in ["value_new", "value_old"]:
    labels += [f"{tgt}\ndev (seen value)", f"{tgt}\nHELD-OUT value"]
    vals += [results2[tgt]["dev_seenvalue_heldfamily_top1"], results2[tgt]["heldout_value_top1"]]
    cols += [GRAY, BLUE]
y = np.arange(len(labels))
ax.barh(y, vals, color=cols)
ax.axvline(results2["chance_top1"], color=INK, ls=":", lw=1.2)
ax.annotate(f"chance {results2['chance_top1']:.2f}", (results2["chance_top1"], len(labels) - 0.5),
            fontsize=8, color=INK)
for yi, v in enumerate(vals):
    ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xlabel("token-embedding retrieval top-1 (delta -> value token)")
ax.set_title("EXP-2: does value reading generalize to UNSEEN values? (lookup vs general basis)", fontsize=10)
fig.tight_layout(); fig.savefig(PLOTS / "2026-07-19_heldout_value_retrieval.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_heldout_value_retrieval.png")
print("DONE")
