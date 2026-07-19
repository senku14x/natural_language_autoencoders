"""Stage A patch-control matrix on eligible dev rows (v1 doc §7 Experiment 1,
§8 causal-target controls; Amendment/backlog §2.1 mean hierarchy).

Patch = add vector at block-20 output, final position of the BASE prompt,
canonical-shape forwards throughout. Conditions per eligible TARGET/REVERSE row:

  real                       own delta (h_cf - h_base)
  reverse_neg                negated own delta
  null_zero                  analytic (zero patch is an exact identity, L0.3)
  unrelated_diff_transition  delta of another family, same format x preamble
                             cell, different UNORDERED value transition
                             (per Orientation-2 finding 4)
  same_transition_other      delta of another family, same ordered answer
                             transition, same cell (prototype probe; coverage
                             limited by transition multiplicity)
  matched_norm_random        N(0,I) direction scaled to own delta norm
  output_token_direction     W_U[new] - W_U[old] (canonical per-format form),
                             scaled to own delta norm
  mean_global                LOFO mean delta over eligible change rows in cell
  mean_transition            LOFO E[delta | ordered answer transition] in cell
  mean_entity_transition     LOFO E[delta | entity, ordered transition] in cell

Eligible DISTRACTOR rows get their own-delta patch (no-change metrics).

Metrics: margin recovery + direction; JS-to-cf recovery; top-10 overlap;
full-vocab argmax. Output: patch_metrics.parquet.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage_a_lib import canonical_forward, js_divergence, load_model_and_tokenizer, topk_overlap

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
MODEL_PATH = os.environ.get(
    "STAGE_A_MODEL_DIR",
    str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
)
SEED = 20260719
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
work = df[(df["split"] == "dev") | (df["cell"] == "NULL_AA")].reset_index(drop=True)
sc = pd.read_parquet(OUT / "screen.parquet")
assert list(sc["pair_id"]) == list(work["pair_id"]), "screen/work row-order mismatch"
h_b = torch.from_numpy(np.load(OUT / "h_base.npy"))
h_c = torch.from_numpy(np.load(OUT / "h_cf.npy"))
delta = h_c - h_b
meta = json.load(open(OUT / "screen_meta.json"))
chat_form = meta["chat_canonical_form_by_stratum"]

model, tok = load_model_and_tokenizer(MODEL_PATH)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
W_U = model.lm_head.weight.detach().float().cpu()

# recompute base/cf final logits (bitwise-identical under the canonical policy)
base_ids_all = [list(x) for x in work["base_input_ids"]]
cf_ids_all = [list(x) for x in work["cf_input_ids"]]
print("recomputing base/cf logits...")
logits_b, h_b2 = canonical_forward(model, base_ids_all, pad_id)
logits_c, _ = canonical_forward(model, cf_ids_all, pad_id)
assert (h_b2 - h_b).abs().max().item() == 0.0, "extraction not reproducible!"

audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
form_ids = {}
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids.setdefault(x["value"], {})[x["form"]] = x["token_ids"][0]

def canon_tok(row, which):
    form = ("leading_space" if row.prompt_format == "raw"
            else chat_form.get(row.stratum, "leading_space"))
    val = row.answer_old if which == "old" else row.answer_new
    return form_ids.get(val, {}).get(form)

sc = sc.reset_index(drop=True)
sc["row_idx"] = sc.index
change = sc[sc.cell.isin(["TARGET_EDIT", "REVERSE"]) & sc.eligible].copy()
distract = sc[(sc.cell == "DISTRACTOR_EDIT") & sc.eligible].copy()
change["tok_old"] = [canon_tok(r, "old") for r in change.itertuples(index=False)]
change["tok_new"] = [canon_tok(r, "new") for r in change.itertuples(index=False)]
change = change[change.tok_old.notna() & change.tok_new.notna()]
change["utrans"] = [frozenset((r.value_old, r.value_new)) for r in change.itertuples(index=False)]
change["otrans"] = change.answer_old + "->" + change.answer_new
change["cellkey"] = change.prompt_format + "/" + change.preamble.astype(str)
print(f"eligible change rows: {len(change)}, distractor rows: {len(distract)}")

# ── build per-row patch vectors for every condition ─────────────────────────
conds = {}
idx = change.row_idx.to_numpy()
conds["real"] = delta[idx]
conds["reverse_neg"] = -delta[idx]
own_norm = delta[idx].norm(dim=-1, keepdim=True)

g = torch.from_numpy(rng.standard_normal((len(change), delta.shape[1]))).float()
conds["matched_norm_random"] = g / g.norm(dim=-1, keepdim=True) * own_norm

u = W_U[change.tok_new.astype(int).to_numpy()] - W_U[change.tok_old.astype(int).to_numpy()]
conds["output_token_direction"] = u / u.norm(dim=-1, keepdim=True) * own_norm

def pick_partner(r, pool):
    cand = pool[pool.family_id != r.family_id]
    if len(cand) == 0:
        return None
    return int(cand.row_idx.to_numpy()[rng.integers(len(cand))])

unrel, same_t = [], []
by_cell = dict(tuple(change.groupby("cellkey")))
for r in change.itertuples(index=False):
    cell = by_cell[r.cellkey]
    j = pick_partner(r, cell[cell.utrans != r.utrans])
    unrel.append(j)
    j2 = pick_partner(r, cell[cell.otrans == r.otrans])
    same_t.append(j2)
conds["unrelated_diff_transition"] = torch.stack(
    [delta[j] if j is not None else torch.full((delta.shape[1],), np.nan) for j in unrel])
conds["same_transition_other"] = torch.stack(
    [delta[j] if j is not None else torch.full((delta.shape[1],), np.nan) for j in same_t])

change_pos = pd.Series(np.arange(len(change)), index=change.index)

def lofo_mean(group_cols):
    """Per-row leave-one-family-out mean of delta within (cellkey, *group_cols)."""
    out = torch.full((len(change), delta.shape[1]), float("nan"))
    for _, g in change.groupby(["cellkey"] + group_cols):
        d = delta[g.row_idx.to_numpy()]
        fam = g.family_id.to_numpy()
        pos = change_pos[g.index].to_numpy()
        for f in np.unique(fam):
            others = np.where(fam != f)[0]
            if len(others) == 0:
                continue
            mu = d[torch.from_numpy(others)].mean(0)
            for rf in np.where(fam == f)[0]:
                out[pos[rf]] = mu
    return out

conds["mean_global"] = lofo_mean([])
conds["mean_transition"] = lofo_mean(["otrans"])
conds["mean_entity_transition"] = lofo_mean(["entity_target", "otrans"])

# ── run ─────────────────────────────────────────────────────────────────────
records = []
base_lg_change = logits_b[idx]
cf_lg_change = logits_c[idx]
js_ref = js_divergence(base_lg_change, cf_lg_change)
tk_ref = topk_overlap(base_lg_change, cf_lg_change)
to = torch.from_numpy(change.tok_old.astype(int).to_numpy())
tn = torch.from_numpy(change.tok_new.astype(int).to_numpy())
ar = torch.arange(len(change))
m_base = (base_lg_change[ar, tn] - base_lg_change[ar, to]).numpy()
m_cf = (cf_lg_change[ar, tn] - cf_lg_change[ar, to]).numpy()

ids_change = [base_ids_all[i] for i in idx]
for cname, vec in conds.items():
    valid = ~torch.isnan(vec[:, 0])
    vi = valid.nonzero().squeeze(-1)
    print(f"condition {cname}: {len(vi)}/{len(change)} rows")
    if len(vi) == 0:
        continue
    lg, _ = canonical_forward(model, [ids_change[int(i)] for i in vi], pad_id,
                              patch=vec[vi])
    arv = torch.arange(len(vi))
    m_patch = (lg[arv, tn[vi]] - lg[arv, to[vi]]).numpy()
    js_p = js_divergence(lg, cf_lg_change[vi]).numpy()
    tk_p = topk_overlap(lg, cf_lg_change[vi]).numpy()
    am = lg.argmax(-1).numpy()
    for k, i in enumerate(vi.numpy()):
        r = change.iloc[i]
        denom = m_cf[i] - m_base[i]
        records.append({
            "pair_id": r.pair_id, "condition": cname,
            "margin_base": m_base[i], "margin_cf": m_cf[i],
            "margin_patch": m_patch[k],
            "margin_recovery": (m_patch[k] - m_base[i]) / denom if denom != 0 else np.nan,
            "direction_correct": bool(m_patch[k] > m_base[i]),
            "js_base_cf": float(js_ref[i]), "js_patch_cf": float(js_p[k]),
            "js_recovery": (float(1.0 - js_p[k] / float(js_ref[i]))
                            if float(js_ref[i]) > 1e-6 else np.nan),
            "topk_base_cf": float(tk_ref[i]), "topk_patch_cf": float(tk_p[k]),
            "argmax_patch": int(am[k]),
            "argmax_patch_decoded": tok.decode([int(am[k])]),
            "patch_norm": float(vec[i].norm()),
        })

# analytic null condition (zero patch == identity, L0.3)
for i in range(len(change)):
    r = change.iloc[i]
    denom = m_cf[i] - m_base[i]
    records.append({
        "pair_id": r.pair_id, "condition": "null_zero",
        "margin_base": m_base[i], "margin_cf": m_cf[i], "margin_patch": m_base[i],
        "margin_recovery": 0.0, "direction_correct": False,
        "js_base_cf": float(js_ref[i]), "js_patch_cf": float(js_ref[i]),
        "js_recovery": 0.0, "topk_base_cf": float(tk_ref[i]),
        "topk_patch_cf": float(tk_ref[i]),
        "argmax_patch": int(sc.loc[change.row_idx.iloc[i], "argmax_base"]),
        "argmax_patch_decoded": sc.loc[change.row_idx.iloc[i], "argmax_base_decoded"],
        "patch_norm": 0.0,
    })

# distractor rows: own delta, no-change metrics
didx = distract.row_idx.to_numpy()
if len(didx):
    lg, _ = canonical_forward(model, [base_ids_all[i] for i in didx], pad_id,
                              patch=delta[didx])
    js_pb = js_divergence(lg, logits_b[didx]).numpy()
    js_cb = js_divergence(logits_c[didx], logits_b[didx]).numpy()
    am = lg.argmax(-1).numpy()
    for k, i in enumerate(didx):
        r = sc.iloc[i]
        forms = form_ids.get(r.answer_old, {})
        records.append({
            "pair_id": r.pair_id, "condition": "distractor_real",
            "margin_base": np.nan, "margin_cf": np.nan, "margin_patch": np.nan,
            "margin_recovery": np.nan,
            "direction_correct": int(am[k]) in set(forms.values()),  # answer retained
            "js_base_cf": float(js_cb[k]),   # cf-vs-base reference (should be small)
            "js_patch_cf": float(js_pb[k]),  # patched-vs-base (should be small)
            "js_recovery": np.nan,
            "topk_base_cf": np.nan, "topk_patch_cf": np.nan,
            "argmax_patch": int(am[k]),
            "argmax_patch_decoded": tok.decode([int(am[k])]),
            "patch_norm": float(delta[i].norm()),
        })

pm = pd.DataFrame(records)
cell_meta = sc[["pair_id", "cell", "stratum", "query_order", "prompt_format",
                "preamble", "family_id", "answer_old", "answer_new", "delta_norm"]]
pm = pm.merge(cell_meta, on="pair_id")
pm.to_parquet(OUT / "patch_metrics.parquet", index=False)
print(f"\n{len(pm)} metric rows -> {OUT/'patch_metrics.parquet'}")

print("\n=== median margin recovery by condition (change cells) ===")
chg = pm[pm.cell.isin(["TARGET_EDIT", "REVERSE"])]
print(chg.groupby("condition")["margin_recovery"].median().sort_values(ascending=False).to_string())
print("\n=== direction-correct rate by condition ===")
print(chg.groupby("condition")["direction_correct"].mean().sort_values(ascending=False).to_string())
