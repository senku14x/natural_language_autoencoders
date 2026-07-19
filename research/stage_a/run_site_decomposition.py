"""Stage A site decomposition: where at L20 does the counterfactual live?

The final-position patch recovered ~1.4% of the margin (patch_metrics.parquet),
so the answer must flow through other positions' L20 states. Decompose by
patch scope on the eligible change rows:

  final_only          (already measured — joined from patch_metrics)
  edit_only           delta at the edit-token position only
  post_edit_excl_edit deltas at every position after the edit token, not the
                      edit token itself
  edit_and_final      edit-token delta + final-position delta
  all_positions       deltas at every position (== every position >= edit_pos,
                      pre-edit deltas are exactly zero by prompt identity —
                      asserted). Positive control: must recover ~1.0.

Edit-site controls: reverse (-delta), matched-norm random, unrelated
different-transition, same-transition other pair.

Canonical-shape forwards; patch = additive at block-20 output.
Output: site_decomposition.parquet.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage_a_lib import (CANON_B, CANON_L, FinalPosPatcher, js_divergence,
                         left_pad_batch, load_model_and_tokenizer, topk_overlap)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
MODEL_PATH = os.environ.get(
    "STAGE_A_MODEL_DIR",
    str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
)
SEED = 20260719
rng = np.random.default_rng(SEED + 1)
torch.manual_seed(SEED + 1)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
work = df[(df["split"] == "dev") | (df["cell"] == "NULL_AA")].reset_index(drop=True)
sc = pd.read_parquet(OUT / "screen.parquet").reset_index(drop=True)
sc["row_idx"] = sc.index
meta = json.load(open(OUT / "screen_meta.json"))
chat_form = meta["chat_canonical_form_by_stratum"]

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

change = sc[sc.cell.isin(["TARGET_EDIT", "REVERSE"]) & sc.eligible].copy()
change["tok_old"] = [canon_tok(r, "old") for r in change.itertuples(index=False)]
change["tok_new"] = [canon_tok(r, "new") for r in change.itertuples(index=False)]
change = change[change.tok_old.notna() & change.tok_new.notna()].reset_index(drop=True)
w = work.set_index("pair_id")
change["edit_pos"] = w.loc[change.pair_id, "edit_pos"].to_numpy()
change["utrans"] = [frozenset((a, b)) for a, b in
                    zip(w.loc[change.pair_id, "value_old"], w.loc[change.pair_id, "value_new"])]
change["otrans"] = change.answer_old + "->" + change.answer_new
change["cellkey"] = change.prompt_format + "/" + change.preamble.astype(str)
base_ids = [list(w.loc[p, "base_input_ids"]) for p in change.pair_id]
cf_ids = [list(w.loc[p, "cf_input_ids"]) for p in change.pair_id]
lens = np.array([len(x) for x in base_ids])
edit_pos = change.edit_pos.to_numpy()
print(f"eligible change rows: {len(change)}")

model, tok = load_model_and_tokenizer(MODEL_PATH)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


class SeqPatcher(FinalPosPatcher):
    """Adds a full [B, CANON_L, d] fp32 tensor to the block-20 output."""

    def __enter__(self):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.captured = h.detach().float().cpu().clone()
            if self.patch is None:
                return output
            h = (h.float() + self.patch.to(h.device)).to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h

        self._handle = self.block.register_forward_hook(hook)
        return self


@torch.no_grad()
def seq_forward(id_lists, patch_full=None):
    """Canonical-shape forward with optional [N, CANON_L, d] additive patch.
    Returns (final logits fp32 [N, V], full h20 fp32 [N, CANON_L, d])."""
    lg_out, h_out = [], []
    with SeqPatcher(model) as p:
        for lo in range(0, len(id_lists), CANON_B):
            chunk = list(id_lists[lo:lo + CANON_B])
            n_real = len(chunk)
            while len(chunk) < CANON_B:
                chunk.append(chunk[-1])
            cp = None
            if patch_full is not None:
                cp = patch_full[lo:lo + n_real]
                if n_real < CANON_B:
                    cp = torch.cat([cp, cp[-1:].expand(CANON_B - n_real, -1, -1)])
            ids, mask, pos = left_pad_batch(chunk, pad_id, pad_to=CANON_L)
            p.patch = cp
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                        logits_to_keep=1)
            lg_out.append(out.logits[:n_real, -1, :].float().cpu())
            h_out.append(p.captured[:n_real])
    return torch.cat(lg_out), torch.cat(h_out)


print("extracting full-sequence h20 (base, cf)...")
lg_base, h20_b = seq_forward(base_ids)
lg_cf, h20_c = seq_forward(cf_ids)
delta_full = h20_c - h20_b  # [N, CANON_L, d]; padded region cancels identically

# padded index helpers
pad_off = CANON_L - lens          # left-pad offset per row
p_edit = pad_off + edit_pos       # padded index of the edit token
p_final = np.full(len(change), CANON_L - 1)

# assert pre-edit deltas are exactly zero (prompt identity up to the edit)
pre_edit_max = 0.0
for i in range(len(change)):
    if p_edit[i] > pad_off[i]:
        pre_edit_max = max(pre_edit_max,
                           delta_full[i, pad_off[i]:p_edit[i]].abs().max().item())
assert pre_edit_max == 0.0, f"pre-edit deltas nonzero: {pre_edit_max}"
print("pre-edit deltas exactly zero ✓")

d_edit = delta_full[np.arange(len(change)), p_edit]      # [N, d]
d_edit_norm = d_edit.norm(dim=-1, keepdim=True)
print("edit-site delta norms:", d_edit_norm.squeeze().quantile(torch.tensor([.25, .5, .75])).tolist())


def scope_tensor(vec_at_edit=None, post_edit=False, final_vec=None):
    """Build [N, CANON_L, d] patch tensors row by row."""
    t = torch.zeros(len(change), CANON_L, delta_full.shape[-1])
    for i in range(len(change)):
        if vec_at_edit is not None:
            t[i, p_edit[i]] = vec_at_edit[i]
        if post_edit:
            t[i, p_edit[i] + 1:] = delta_full[i, p_edit[i] + 1:]
        if final_vec is not None:
            t[i, -1] = final_vec[i]
    return t


# edit-site control vectors
g = torch.from_numpy(rng.standard_normal(d_edit.shape)).float()
rand_edit = g / g.norm(dim=-1, keepdim=True) * d_edit_norm

unrel, same_t = [], []
by_cell = dict(tuple(change.groupby("cellkey")))
for r in change.itertuples(index=False):
    cell = by_cell[r.cellkey]
    cand = cell[(cell.family_id != r.family_id) & (cell.utrans != r.utrans)]
    unrel.append(int(cand.index[rng.integers(len(cand))]) if len(cand) else None)
    cand2 = cell[(cell.family_id != r.family_id) & (cell.otrans == r.otrans)]
    same_t.append(int(cand2.index[rng.integers(len(cand2))]) if len(cand2) else None)

CONDS = {
    "edit_only_real": scope_tensor(vec_at_edit=d_edit),
    "edit_only_reverse": scope_tensor(vec_at_edit=-d_edit),
    "edit_only_random": scope_tensor(vec_at_edit=rand_edit),
    "edit_only_unrelated": scope_tensor(
        vec_at_edit=torch.stack([d_edit[j] if j is not None else torch.full_like(d_edit[0], float("nan"))
                                 for j in unrel])),
    "edit_only_same_transition": scope_tensor(
        vec_at_edit=torch.stack([d_edit[j] if j is not None else torch.full_like(d_edit[0], float("nan"))
                                 for j in same_t])),
    "post_edit_excl_edit": scope_tensor(post_edit=True),
    "edit_and_final": scope_tensor(
        vec_at_edit=d_edit,
        final_vec=delta_full[np.arange(len(change)), p_final]),
    "all_positions": scope_tensor(vec_at_edit=d_edit, post_edit=True),
}

to = torch.from_numpy(change.tok_old.astype(int).to_numpy())
tn = torch.from_numpy(change.tok_new.astype(int).to_numpy())
ar = torch.arange(len(change))
m_base = (lg_base[ar, tn] - lg_base[ar, to]).numpy()
m_cf = (lg_cf[ar, tn] - lg_cf[ar, to]).numpy()
js_ref = js_divergence(lg_base, lg_cf)
tk_ref = topk_overlap(lg_base, lg_cf)

records = []
for cname, pt in CONDS.items():
    valid = ~torch.isnan(pt.sum((1, 2)))
    vi = valid.nonzero().squeeze(-1)
    print(f"condition {cname}: {len(vi)}/{len(change)} rows")
    if len(vi) == 0:
        continue
    lg, _ = seq_forward([base_ids[int(i)] for i in vi], patch_full=pt[vi])
    arv = torch.arange(len(vi))
    m_patch = (lg[arv, tn[vi]] - lg[arv, to[vi]]).numpy()
    js_p = js_divergence(lg, lg_cf[vi]).numpy()
    tk_p = topk_overlap(lg, lg_cf[vi]).numpy()
    am = lg.argmax(-1).numpy()
    for k, i in enumerate(vi.numpy()):
        r = change.iloc[i]
        denom = m_cf[i] - m_base[i]
        records.append({
            "pair_id": r.pair_id, "condition": cname,
            "margin_base": m_base[i], "margin_cf": m_cf[i], "margin_patch": m_patch[k],
            "margin_recovery": (m_patch[k] - m_base[i]) / denom if denom != 0 else np.nan,
            "direction_correct": bool(m_patch[k] > m_base[i]),
            "js_base_cf": float(js_ref[i]), "js_patch_cf": float(js_p[k]),
            "js_recovery": (float(1.0 - js_p[k] / float(js_ref[i]))
                            if float(js_ref[i]) > 1e-6 else np.nan),
            "topk_base_cf": float(tk_ref[i]), "topk_patch_cf": float(tk_p[k]),
            "argmax_patch_decoded": tok.decode([int(am[k])]),
            "edit_delta_norm": float(d_edit_norm[i]),
        })

sd = pd.DataFrame(records)
cm = change[["pair_id", "cell", "stratum", "query_order", "prompt_format",
             "preamble", "family_id", "answer_old", "answer_new", "delta_norm"]]
sd = sd.merge(cm, on="pair_id")
sd.to_parquet(OUT / "site_decomposition.parquet", index=False)
print(f"\n{len(sd)} rows -> {OUT/'site_decomposition.parquet'}")

print("\n=== median margin recovery by condition ===")
print(sd.groupby("condition")["margin_recovery"].median().sort_values(ascending=False).to_string())
print("\n=== direction-correct rate ===")
print(sd.groupby("condition")["direction_correct"].mean().sort_values(ascending=False).to_string())
print("\n=== argmax after patch == cf answer? (all_positions should be ~1) ===")
for c in ["all_positions", "edit_only_real", "post_edit_excl_edit", "edit_and_final"]:
    d = sd[sd.condition == c]
    if len(d):
        print(c, ":", (d.margin_recovery > 0.5).mean().round(3), "rows>0.5 |",
              "median", d.margin_recovery.median().round(3))
