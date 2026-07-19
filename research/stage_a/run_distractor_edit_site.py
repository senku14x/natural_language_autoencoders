"""Edit-site distractor arm: patching the (large) edit-site delta of a
DISTRACTOR edit must leave the queried answer unchanged. Grounds the
"answer unaffected" clause (Amendment 1 §3) at the certified site.

Output: distractor_edit_site.parquet
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage_a_lib import CANON_L, js_divergence, load_model_and_tokenizer
from run_site_decomposition_lib import seq_forward_factory

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
MODEL_PATH = os.environ.get(
    "STAGE_A_MODEL_DIR",
    str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
work = df[(df["split"] == "dev") | (df["cell"] == "NULL_AA")].reset_index(drop=True)
sc = pd.read_parquet(OUT / "screen.parquet").reset_index(drop=True)

audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
form_ids = {}
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids.setdefault(x["value"], {})[x["form"]] = x["token_ids"][0]

dis = sc[(sc.cell == "DISTRACTOR_EDIT") & sc.eligible].copy().reset_index(drop=True)
w = work.set_index("pair_id")
dis["edit_pos"] = w.loc[dis.pair_id, "edit_pos"].to_numpy()
base_ids = [list(w.loc[p, "base_input_ids"]) for p in dis.pair_id]
cf_ids = [list(w.loc[p, "cf_input_ids"]) for p in dis.pair_id]
lens = np.array([len(x) for x in base_ids])
print(f"eligible distractor rows: {len(dis)}")

model, tok = load_model_and_tokenizer(MODEL_PATH)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
seq_forward = seq_forward_factory(model, pad_id)

lg_base, h20_b = seq_forward(base_ids)
lg_cf, h20_c = seq_forward(cf_ids)
delta_full = h20_c - h20_b
pad_off = CANON_L - lens
p_edit = pad_off + dis.edit_pos.to_numpy()
d_edit = delta_full[np.arange(len(dis)), p_edit]

patch = torch.zeros_like(delta_full)
patch[np.arange(len(dis)), p_edit] = d_edit
lg_p, _ = seq_forward(base_ids, patch_full=patch)

records = []
js_pb = js_divergence(lg_p, lg_base).numpy()
js_cb = js_divergence(lg_cf, lg_base).numpy()
am = lg_p.argmax(-1).numpy()
for i in range(len(dis)):
    r = dis.iloc[i]
    forms = set(form_ids.get(r.answer_old, {}).values())
    records.append({
        "pair_id": r.pair_id, "condition": "distractor_edit_site",
        "answer_retained": int(am[i]) in forms,
        "argmax_patch_decoded": tok.decode([int(am[i])]),
        "js_patch_base": float(js_pb[i]), "js_cf_base": float(js_cb[i]),
        "edit_delta_norm": float(d_edit[i].norm()),
        "cell": r.cell, "stratum": r.stratum, "query_order": r.query_order,
        "prompt_format": r.prompt_format, "preamble": r.preamble,
        "family_id": r.family_id, "distractor_flavor": r.distractor_flavor,
    })
out = pd.DataFrame(records)
out.to_parquet(OUT / "distractor_edit_site.parquet", index=False)
print("answer retained:", out.answer_retained.mean().round(4))
print("JS(patch,base) median:", out.js_patch_base.median().round(4),
      "| JS(cf,base) median:", out.js_cf_base.median().round(4))
print("edit-site delta norm median:", out.edit_delta_norm.median().round(1))
print("by flavor:")
print(out.groupby("distractor_flavor")[["answer_retained", "js_patch_base"]]
      .agg({"answer_retained": "mean", "js_patch_base": "median"}).to_string())
