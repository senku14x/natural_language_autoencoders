"""Extract edit-position L20 states (base, cf) for ALL eligible dev change +
distractor rows across every cell. Feeds the position-sweep / discriminator
probes. Canonical-shape forwards; asserts pre-edit deltas exactly zero.
Output: research/data/artifacts/v1/position_sweep/{h_edit_base,h_edit_cf}.npy,
rows.parquet.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/stage_a"))
from stage_a_lib import load_model_and_tokenizer
from run_site_decomposition_lib import seq_forward_factory

OUT = REPO / "research/data/artifacts/v1/position_sweep"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
            "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
sc = pd.read_parquet(REPO / "research/data/artifacts/v1/stage_a/screen.parquet")
w = df.set_index("pair_id")

rows = sc[(sc.split == "dev") & sc.eligible
          & sc.cell.isin(["TARGET_EDIT", "REVERSE", "DISTRACTOR_EDIT"])].copy()
for c in ["edit_pos", "value_old", "value_new", "family_id", "distractor_flavor"]:
    rows[c] = w.loc[rows.pair_id, c].to_numpy()
rows = rows.reset_index(drop=True)
print(f"rows: {len(rows)}  |  cells {rows.cell.value_counts().to_dict()}")
print(f"edit_pos by cell:\n{rows.groupby(['prompt_format','preamble']).edit_pos.agg(['min','max','count'])}")

model, tok = load_model_and_tokenizer(MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
seq_forward = seq_forward_factory(model, pad_id)

base_ids = [list(w.loc[p, "base_input_ids"]) for p in rows.pair_id]
cf_ids = [list(w.loc[p, "cf_input_ids"]) for p in rows.pair_id]
print("extracting base...")
_, hb = seq_forward(base_ids)
print("extracting cf...")
_, hc = seq_forward(cf_ids)
CANON_L = hb.shape[1]
lens = np.array([len(x) for x in base_ids])
p_edit = (CANON_L - lens) + rows.edit_pos.to_numpy()
ar = np.arange(len(rows))
heb = hb[ar, p_edit].clone().numpy()
hec = hc[ar, p_edit].clone().numpy()
# pre-edit deltas exactly zero
dfull = hc - hb
pre = max((dfull[i, CANON_L - lens[i]:p_edit[i]].abs().max().item()
           if p_edit[i] > CANON_L - lens[i] else 0.0) for i in range(len(rows)))
assert pre == 0.0, f"pre-edit nonzero {pre}"
np.save(OUT / "h_edit_base.npy", heb)
np.save(OUT / "h_edit_cf.npy", hec)
rows[["pair_id", "semantic_id", "family_id", "cell", "stratum", "query_order",
      "prompt_format", "preamble", "distractor_flavor", "value_old", "value_new",
      "edit_pos"]].to_parquet(OUT / "rows.parquet")
print(f"saved {len(rows)} edit-position states -> {OUT}")
