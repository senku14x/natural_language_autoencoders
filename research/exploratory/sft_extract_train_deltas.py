"""Work item 2, step 1: behavioral screen + edit-site extraction for TRAIN
raw/preamble change rows (never screened -- Stage A covered dev only), plus
dev raw/preamble change-row re-extraction on this instance for train/eval
consistency.

Eligibility rule = run_screen.py's: full-vocab argmax is an audited surface
form of the correct answer on both prompts, AND margin_cf > margin_base
(raw format -> frozen leading_space ids). Canonical-shape forwards.

Outputs -> research/data/artifacts/v1/sft_transition/:
  train_rows.parquet, train_h_edit_{base,cf}.npy   (all 6,000 rows + eligible flag)
  dev_rows.parquet,   dev_h_edit_{base,cf}.npy     (all 800 rows; frozen eligibility
                                                    + this-instance agreement col)
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/stage_a"))
from stage_a_lib import CANON_B, CANON_L, left_pad_batch, load_model_and_tokenizer  # noqa: E402
from run_site_decomposition_lib import SeqPatcher  # noqa: E402

OUT = REPO / "research/data/artifacts/v1/sft_transition"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
            "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
torch.manual_seed(20260719)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
form_ids = defaultdict(set)
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids[x["value"]].add(x["token_ids"][0])

model, tok = load_model_and_tokenizer(MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


@torch.no_grad()
def screen_and_extract(rows):
    """Per row: base+cf canonical forwards; capture final logits (screen) and
    edit-position h20 for both. Streams chunk by chunk (no full-seq retention)."""
    n = len(rows)
    heb = np.empty((n, model.config.hidden_size), dtype=np.float32)
    hec = np.empty((n, model.config.hidden_size), dtype=np.float32)
    recs = []
    base_ids = [list(x) for x in rows.base_input_ids]
    cf_ids = [list(x) for x in rows.cf_input_ids]
    edit_pos = rows.edit_pos.to_numpy()
    with SeqPatcher(model) as p:
        for lo in range(0, n, CANON_B):
            hi = min(lo + CANON_B, n)
            out = {}
            for tag, idlists in (("base", base_ids[lo:hi]), ("cf", cf_ids[lo:hi])):
                chunk = list(idlists)
                n_real = len(chunk)
                while len(chunk) < CANON_B:
                    chunk.append(chunk[-1])
                ids, mask, pos = left_pad_batch(chunk, pad_id, pad_to=CANON_L)
                p.patch = None
                o = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                          logits_to_keep=1)
                out[tag] = (o.logits[:n_real, -1, :].float().cpu(),
                            p.captured[:n_real].clone())
            lens = np.array([len(x) for x in base_ids[lo:hi]])
            pedit = (CANON_L - lens) + edit_pos[lo:hi]
            ar = np.arange(hi - lo)
            hb_full, hc_full = out["base"][1], out["cf"][1]
            # pre-edit deltas exactly zero (canonical-shape determinism check)
            dfull = hc_full - hb_full
            for i in range(hi - lo):
                s0 = CANON_L - lens[i]
                if pedit[i] > s0:
                    m = dfull[i, s0:pedit[i]].abs().max().item()
                    assert m == 0.0, f"pre-edit delta nonzero {m} row {lo+i}"
            heb[lo:hi] = hb_full[ar, pedit].numpy()
            hec[lo:hi] = hc_full[ar, pedit].numpy()
            for i, (_, row) in enumerate(rows.iloc[lo:hi].iterrows()):
                lg_b, lg_c = out["base"][0][i], out["cf"][0][i]
                ok = form_ids[row.answer_old]
                nw = form_ids[row.answer_new]
                am_b, am_c = int(lg_b.argmax()), int(lg_c.argmax())
                mb = float(lg_b[row.answer_token_id_new] - lg_b[row.answer_token_id_old])
                mc = float(lg_c[row.answer_token_id_new] - lg_c[row.answer_token_id_old])
                recs.append({
                    "pair_id": row.pair_id, "base_correct": am_b in ok,
                    "cf_correct": am_c in nw, "margin_base": mb, "margin_cf": mc,
                    "eligible_here": (am_b in ok) and (am_c in nw) and (mc > mb),
                })
            if (lo // CANON_B) % 10 == 0:
                print(f"  {hi}/{n}", flush=True)
    return pd.DataFrame(recs), heb, hec


META = ["pair_id", "semantic_id", "family_id", "stratum", "cell", "query_order",
        "edited_slot", "value_old", "value_new", "edit_pos", "final_pos",
        "caption_arrow_transition"]

# ── train ───────────────────────────────────────────────────────────────────
train = df[(df.split == "train") & (df.prompt_format == "raw") & df.preamble
           & df.cell.isin(["TARGET_EDIT", "REVERSE"])].reset_index(drop=True)
print(f"train raw/pre change rows: {len(train)}")
sc_t, heb, hec = screen_and_extract(train)
tr = train[META + ["base_input_ids", "cf_input_ids"]].copy()
tr = tr.drop(columns=["base_input_ids", "cf_input_ids"]).merge(sc_t, on="pair_id")
tr["eligible"] = tr.eligible_here
np.save(OUT / "train_h_edit_base.npy", heb)
np.save(OUT / "train_h_edit_cf.npy", hec)
tr.to_parquet(OUT / "train_rows.parquet")
print(f"train eligibility: {tr.eligible.mean():.3f} ({tr.eligible.sum()}/{len(tr)})")
print(tr.groupby('stratum').eligible.agg(['mean', 'sum', 'count']).to_string())

# ── dev (re-extraction + agreement check; frozen labels govern selection) ───
dev = df[(df.split == "dev") & (df.prompt_format == "raw") & df.preamble
         & df.cell.isin(["TARGET_EDIT", "REVERSE"])].reset_index(drop=True)
print(f"\ndev raw/pre change rows: {len(dev)}")
sc_d, heb_d, hec_d = screen_and_extract(dev)
dv = dev[META].merge(sc_d, on="pair_id")
frozen = pd.read_parquet(REPO / "research/data/artifacts/v1/stage_a/screen.parquet")
dv = dv.merge(frozen[["pair_id", "eligible"]], on="pair_id")  # frozen labels
np.save(OUT / "dev_h_edit_base.npy", heb_d)
np.save(OUT / "dev_h_edit_cf.npy", hec_d)
dv.to_parquet(OUT / "dev_rows.parquet")
agree = (dv.eligible == dv.eligible_here).mean()
print(f"dev eligibility agreement (this instance vs frozen): {agree:.4f} "
      f"({(dv.eligible != dv.eligible_here).sum()} flips / {len(dv)})")
print(f"dev eligible (frozen): {dv.eligible.sum()}")

d_norm = np.linalg.norm(hec - heb, axis=1)
print(f"\ntrain edit-delta norms: q25/50/75 = "
      f"{np.percentile(d_norm[tr.eligible], [25, 50, 75]).round(1)}")
json.dump({"train_rows": len(tr), "train_eligible": int(tr.eligible.sum()),
           "dev_rows": len(dv), "dev_eligible_frozen": int(dv.eligible.sum()),
           "dev_agreement": float(agree), "canonical_shape": [CANON_B, CANON_L],
           "torch": torch.__version__},
          open(OUT / "extract_meta.json", "w"), indent=1)
print("DONE")
