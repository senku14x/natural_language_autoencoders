"""Stage A behavioral screen + L20 final-position extraction on dev (+ NULL_AA).

Per row, both prompts, under the canonical-shape forward policy:
  - full-vocab argmax at the final position (decoded and stored)
  - logits for all four audited surface forms of answer_old / answer_new /
    answer_distractor
  - eligibility: full-vocab argmax is one of the four forms of the correct
    value (base -> answer_old, cf -> answer_new; equal for DISTRACTOR/NULL)
  - margin convention: raw format uses the frozen leading_space ids
    (answer_token_id_* columns); chat format uses the dominant audited form
    measured over dev base rows (frozen in the output json, reported per kind)
  - h20 final-position vectors (fp32) for base and cf; delta = h_cf - h_base

Outputs under research/data/artifacts/v1/stage_a/:
  screen.parquet, h_base.npy, h_cf.npy, screen_meta.json
Frozen pairs.parquet is not modified.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage_a_lib import canonical_forward, load_model_and_tokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get(
    "STAGE_A_MODEL_DIR",
    str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
)
torch.manual_seed(20260719)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
work = df[(df["split"] == "dev") | (df["cell"] == "NULL_AA")].reset_index(drop=True)
print(f"rows: {len(work)} (dev {int((work['split']=='dev').sum())} "
      f"+ non-dev NULL_AA {int(((work['cell']=='NULL_AA') & (work['split']!='dev')).sum())})")

audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
FORMS = ["leading_space", "bare", "leading_space_cap", "bare_cap"]
form_ids = defaultdict(dict)  # value -> form -> token_id (kept single-token only)
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids[x["value"]][x["form"]] = x["token_ids"][0]

model, tok = load_model_and_tokenizer(MODEL_PATH)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# ── forwards ────────────────────────────────────────────────────────────────
base_ids = [list(x) for x in work["base_input_ids"]]
cf_ids = [list(x) for x in work["cf_input_ids"]]
print("running base forwards...")
logits_b, h_b = canonical_forward(model, base_ids, pad_id)
print("running cf forwards...")
logits_c, h_c = canonical_forward(model, cf_ids, pad_id)

delta = h_c - h_b
np.save(OUT / "h_base.npy", h_b.numpy())
np.save(OUT / "h_cf.npy", h_c.numpy())
np.save(OUT / "logits_base_fp16.npy", logits_b.half().numpy())
np.save(OUT / "logits_cf_fp16.npy", logits_c.half().numpy())

# ── per-row measurements ────────────────────────────────────────────────────
argmax_b = logits_b.argmax(-1)
argmax_c = logits_c.argmax(-1)

def form_token_ids(value):
    return {f: form_ids.get(value, {}).get(f) for f in FORMS}

records = []
for i, row in enumerate(work.itertuples(index=False)):
    fo, fn = form_token_ids(row.answer_old), form_token_ids(row.answer_new)
    ok_ids = {t for t in fo.values() if t is not None}
    nw_ids = {t for t in fn.values() if t is not None}
    rec = {
        "pair_id": row.pair_id, "semantic_id": row.semantic_id,
        "family_id": row.family_id, "split": row.split,
        "argmax_base": int(argmax_b[i]), "argmax_cf": int(argmax_c[i]),
        "argmax_base_decoded": tok.decode([int(argmax_b[i])]),
        "argmax_cf_decoded": tok.decode([int(argmax_c[i])]),
        "base_correct": int(argmax_b[i]) in ok_ids,
        "cf_correct": int(argmax_c[i]) in (nw_ids if row.cell in ("TARGET_EDIT", "REVERSE") else ok_ids),
        "delta_norm": float(delta[i].norm()),
        "h_base_norm": float(h_b[i].norm()), "h_cf_norm": float(h_c[i].norm()),
    }
    for side, lg in (("base", logits_b[i]), ("cf", logits_c[i])):
        for tag, fids in (("old", fo), ("new", fn)):
            for f in FORMS:
                t = fids[f]
                rec[f"logit_{side}_{tag}_{f}"] = float(lg[t]) if t is not None else np.nan
    records.append(rec)

sc = pd.DataFrame(records)
meta_cols = ["cell", "stratum", "distractor_flavor", "query_order",
             "prompt_format", "preamble", "answer_old", "answer_new",
             "entity_target", "value_old", "value_new", "final_pos"]
sc = sc.merge(work[["pair_id"] + meta_cols], on="pair_id")

# ── chat canonical form, per stratum (frozen from dev base rows) ────────────
# Colors (S) and names (N) capitalize differently after the chat template;
# freeze the dominant argmax form per stratum from correct dev base rows.
form_of_tok = {}
for v, fs in form_ids.items():
    for f, t in fs.items():
        form_of_tok[(v, t)] = f
chat_form, chat_form_counts = {}, {}
for stratum, g in sc[(sc.prompt_format == "chat") & sc.base_correct].groupby("stratum"):
    c = Counter(form_of_tok.get((r.answer_old, r.argmax_base))
                for r in g.itertuples(index=False))
    chat_form[stratum] = c.most_common(1)[0][0] if c else "leading_space"
    chat_form_counts[stratum] = dict(c)
print("chat argmax form distribution by stratum:", chat_form_counts)
print("frozen chat canonical form by stratum:", chat_form)

def margin(rec_row, side):
    """logit(new) - logit(old) with the per-format canonical form."""
    f = ("leading_space" if rec_row.prompt_format == "raw"
         else chat_form.get(rec_row.stratum, "leading_space"))
    return (getattr(rec_row, f"logit_{side}_new_{f}")
            - getattr(rec_row, f"logit_{side}_old_{f}"))

sc["margin_base"] = [margin(r, "base") for r in sc.itertuples(index=False)]
sc["margin_cf"] = [margin(r, "cf") for r in sc.itertuples(index=False)]

is_change = sc.cell.isin(["TARGET_EDIT", "REVERSE"])
sc["eligible"] = np.where(
    is_change,
    sc.base_correct & sc.cf_correct & (sc.margin_cf > sc.margin_base),
    sc.base_correct & sc.cf_correct,
)
sc.to_parquet(OUT / "screen.parquet", index=False)

# ── summary ─────────────────────────────────────────────────────────────────
print("\n=== eligibility by cell ===")
print(sc.groupby("cell")["eligible"].agg(["mean", "sum", "count"]).to_string())
print("\n=== eligibility by format x preamble (change cells) ===")
ch = sc[is_change]
print(ch.groupby(["prompt_format", "preamble"])["eligible"].agg(["mean", "sum", "count"]).to_string())
print("\n=== delta norms by cell ===")
print(sc.groupby("cell")["delta_norm"].describe()[["min", "25%", "50%", "75%", "max"]].to_string())

meta = {
    "model_path": MODEL_PATH, "seed": 20260719,
    "canonical_shape": [64, 131],
    "chat_canonical_form_by_stratum": chat_form,
    "chat_form_counts_by_stratum": chat_form_counts,
    "n_rows": len(sc),
    "eligible_change_rows": int(sc[is_change]["eligible"].sum()),
    "eligible_distractor_rows": int(sc[sc.cell == "DISTRACTOR_EDIT"]["eligible"].sum()),
}
with open(OUT / "screen_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nwritten: screen.parquet, h_base.npy, h_cf.npy, screen_meta.json -> {OUT}")
