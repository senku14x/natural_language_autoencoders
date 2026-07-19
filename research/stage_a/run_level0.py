"""Level-0 plumbing checks (v1 doc §5 Level 0, Experiment-1 sanity checks).

Checks:
  L0.1 frozen input_ids: re-encode raw rows with the live pinned tokenizer;
       verify final_pos == len(input_ids) - 1 for every dev row (both formats)
  L0.2 hook parity: layers[20] forward-hook output == hidden_states[21]
  L0.3 alpha-zero identity: zero patch reproduces unpatched logits
  L0.4 batch-size-1 vs batched (left-padded) parity
  L0.5 extraction noise floor: repeated forwards under different batch
       compositions; NULL_AA prompt-identity deltas -> epsilon guard

Writes research/data/artifacts/v1/stage_a/level0.json. Read-only w.r.t. the
frozen dataset.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage_a_lib import (LAYER, FinalPosPatcher, forward_final,
                         load_model_and_tokenizer, left_pad_batch, batched)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/stage_a"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get(
    "STAGE_A_MODEL_DIR",
    str(Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
)

torch.manual_seed(20260719)
report = {"model_path": MODEL_PATH, "layer": LAYER, "seed": 20260719}

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
dev = df[df["split"] == "dev"].reset_index(drop=True)
null_aa = df[df["cell"] == "NULL_AA"].reset_index(drop=True)
print(f"dev rows: {len(dev)}, NULL_AA rows: {len(null_aa)}")

model, tok = load_model_and_tokenizer(MODEL_PATH)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# ── L0.1 tokenization / position convention ─────────────────────────────────
raw_dev = dev[dev["prompt_format"] == "raw"]
n_check = 0
for row in raw_dev.head(500).itertuples(index=False):
    enc = tok.encode(row.base_prompt, add_special_tokens=False)
    assert enc == list(row.base_input_ids), f"re-encode mismatch: {row.pair_id}"
    n_check += 1
bad_final = int((dev["base_input_ids"].apply(len) - 1 != dev["final_pos"]).sum())
bad_final += int((dev["cf_input_ids"].apply(len) - 1 != dev["final_pos"]).sum())
report["l0_1"] = {"raw_reencode_checked": n_check, "raw_reencode_fail": 0,
                  "final_pos_not_last": bad_final}
assert bad_final == 0, "final_pos is not the last input_ids index somewhere"
print(f"L0.1 ok: {n_check} raw rows re-encode exactly; final_pos == last index for all dev rows")

# ── L0.2 hook parity ────────────────────────────────────────────────────────
sample = dev.sample(16, random_state=0)
ids = [list(x) for x in sample["base_input_ids"]]
with torch.no_grad(), FinalPosPatcher(model) as p:
    ids_t, mask, pos = left_pad_batch(ids, pad_id)
    out = model(input_ids=ids_t, attention_mask=mask, position_ids=pos,
                output_hidden_states=True)
    hs21 = out.hidden_states[LAYER + 1][:, -1, :].float()
    diff = (p.captured - hs21).abs().max().item()
report["l0_2"] = {"hook_vs_hidden_states21_max_abs_diff": diff}
assert diff == 0.0, f"hook != hidden_states[21] (max abs diff {diff})"
print(f"L0.2 ok: hook output == hidden_states[{LAYER+1}] exactly (max abs diff {diff})")

# ── L0.3 alpha-zero identity ────────────────────────────────────────────────
logits_plain, h_plain = forward_final(model, ids, pad_id, patch=None)
zero = torch.zeros(len(ids), model.config.hidden_size)
logits_zero, _ = forward_final(model, ids, pad_id, patch=zero)
dz = (logits_plain - logits_zero).abs().max().item()
report["l0_3"] = {"zero_patch_max_abs_logit_diff": dz}
assert dz == 0.0, f"zero patch changed logits (max abs diff {dz})"
print(f"L0.3 ok: zero patch is exact identity (max abs logit diff {dz})")

# ── L0.4 batch-size-1 vs batched parity ─────────────────────────────────────
per_row_logits, per_row_h = [], []
for i in range(len(ids)):
    lg, hh = forward_final(model, [ids[i]], pad_id)
    per_row_logits.append(lg[0])
    per_row_h.append(hh[0])
per_row_logits = torch.stack(per_row_logits)
per_row_h = torch.stack(per_row_h)
logit_dev = (per_row_logits - logits_plain).abs().max().item()
h_dev = (per_row_h - h_plain).norm(dim=-1)
h_norm = h_plain.norm(dim=-1)
report["l0_4"] = {
    "batched_vs_single_max_abs_logit_diff": logit_dev,
    "batched_vs_single_h20_reldiff_max": (h_dev / h_norm).max().item(),
    "note": "bf16 + padded batching is not expected to be bitwise; bounded below by noise floor",
}
print(f"L0.4: batched-vs-single max |Δlogit| {logit_dev:.4f}, "
      f"max rel ‖Δh20‖ {(h_dev/h_norm).max().item():.2e}")

# ── L0.6 canonical-shape policy validation ──────────────────────────────────
# All Stage A measurement forwards use shape (CANON_B, CANON_L). Validate:
# (a) repeat determinism, (b) invariance to batch composition within the
# canonical shape, (c) NULL_AA identical prompts -> exactly zero delta.
from stage_a_lib import canonical_forward, CANON_B, CANON_L

ids96 = [list(x) for x in dev.sample(96, random_state=1)["base_input_ids"]]
lg_a, h_a = canonical_forward(model, ids96, pad_id)
lg_b, h_b = canonical_forward(model, ids96, pad_id)
rep = (lg_a - lg_b).abs().max().item() + (h_a - h_b).norm(dim=-1).max().item()

perm2 = torch.randperm(64).tolist()  # recompose first canonical chunk
ids_c = [ids96[i] for i in perm2] + ids96[64:]
lg_c, h_c = canonical_forward(model, ids_c, pad_id)
inv = max((lg_c[j] - lg_a[i]).abs().max().item() for j, i in enumerate(perm2))

na_b = [list(x) for x in null_aa["base_input_ids"]]
na_c = [list(x) for x in null_aa["cf_input_ids"]]
_, h_nb = canonical_forward(model, na_b, pad_id)
_, h_nc = canonical_forward(model, na_c, pad_id)
null_canon = (h_nb - h_nc).norm(dim=-1).max().item()

report["l0_6"] = {
    "canonical_shape": [CANON_B, CANON_L],
    "repeat_max_diff": rep,
    "recomposition_max_logit_diff": inv,
    "null_aa_delta_norm_max_canonical": null_canon,
}
assert rep == 0.0 and inv == 0.0 and null_canon == 0.0, report["l0_6"]
print(f"L0.6 ok: canonical shape ({CANON_B},{CANON_L}) is bitwise deterministic; "
      f"NULL_AA deltas exactly zero ({len(na_b)} rows)")

# ── L0.5 noise floor + epsilon ──────────────────────────────────────────────
# (a) same prompts, two different batch compositions (shuffled order/padding)
perm = torch.randperm(len(ids)).tolist()
ids_shuf = [ids[i] for i in perm]
_, h_shuf = forward_final(model, ids_shuf, pad_id)
h_shuf_unperm = torch.empty_like(h_shuf)
for j, i in enumerate(perm):
    h_shuf_unperm[i] = h_shuf[j]
noise_batch = (h_shuf_unperm - h_plain).norm(dim=-1)

# (b) NULL_AA rows: base and cf prompts are identical strings -> delta is
# exactly zero when extracted in the SAME forward mode; across separate
# batched forwards, the residual is the measurement floor.
na_ids_b = [list(x) for x in null_aa["base_input_ids"].head(64)]
na_ids_c = [list(x) for x in null_aa["cf_input_ids"].head(64)]
assert all(a == b for a, b in zip(na_ids_b, na_ids_c)), "NULL_AA prompts differ!"
_, h_na_1 = forward_final(model, na_ids_b, pad_id)
_, h_na_2 = forward_final(model, na_ids_c, pad_id)  # identical ids, separate call
null_delta = (h_na_1 - h_na_2).norm(dim=-1)

# Same-shape forwards are bitwise deterministic (L0.6), so the same-shape
# floor is 0. The operative floor is CROSS-shape variation (L0.4 and the
# probe session): ~2.2 h20-norm units. Epsilon guards Stage B if shapes are
# ever mixed; within the canonical policy NULL_AA deltas are exactly zero.
cross_shape_h = float((h_dev).max().item())
h20_typ = float(h_plain.norm(dim=-1).median().item())
eps = round(1.5 * max(cross_shape_h, 2.2), 2)
report["l0_5"] = {
    "same_shape_recomposition_delta_norm_max": float(noise_batch.max().item()),
    "null_aa_identical_prompt_delta_norm_max": float(null_delta.max().item()),
    "cross_shape_delta_norm_max_observed": cross_shape_h,
    "typical_h20_final_norm_median": h20_typ,
    "epsilon_guard_provisional": eps,
    "rule": "treat ||delta|| < epsilon as exact zero before any AV-side rescale; "
            "within the canonical-shape policy true-null deltas are exactly 0",
}
print(f"L0.5: same-shape floor 0; cross-shape max ‖Δh‖ {cross_shape_h:.2f} "
      f"(typical ‖h20‖ {h20_typ:.1f}); provisional epsilon = {eps}")

with open(OUT / "level0.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nLevel-0 report -> {OUT/'level0.json'}")
