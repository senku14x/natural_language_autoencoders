"""Exploratory: trivial-decoder (logit-lens) baseline on edit-site deltas
(work item 1b).

The zero-shot report said "logit-lens-like" four times without measuring it.
This measures it: top-1/top-5 of W_U @ delta and W_U @ RMSNorm_final(delta)
for the SAME 50 sampled pairs, scored with the token-level translation of the
same string rule (hit = any top-k id is an audited single-token surface form
of the value). Arms: +d->new (primary), -d->old, cross terms, shuffled-donor
floor (exact donors from scored.parquet), and h_cf/h_base state positive
controls for instrument sensitivity.

Predictions + decision rule registered first in
temporary_artifacts/2026-07-19_negdelta_logitlens_predictions.md.
No model forward passes -- only the unembedding matrix and final-norm weights
are loaded from the base-model safetensors.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/data/artifacts/v1/zeroshot_av"
HF = os.environ["HF_HOME"]
BASE = Path(HF) / ("hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
                   "a09a35458c702b33eeacc393d103063234e8bc28")
SEED = 20260719
rng = np.random.default_rng(SEED)

# ── weights: W_U + final RMSNorm gamma ──────────────────────────────────────
cfg = json.load(open(BASE / "config.json"))
assert not cfg.get("tie_word_embeddings", False), "expected untied lm_head"
RMS_EPS = float(cfg["rms_norm_eps"])
index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
tensors = {}
for name in ["lm_head.weight", "model.norm.weight"]:
    with safe_open(BASE / index[name], framework="pt") as f:
        tensors[name] = f.get_tensor(name).float()
W_U = tensors["lm_head.weight"]          # [V, d]
gamma = tensors["model.norm.weight"]     # [d]
print(f"W_U {tuple(W_U.shape)}, gamma {tuple(gamma.shape)}, rms_eps {RMS_EPS}")

# ── data: saved states, exact prior sample, exact prior shuffled donors ─────
pool = pd.read_parquet(OUT / "pool.parquet").reset_index(drop=True)
h_b = torch.from_numpy(np.load(OUT / "h_edit_base.npy")).float()
h_c = torch.from_numpy(np.load(OUT / "h_edit_cf.npy")).float()
d_edit = h_c - h_b
man = json.load(open(OUT / "sample_manifest.json"))
sample = pool[pool.pair_id.isin(man["pair_ids"])].sort_values("pair_id").reset_index(drop=True)
assert len(sample) == 50
sidx = {p: i for i, p in enumerate(pool.pair_id)}
spos = sample.pair_id.map(sidx).to_numpy()

sc = pd.read_parquet(OUT / "scored.parquet")
donor_of = (sc[sc.arm == "shuffled"].groupby("pair_id").shuffled_from.first())
assert set(sample.pair_id) == set(donor_of.index)
donor_pos = np.array([sidx[donor_of[p]] for p in sample.pair_id])
print(f"sample 50 pairs; donors resolved for all (all in pool: "
      f"{all(donor_of[p] in sidx for p in sample.pair_id)})")

# ── audited surface-form token ids per value ────────────────────────────────
audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
form_ids = defaultdict(set)
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids[x["value"]].add(x["token_ids"][0])
missing = [v for v in set(sample.value_old) | set(sample.value_new) if not form_ids[v]]
assert not missing, f"values with no audited single-token form: {missing}"

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(REPO / "research/data/tokenizer/qwen25"))


def readout(vecs, convention):
    if convention == "rmsnorm":
        v = vecs * torch.rsqrt(vecs.pow(2).mean(-1, keepdim=True) + RMS_EPS) * gamma
    else:
        v = vecs
    return v @ W_U.T  # [n, V]


def hit_and_rank(logits, values):
    """Per row: rank of the best surface form of `values[i]` (0 = top-1)."""
    order = logits.argsort(dim=-1, descending=True)
    ranks = torch.empty(len(values), dtype=torch.long)
    inv = torch.empty(logits.shape[-1], dtype=torch.long)
    for i, val in enumerate(values):
        inv[order[i]] = torch.arange(logits.shape[-1])
        ranks[i] = min(int(inv[t]) for t in form_ids[val])
    return ranks


vec_arms = {
    "pos_delta": d_edit[spos],
    "neg_delta": -d_edit[spos],
    "shuffled_donor": d_edit[donor_pos],
    "h_cf": h_c[spos],
    "h_base": h_b[spos],
}
score_targets = {
    ("pos_delta", "new"), ("pos_delta", "old"),
    ("neg_delta", "old"), ("neg_delta", "new"),
    ("shuffled_donor", "new"),          # own pair's new value from donor delta
    ("h_cf", "new"), ("h_base", "old"),
}
vals = {"new": sample.value_new.tolist(), "old": sample.value_old.tolist()}


def boot_ci(hits, nboot=2000):
    b = [np.mean(rng.choice(hits, len(hits), replace=True)) for _ in range(nboot)]
    return np.percentile(b, [2.5, 97.5])


rows = []
top5_dump = {}
for conv in ["raw", "rmsnorm"]:
    for arm, vecs in vec_arms.items():
        logits = readout(vecs, conv)
        top5 = logits.topk(5, dim=-1).indices
        top5_dump[f"{conv}/{arm}"] = [
            [tok.decode([int(t)]) for t in top5[i]] for i in range(len(sample))]
        for tgt in ["new", "old"]:
            if (arm, tgt) not in score_targets:
                continue
            ranks = hit_and_rank(logits, vals[tgt])
            for k in (1, 5):
                hits = (ranks < k).numpy().astype(float)
                lo, hi = boot_ci(hits)
                rows.append({
                    "convention": conv, "arm": arm, "target": tgt, "topk": k,
                    "hit_rate": round(float(hits.mean()), 3),
                    "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                    "hit_rate_S": round(float(hits[(sample.stratum == "S").to_numpy()].mean()), 3),
                    "hit_rate_N": round(float(hits[(sample.stratum == "N").to_numpy()].mean()), 3),
                    "median_rank": int(np.median(ranks.numpy())),
                })

res = pd.DataFrame(rows)
res.to_parquet(OUT / "trivial_decoder.parquet", index=False)
json.dump(top5_dump, open(OUT / "trivial_decoder_top5.json", "w"), indent=1)
pd.set_option("display.width", 200)
print("\n=== trivial-decoder hit rates (50 pairs, bootstrap 95% CI) ===")
print(res.sort_values(["target", "arm", "convention", "topk"]).to_string(index=False))
print("\nDONE")
