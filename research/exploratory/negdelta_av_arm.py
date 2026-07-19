"""Exploratory: -delta arm through the zero-shot AV (work item 1a).

Adds the single missing arm to the 2026-07-19 zero-shot experiment: inject
NEGATED edit-site deltas (-delta) for the SAME 50 sampled pairs and score with
the SAME string rule. Predictions registered first in
temporary_artifacts/2026-07-19_negdelta_logitlens_predictions.md.

Reuses the saved edit-site states (h_edit_{base,cf}.npy), pool.parquet, and
sample_manifest.json from research/data/artifacts/v1/zeroshot_av/ -- no new
target-model forwards. Only the AV runs.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import nla_inference as NLA  # noqa: E402

OUT = REPO / "research/data/artifacts/v1/zeroshot_av"
HF = os.environ["HF_HOME"]
SEED = 20260719
N_SAMPLES = 5
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

av_root = Path(HF) / "hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots"
AV_PATH = str(next(av_root.iterdir()))
print("AV checkpoint:", AV_PATH)

# ── data: saved states + the exact prior sample ─────────────────────────────
df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
pool = pd.read_parquet(OUT / "pool.parquet").reset_index(drop=True)
h_edit_base = torch.from_numpy(np.load(OUT / "h_edit_base.npy"))
h_edit_cf = torch.from_numpy(np.load(OUT / "h_edit_cf.npy"))
assert len(pool) == h_edit_base.shape[0] == h_edit_cf.shape[0]
d_edit = h_edit_cf - h_edit_base

man = json.load(open(OUT / "sample_manifest.json"))
assert man["seed"] == SEED
sample = pool[pool.pair_id.isin(man["pair_ids"])].copy()
sample = sample.sort_values("pair_id").reset_index(drop=True)
assert sample.pair_id.tolist() == sorted(man["pair_ids"])
assert len(sample) == 50
sidx = {p: i for i, p in enumerate(pool.pair_id)}
sample_pos = sample.pair_id.map(sidx).to_numpy()
print(f"sample: {len(sample)} pairs; strata {sample.stratum.value_counts().to_dict()}")

# value / entity inventory + regex rule -- identical to zeroshot_av_edit_site.py
inv_vals = sorted({str(x) for c in ["value_old", "value_new", "value_distractor",
                                     "answer_old", "answer_new"]
                   for x in df[c].dropna() if str(x) not in ("", "None")})
inv_ents = sorted({str(x) for c in ["entity_target", "entity_distractor"]
                   for x in df[c].dropna() if str(x) not in ("", "None")})
val_re = {v: re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE) for v in inv_vals}
ent_re = {e: re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE) for e in inv_ents}

# ── AV load + prompt build (two-step render->encode; sidecar-driven) ────────
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
av_tok = AutoTokenizer.from_pretrained(AV_PATH)
av = AutoModelForCausalLM.from_pretrained(AV_PATH, torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to("cuda").eval()
meta = yaml.safe_load(open(Path(AV_PATH) / "nla_meta.yaml"))
INJ_CHAR = meta["tokens"]["injection_char"]
INJ_ID = meta["tokens"]["injection_token_id"]
INJ_L = meta["tokens"]["injection_left_neighbor_id"]
INJ_R = meta["tokens"]["injection_right_neighbor_id"]
INJ_SCALE = float(meta["extraction"]["injection_scale"])
AV_TEMPLATE = meta["prompt_templates"]["av"]
embed_scale = NLA.resolve_embed_scale(AV_PATH)
print(f"inj_char={INJ_CHAR!r} id={INJ_ID} scale={INJ_SCALE} embed_scale={embed_scale}")

content = AV_TEMPLATE.format(injection_char=INJ_CHAR)
rendered = av_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)
prompt_ids = av_tok(rendered, add_special_tokens=False).input_ids
matches = [i for i, t in enumerate(prompt_ids) if t == INJ_ID]
assert len(matches) == 1, f"injection token appears {len(matches)}x (expected 1)"
_p = matches[0]
assert prompt_ids[_p - 1] == INJ_L and prompt_ids[_p + 1] == INJ_R
print(f"injection at pos {_p}/{len(prompt_ids)}, neighbors verified")
prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    base_embeds = (av.get_input_embeddings()(prompt_ids_t.to("cuda"))
                   * embed_scale).float().cpu()
T = len(prompt_ids)
EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


@torch.no_grad()
def gen_batch(vectors):
    B = len(vectors)
    emb = base_embeds.repeat(B, 1, 1).clone()
    for k, v in enumerate(vectors):
        vs = NLA.normalize_activation(v.float().view(1, -1), INJ_SCALE)
        emb[k] = NLA.inject_at_marked_positions(
            prompt_ids_t, emb[k:k+1], vs, INJ_ID, INJ_L, INJ_R)[0]
    emb = emb.to("cuda", torch.bfloat16)
    mask = torch.ones(B, T, dtype=torch.long, device="cuda")
    out = av.generate(inputs_embeds=emb, attention_mask=mask,
                      do_sample=True, temperature=1.0, top_p=1.0,
                      num_return_sequences=N_SAMPLES, max_new_tokens=220,
                      pad_token_id=av_tok.eos_token_id)
    texts = av_tok.batch_decode(out, skip_special_tokens=True)
    return [texts[k*N_SAMPLES:(k+1)*N_SAMPLES] for k in range(B)]


# ── the -delta arm ──────────────────────────────────────────────────────────
records, dump = [], []
BATCH = 12
jobs = [(i, -d_edit[sample_pos[i]]) for i in range(len(sample))]
for lo in range(0, len(jobs), BATCH):
    chunk = jobs[lo:lo + BATCH]
    outs = gen_batch([v for _, v in chunk])
    for (pair_i, _), samples in zip(chunk, outs):
        row = sample.iloc[pair_i]
        vnew, vold, ent = row.value_new, row.value_old, row.entity_target
        for s, raw in enumerate(samples):
            m = EXPL_RE.search(raw)
            expl = m.group(1).strip() if m else raw.strip()
            others = [v for v in inv_vals if v not in (vnew, vold) and val_re[v].search(expl)]
            rec = {
                "pair_id": row.pair_id, "semantic_id": row.semantic_id,
                "stratum": row.stratum, "query_order": row.query_order,
                "arm": "neg_delta", "sample": s, "shuffled_from": "",
                "value_new": vnew, "value_old": vold, "entity": ent,
                "new_hit": bool(val_re[vnew].search(expl)),
                "old_hit": bool(val_re[vold].search(expl)),
                "n_other_values": len(others),
                "other_values": ";".join(others[:8]),
                "entity_hit": bool(ent_re.get(ent, re.compile(r"$^")).search(expl)),
                "no_explanation_tag": m is None,
                "expl_len_chars": len(expl),
            }
            records.append(rec)
            dump.append({**{k: rec[k] for k in ("pair_id", "arm", "sample",
                          "value_old", "value_new", "entity")},
                         "explanation": expl, "raw_if_no_tag": raw if m is None else ""})
    print(f"  {min(lo+BATCH,len(jobs))}/{len(jobs)} vectors done", flush=True)

res = pd.DataFrame(records)
res.to_parquet(OUT / "negdelta_scored.parquet", index=False)
json.dump(dump, open(OUT / "negdelta_explanations.json", "w"), indent=1)
print(f"wrote {len(res)} scored generations")


# ── rates + pair bootstrap (same estimator as prior run) ────────────────────
def pair_bootstrap(d, col, nboot=2000):
    per_pair = d.groupby("semantic_id")[col].mean()
    vals = per_pair.to_numpy()
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(nboot)]
    return per_pair.mean(), np.percentile(boots, [2.5, 97.5])

print("\n=== -delta arm rates (pair-bootstrap 95% CI) ===")
summary = {"arm": "neg_delta", "n_gen": len(res)}
for col in ["new_hit", "old_hit", "entity_hit"]:
    m, (lo, hi) = pair_bootstrap(res, col)
    summary[col] = round(float(m), 3)
    summary[f"{col}_ci"] = [round(float(lo), 3), round(float(hi), 3)]
    print(f"  {col}: {m:.3f} [{lo:.3f},{hi:.3f}]")
for stratum, g in res.groupby("stratum"):
    m, (lo, hi) = pair_bootstrap(g, "old_hit")
    summary[f"old_hit_{stratum}"] = round(float(m), 3)
    print(f"  old_hit [{stratum}]: {m:.3f} [{lo:.3f},{hi:.3f}] ({g.semantic_id.nunique()} pairs)")
summary["mean_other_vals"] = round(float(res.n_other_values.mean()), 2)
summary["no_tag_rate"] = round(float(res.no_explanation_tag.mean()), 3)

# paired comparison vs the prior real arm: -delta old-rate vs real new-rate
prior = pd.read_parquet(OUT / "scored.parquet")
real_new = prior[prior.arm == "real"].groupby("semantic_id").new_hit.mean()
neg_old = res.groupby("semantic_id").old_hit.mean()
common = real_new.index.intersection(neg_old.index)
diff = (neg_old[common] - real_new[common]).to_numpy()
boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(5000)]
print(f"\nPAIRED  negdelta_old - real_new = {diff.mean():+.3f} "
      f"[{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}] (n={len(common)})")
summary["negold_minus_realnew"] = float(diff.mean())
summary["negold_minus_realnew_ci"] = [float(np.percentile(boots, 2.5)),
                                      float(np.percentile(boots, 97.5))]
json.dump(summary, open(OUT / "negdelta_summary.json", "w"), indent=1)
print("\nDONE")
