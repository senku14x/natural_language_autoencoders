"""Exploratory: zero-shot AV read of edit-site deltas.

NOT Stage B, NOT a verbalizer result. Question: does the RELEASED AV
(kitft/nla-qwen2.5-7b-L20-av), with no fine-tuning, mention the new value
when handed the edit-site L20 activation delta?

Two phases in one job:
  Phase 1  base Qwen2.5-7B → edit-position h_base/h_cf for ALL eligible
           raw/preamble change rows (for transition-mean coverage); free it.
  Phase 2  load AV; sample 50 semantic pairs; build 6 arms; generate 5 @ T=1;
           string-match score; dump all ~1500 explanations; bootstrap.

Serving note: injection uses the repo's OWN pure functions from
nla_inference.py (load_nla_config / resolve_embed_scale /
normalize_activation / inject_at_marked_positions) — sidecar-driven, nothing
hardcoded. Generation is transformers-native model.generate(inputs_embeds=…),
which runs identical forward math to SGLang's input_embeds path but as a
single reproducible job (no server). Documented deviation from the SGLang
suggestion in the session brief.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/stage_a"))
sys.path.insert(0, str(REPO))
from stage_a_lib import load_model_and_tokenizer  # noqa: E402
from run_site_decomposition_lib import seq_forward_factory  # noqa: E402
import nla_inference as NLA  # noqa: E402

OUT = REPO / "research/data/artifacts/v1/zeroshot_av"
OUT.mkdir(parents=True, exist_ok=True)
HF = os.environ["HF_HOME"]
BASE_MODEL = f"{HF}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
SEED = 20260719
N_PAIRS = 50
N_SAMPLES = 5
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ── locate AV checkpoint dir (snapshot) ─────────────────────────────────────
av_root = Path(HF) / "hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots"
AV_PATH = str(next(av_root.iterdir()))
print("AV checkpoint:", AV_PATH)

# ── data ────────────────────────────────────────────────────────────────────
df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
sc = pd.read_parquet(REPO / "research/data/artifacts/v1/stage_a/screen.parquet")
pool = sc[(sc.prompt_format == "raw") & (sc.preamble)
          & sc.cell.isin(["TARGET_EDIT", "REVERSE"]) & sc.eligible].copy()
w = df.set_index("pair_id")
pool["edit_pos"] = w.loc[pool.pair_id, "edit_pos"].to_numpy()
pool["value_old"] = w.loc[pool.pair_id, "value_old"].to_numpy()
pool["value_new"] = w.loc[pool.pair_id, "value_new"].to_numpy()
pool["entity_target"] = w.loc[pool.pair_id, "entity_target"].to_numpy()
pool = pool.reset_index(drop=True)
print(f"pool: {len(pool)} eligible raw/pre change rows, {pool.semantic_id.nunique()} semantic pairs")

# value / entity inventory
inv_vals = sorted({str(x) for c in ["value_old", "value_new", "value_distractor",
                                     "answer_old", "answer_new"]
                   for x in df[c].dropna() if str(x) not in ("", "None")})
inv_ents = sorted({str(x) for c in ["entity_target", "entity_distractor"]
                   for x in df[c].dropna() if str(x) not in ("", "None")})
val_re = {v: re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE) for v in inv_vals}
ent_re = {e: re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE) for e in inv_ents}

# ── Phase 1: extract edit-position states for the whole pool ─────────────────
print("\n=== Phase 1: edit-position extraction (base model) ===")
base_model, tok = load_model_and_tokenizer(BASE_MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
seq_forward = seq_forward_factory(base_model, pad_id)

base_ids = [list(w.loc[p, "base_input_ids"]) for p in pool.pair_id]
cf_ids = [list(w.loc[p, "cf_input_ids"]) for p in pool.pair_id]
_, h20_b = seq_forward(base_ids)
_, h20_c = seq_forward(cf_ids)
CANON_L = h20_b.shape[1]
lens = np.array([len(x) for x in base_ids])
p_edit = (CANON_L - lens) + pool.edit_pos.to_numpy()
ar = np.arange(len(pool))
h_edit_base = h20_b[ar, p_edit].clone()
h_edit_cf = h20_c[ar, p_edit].clone()
# assert pre-edit deltas exactly zero
delta_full = h20_c - h20_b
pre_max = max((delta_full[i, CANON_L - lens[i]:p_edit[i]].abs().max().item()
               if p_edit[i] > CANON_L - lens[i] else 0.0) for i in range(len(pool)))
assert pre_max == 0.0, f"pre-edit delta nonzero {pre_max}"
d_edit = (h_edit_cf - h_edit_base)
print(f"edit-site delta norms: {d_edit.norm(dim=-1).quantile(torch.tensor([.25,.5,.75])).tolist()}")
np.save(OUT / "h_edit_base.npy", h_edit_base.numpy())
np.save(OUT / "h_edit_cf.npy", h_edit_cf.numpy())
pool[["pair_id", "semantic_id", "stratum", "query_order", "cell",
      "value_old", "value_new", "entity_target", "edit_pos"]].to_parquet(OUT / "pool.parquet")

del base_model, h20_b, h20_c, delta_full
torch.cuda.empty_cache()

# ── sample 50 semantic pairs ────────────────────────────────────────────────
sem_ids = pool.semantic_id.unique()
pick_sem = rng.choice(sem_ids, N_PAIRS, replace=False)
sample = pool[pool.semantic_id.isin(pick_sem)].groupby("semantic_id", as_index=False).first()
sample = sample.sort_values("pair_id").reset_index(drop=True)
sidx = {p: i for i, p in enumerate(pool.pair_id)}
sample_pos = sample.pair_id.map(sidx).to_numpy()
print(f"\nsampled {len(sample)} pairs; strata {sample.stratum.value_counts().to_dict()}, "
      f"query {sample.query_order.value_counts().to_dict()}")
json.dump({"seed": SEED, "pair_ids": sample.pair_id.tolist(),
           "semantic_ids": sample.semantic_id.tolist()},
          open(OUT / "sample_manifest.json", "w"), indent=1)

# ── build arms ──────────────────────────────────────────────────────────────
def transition_mean(i):
    row = sample.iloc[i]
    m = (pool.value_old == row.value_old) & (pool.value_new == row.value_new) \
        & (pool.pair_id != row.pair_id)
    idxs = np.where(m.to_numpy())[0]
    if len(idxs) == 0:
        return None
    return d_edit[idxs].mean(0)

def shuffled_partner(i):
    row = sample.iloc[i]
    cand = pool[(pool.value_old != row.value_old) | (pool.value_new != row.value_new)]
    cand = cand[cand.semantic_id != row.semantic_id]
    j = int(cand.index[rng.integers(len(cand))])
    return d_edit[j], pool.iloc[j].pair_id

ARMS = ["real", "shuffled", "random", "h_cf", "h_base", "transition_mean"]
jobs = []  # (pair_i, arm, vector, extra)
for i in range(len(sample)):
    pos = sample_pos[i]
    jobs.append((i, "real", d_edit[pos], ""))
    sv, spid = shuffled_partner(i)
    jobs.append((i, "shuffled", sv, spid))
    g = torch.from_numpy(rng.standard_normal(d_edit.shape[1])).float()
    jobs.append((i, "random", g, ""))
    jobs.append((i, "h_cf", h_edit_cf[pos], ""))
    jobs.append((i, "h_base", h_edit_base[pos], ""))
    tm = transition_mean(i)
    if tm is not None:
        jobs.append((i, "transition_mean", tm, ""))
print(f"jobs: {len(jobs)} vectors × {N_SAMPLES} samples = {len(jobs)*N_SAMPLES} generations")

# ── Phase 2: AV generation ──────────────────────────────────────────────────
print("\n=== Phase 2: AV generation ===")
import yaml  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
av_tok = AutoTokenizer.from_pretrained(AV_PATH)
av = AutoModelForCausalLM.from_pretrained(AV_PATH, torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to("cuda").eval()
# Read sidecar directly (never hardcode). We build prompt ids via the two-step
# render->encode path per the project's nla-infrastructure doc: one-step
# apply_chat_template(tokenize=True) lets NFKC eat U+320E (㈎ -> "(가)") and
# drops the injection token. Verified this session against transformers 5.14.1.
meta = yaml.safe_load(open(Path(AV_PATH) / "nla_meta.yaml"))
INJ_CHAR = meta["tokens"]["injection_char"]
INJ_ID = meta["tokens"]["injection_token_id"]
INJ_L = meta["tokens"]["injection_left_neighbor_id"]
INJ_R = meta["tokens"]["injection_right_neighbor_id"]
INJ_SCALE = float(meta["extraction"]["injection_scale"])
AV_TEMPLATE = meta["prompt_templates"]["av"]
embed_scale = NLA.resolve_embed_scale(AV_PATH)  # 1.0 for Qwen (no tokenization)
print(f"inj_char={INJ_CHAR!r} id={INJ_ID} scale={INJ_SCALE} embed_scale={embed_scale}")

content = AV_TEMPLATE.format(injection_char=INJ_CHAR)
rendered = av_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)
prompt_ids = av_tok(rendered, add_special_tokens=False).input_ids
matches = [i for i, t in enumerate(prompt_ids) if t == INJ_ID]
assert len(matches) == 1, f"injection token appears {len(matches)}× (expected 1)"
_p = matches[0]
assert prompt_ids[_p - 1] == INJ_L and prompt_ids[_p + 1] == INJ_R, (
    f"neighbor drift: {prompt_ids[_p-1]}/{prompt_ids[_p+1]} vs sidecar {INJ_L}/{INJ_R}")
print(f"injection at pos {_p}/{len(prompt_ids)}, neighbors verified ({INJ_L},{INJ_R})")
prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
embed_layer = av.get_input_embeddings()
with torch.no_grad():
    base_embeds = (embed_layer(prompt_ids_t.to("cuda")) * embed_scale).float().cpu()
T = len(prompt_ids)
print(f"AV prompt length T={T}")
EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


@torch.no_grad()
def gen_batch(vectors):
    """vectors: list of [d] tensors. Returns list of N_SAMPLES texts each."""
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


records, dump = [], []
BATCH = 12
for lo in range(0, len(jobs), BATCH):
    chunk = jobs[lo:lo + BATCH]
    outs = gen_batch([v for _, _, v, _ in chunk])
    for (pair_i, arm, _, extra), samples in zip(chunk, outs):
        row = sample.iloc[pair_i]
        vnew, vold, ent = row.value_new, row.value_old, row.entity_target
        for s, raw in enumerate(samples):
            m = EXPL_RE.search(raw)
            expl = m.group(1).strip() if m else raw.strip()
            no_tag = m is None
            others = [v for v in inv_vals if v not in (vnew, vold) and val_re[v].search(expl)]
            rec = {
                "pair_id": row.pair_id, "semantic_id": row.semantic_id,
                "stratum": row.stratum, "query_order": row.query_order,
                "arm": arm, "sample": s, "shuffled_from": extra,
                "value_new": vnew, "value_old": vold, "entity": ent,
                "new_hit": bool(val_re[vnew].search(expl)),
                "old_hit": bool(val_re[vold].search(expl)),
                "n_other_values": len(others),
                "other_values": ";".join(others[:8]),
                "entity_hit": bool(ent_re.get(ent, re.compile(r"$^")).search(expl)),
                "no_explanation_tag": no_tag,
                "expl_len_chars": len(expl),
            }
            records.append(rec)
            dump.append({**{k: rec[k] for k in ("pair_id", "arm", "sample",
                          "value_old", "value_new", "entity")},
                         "explanation": expl, "raw_if_no_tag": raw if no_tag else ""})
    print(f"  {min(lo+BATCH,len(jobs))}/{len(jobs)} vectors done", flush=True)

res = pd.DataFrame(records)
res.to_parquet(OUT / "scored.parquet", index=False)
json.dump(dump, open(OUT / "all_explanations.json", "w"), indent=1)
print(f"\nwrote {len(res)} scored generations, {len(dump)} explanations")

# ── per-arm rates + pair bootstrap ──────────────────────────────────────────
def pair_bootstrap(d, col, nboot=2000):
    per_pair = d.groupby("semantic_id")[col].mean()
    pids = per_pair.index.to_numpy()
    vals = per_pair.to_numpy()
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(nboot)]
    return per_pair.mean(), np.percentile(boots, [2.5, 97.5])

print("\n=== per-arm rates (pair-bootstrap 95% CI) ===")
summ = []
for arm in ARMS:
    d = res[res.arm == arm]
    if len(d) == 0:
        continue
    row = {"arm": arm, "n_gen": len(d)}
    for col in ["new_hit", "old_hit", "entity_hit"]:
        m, (lo, hi) = pair_bootstrap(d, col)
        row[col] = round(m, 3)
        row[f"{col}_ci"] = f"[{lo:.3f},{hi:.3f}]"
    row["mean_other_vals"] = round(d.n_other_values.mean(), 2)
    row["no_tag_rate"] = round(d.no_explanation_tag.mean(), 3)
    summ.append(row)
summ = pd.DataFrame(summ)
print(summ.to_string(index=False))
summ.to_parquet(OUT / "arm_summary.parquet", index=False)

# headline number: real - shuffled new-value rate, pair-paired bootstrap
real_pp = res[res.arm == "real"].groupby("semantic_id").new_hit.mean()
shuf_pp = res[res.arm == "shuffled"].groupby("semantic_id").new_hit.mean()
common = real_pp.index.intersection(shuf_pp.index)
diff = (real_pp[common] - shuf_pp[common]).to_numpy()
boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(5000)]
print(f"\nHEADLINE  real−shuffled new-value rate = {diff.mean():+.3f} "
      f"[{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}] "
      f"(n={len(common)} pairs)")
json.dump({"real_minus_shuffled_new_hit": float(diff.mean()),
           "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "n_pairs": int(len(common))},
          open(OUT / "headline.json", "w"), indent=1)

# ── plot: mention rates by arm with shuffled/random floor ───────────────────
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
PLOTS = REPO / "research/plots"
BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9a94", "#0b0b0b", "#52514e"
ARM_LABEL = {"real": "real Δ (own pair)", "shuffled": "shuffled Δ (diff. pair)",
             "random": "random Gaussian", "h_cf": "h_cf state (pos. control)",
             "h_base": "h_base state", "transition_mean": "E[Δ|old,new] mean"}
order = [a for a in ["h_cf", "real", "transition_mean", "h_base", "shuffled", "random"]
         if a in set(res.arm)]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
floor = float(res[res.arm.isin(["shuffled", "random"])].groupby("semantic_id").new_hit.mean().mean())
y = np.arange(len(order))
for metric, color, off, lab in [("new_hit", BLUE, -0.22, "new value"),
                                ("old_hit", GRAY, 0.0, "old value"),
                                ("entity_hit", MUTED, 0.22, "edited entity")]:
    rates, los, his = [], [], []
    for a in order:
        d = res[res.arm == a]
        m, (lo, hi) = pair_bootstrap(d, metric)
        rates.append(m); los.append(m - lo); his.append(hi - m)
    ax.barh(y + off, rates, height=0.2, color=color, label=lab,
            xerr=[los, his], error_kw=dict(ecolor=MUTED, lw=0.7, capsize=2))
ax.axvline(floor, color=RED, ls="--", lw=1.2)
ax.annotate(f"shuffled/random new-value floor = {floor:.2f}", (floor, len(order) - 0.5),
            xytext=(5, 0), textcoords="offset points", color=RED, fontsize=8, va="center")
ax.set_yticks(y); ax.set_yticklabels([ARM_LABEL[a] for a in order])
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("mention rate (pair-bootstrap 95% CI)")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Zero-shot AV read of edit-site deltas — value/entity mention rates by arm\n"
             "released kitft Qwen2.5-7B-L20 AV, no fine-tuning; 50 pairs × 5 samples @ T=1",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(PLOTS / "2026-07-19_zeroshot_av_mention_rates.png", dpi=170)
print("wrote", PLOTS / "2026-07-19_zeroshot_av_mention_rates.png")
print("\nDONE")
