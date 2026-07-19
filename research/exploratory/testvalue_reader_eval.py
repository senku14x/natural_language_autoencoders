"""test_value reader-level eval — the single registered spend.

Phase 1: screen + extract test_value raw/preamble change rows (same rule and
machinery as sft_extract_train_deltas.py).
Phase 2: three arms on eligible rows:
  1. real-SFT adapter x real Delta   (greedy, strict parse)
  2. real-SFT adapter x permuted Delta (floor)
  3. zero-shot released AV x real Delta (5 samples @ T=1, string match)
Predictions frozen in 2026-07-19_testvalue_predictions.md.
Outputs -> research/data/artifacts/v1/testvalue_reader/
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/data"))
sys.path.insert(0, str(REPO / "research/stage_a"))
import nla_inference as NLA  # noqa: E402
from ctf_data.rich_captions import parse_arrow_transition  # noqa: E402
from stage_a_lib import CANON_B, CANON_L, left_pad_batch, load_model_and_tokenizer  # noqa: E402
from run_site_decomposition_lib import SeqPatcher  # noqa: E402

OUT = REPO / "research/data/artifacts/v1/testvalue_reader"
OUT.mkdir(parents=True, exist_ok=True)
HF = os.environ["HF_HOME"]
BASE_MODEL = f"{HF}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
AV_PATH = str(next((Path(HF) / "hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots").iterdir()))
ADAPTER = REPO / "research/data/artifacts/v1/sft_transition/adapter_real"
SEED = 20260719
HELD = {"charcoal", "cream", "green", "lavender"}
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

df = pd.read_parquet(REPO / "research/data/artifacts/v1/pairs.parquet")
audit = json.load(open(REPO / "research/data/artifacts/v1/values_audit.json"))
form_ids = defaultdict(set)
for x in audit:
    if x["kept"] and x["n_tokens"] == 1:
        form_ids[x["value"]].add(x["token_ids"][0])

tv = df[(df.split == "test_value") & (df.prompt_format == "raw") & df.preamble
        & df.cell.isin(["TARGET_EDIT", "REVERSE"])].reset_index(drop=True)
print(f"test_value raw/pre change rows: {len(tv)}")

# ── Phase 1: screen + extract (identical logic to the train pass) ───────────
model, tok = load_model_and_tokenizer(BASE_MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
n = len(tv)
heb = np.empty((n, model.config.hidden_size), dtype=np.float32)
hec = np.empty((n, model.config.hidden_size), dtype=np.float32)
base_ids = [list(x) for x in tv.base_input_ids]
cf_ids = [list(x) for x in tv.cf_input_ids]
edit_pos = tv.edit_pos.to_numpy()
recs = []
with SeqPatcher(model) as p, torch.no_grad():
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
        dfull = hc_full - hb_full
        for i in range(hi - lo):
            s0 = CANON_L - lens[i]
            if pedit[i] > s0:
                m = dfull[i, s0:pedit[i]].abs().max().item()
                assert m == 0.0, f"pre-edit delta nonzero {m}"
        heb[lo:hi] = hb_full[ar, pedit].numpy()
        hec[lo:hi] = hc_full[ar, pedit].numpy()
        for i, (_, row) in enumerate(tv.iloc[lo:hi].iterrows()):
            lg_b, lg_c = out["base"][0][i], out["cf"][0][i]
            am_b, am_c = int(lg_b.argmax()), int(lg_c.argmax())
            mb = float(lg_b[row.answer_token_id_new] - lg_b[row.answer_token_id_old])
            mc = float(lg_c[row.answer_token_id_new] - lg_c[row.answer_token_id_old])
            recs.append({"pair_id": row.pair_id,
                         "base_correct": am_b in form_ids[row.answer_old],
                         "cf_correct": am_c in form_ids[row.answer_new],
                         "margin_base": mb, "margin_cf": mc,
                         "eligible": (am_b in form_ids[row.answer_old])
                                     and (am_c in form_ids[row.answer_new]) and (mc > mb)})
        if (lo // CANON_B) % 4 == 0:
            print(f"  {hi}/{n}", flush=True)

sc = pd.DataFrame(recs)
rows = tv[["pair_id", "semantic_id", "family_id", "stratum", "cell", "query_order",
           "value_old", "value_new", "edit_pos", "caption_arrow_transition"]].merge(sc, on="pair_id")
rows["old_held"] = rows.value_old.isin(HELD)
rows["new_held"] = rows.value_new.isin(HELD)
rows["ep_class"] = rows.old_held.map({False: "seen", True: "held"}) + "->" + \
    rows.new_held.map({False: "seen", True: "held"})
d_all = hec - heb
np.save(OUT / "h_edit_base.npy", heb)
np.save(OUT / "h_edit_cf.npy", hec)
rows.to_parquet(OUT / "rows.parquet")
print(f"\neligibility: {rows.eligible.mean():.3f} ({rows.eligible.sum()}/{len(rows)})")
print(rows.groupby("ep_class").eligible.agg(["mean", "sum", "count"]).to_string())
el = rows.eligible.to_numpy()
rows_e = rows[el].reset_index(drop=True)
d_e = d_all[el]
perm = rng.permutation(len(rows_e))
del model
torch.cuda.empty_cache()

# ── AV prompt (sidecar) ─────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
av_tok = AutoTokenizer.from_pretrained(AV_PATH)
meta = yaml.safe_load(open(Path(AV_PATH) / "nla_meta.yaml"))
INJ_ID = meta["tokens"]["injection_token_id"]
INJ_L, INJ_R = meta["tokens"]["injection_left_neighbor_id"], meta["tokens"]["injection_right_neighbor_id"]
INJ_SCALE = float(meta["extraction"]["injection_scale"])
content = meta["prompt_templates"]["av"].format(injection_char=meta["tokens"]["injection_char"])
rendered = av_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)
prompt_ids = av_tok(rendered, add_special_tokens=False).input_ids
mpos = [i for i, t in enumerate(prompt_ids) if t == INJ_ID]
assert len(mpos) == 1 and prompt_ids[mpos[0]-1] == INJ_L and prompt_ids[mpos[0]+1] == INJ_R
MARK, T = mpos[0], len(prompt_ids)
EOS = av_tok.eos_token_id
embed_scale = NLA.resolve_embed_scale(AV_PATH)


def load_av(with_adapter):
    m = AutoModelForCausalLM.from_pretrained(AV_PATH, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda").eval()
    if with_adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(ADAPTER)).eval()
    with torch.no_grad():
        pe = (m.get_input_embeddings()(
            torch.tensor(prompt_ids, device="cuda").unsqueeze(0)) * embed_scale).float().cpu()[0]
    return m, pe


@torch.no_grad()
def gen(m, pe, deltas, greedy, n_samples, max_new, bs=64):
    texts = []
    for lo in range(0, len(deltas), bs):
        B = min(bs, len(deltas) - lo)
        emb = pe.unsqueeze(0).repeat(B, 1, 1).clone()
        vs = NLA.normalize_activation(
            torch.as_tensor(deltas[lo:lo+B], dtype=torch.float32), INJ_SCALE)
        emb[:, MARK, :] = vs
        emb = emb.to("cuda", torch.bfloat16)
        msk = torch.ones(B, T, dtype=torch.long, device="cuda")
        out = m.generate(inputs_embeds=emb, attention_mask=msk,
                         do_sample=not greedy,
                         temperature=1.0 if not greedy else None,
                         top_p=1.0 if not greedy else None,
                         num_return_sequences=n_samples, max_new_tokens=max_new,
                         pad_token_id=EOS)
        dec = av_tok.batch_decode(out, skip_special_tokens=True)
        texts += [dec[k*n_samples:(k+1)*n_samples] for k in range(B)]
    return texts


def fam_boot(d, col, nboot=2000):
    per = d.groupby("family_id")[col].mean()
    v = per.to_numpy()
    if len(v) == 0:
        return float("nan"), [float("nan")] * 2
    b = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(nboot)]
    return float(np.mean(d[col])), [float(np.percentile(b, 2.5)),
                                    float(np.percentile(b, 97.5))]


# ── Arms 1+2: SFT adapter, greedy ───────────────────────────────────────────
print("\n=== SFT adapter arms ===", flush=True)
sft, sft_pe = load_av(with_adapter=True)


def grade_sft(deltas, tag):
    outs = gen(sft, sft_pe, deltas, greedy=True, n_samples=1, max_new=16)
    res = rows_e.copy()
    gens, old_ok, new_ok, pair_ok, parse_ok = [], [], [], [], []
    for txts, (_, r) in zip(outs, rows_e.iterrows()):
        s = txts[0].strip().split("\n")[0].strip()
        try:
            c = parse_arrow_transition(s)
            po, pn = (c.old, c.new) if c.kind == "change" else (None, None)
            ok = True
        except Exception:
            po = pn = None
            ok = False
        gens.append(s)
        old_ok.append(po == r.value_old)
        new_ok.append(pn == r.value_new)
        pair_ok.append((po == r.value_old) and (pn == r.value_new))
        parse_ok.append(ok)
    res["gen"], res["old_ok"], res["new_ok"], res["pair_ok"], res["parse_ok"] = \
        gens, old_ok, new_ok, pair_ok, parse_ok
    res.to_parquet(OUT / f"eval_sft_{tag}.parquet")
    return res


res_real = grade_sft(d_e, "real")
res_perm = grade_sft(d_e[perm], "perm")
donor = rows_e.iloc[perm].reset_index(drop=True)
donor_ok = [(g == f"{o} -> {n}") for g, o, n
            in zip(res_perm.gen, donor.value_old, donor.value_new)]
print(f"perm arm: own pair {res_perm.pair_ok.mean():.3f}, donor pair {np.mean(donor_ok):.3f}")
del sft
torch.cuda.empty_cache()

# ── Arm 3: zero-shot released AV, string match, 5 @ T=1 ─────────────────────
print("\n=== zero-shot arm ===", flush=True)
zs, zs_pe = load_av(with_adapter=False)
inv_vals = sorted({str(x) for c in ["value_old", "value_new", "value_distractor",
                                     "answer_old", "answer_new"]
                   for x in df[c].dropna() if str(x) not in ("", "None")})
val_re = {v: re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE) for v in inv_vals}
EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
outs = gen(zs, zs_pe, d_e, greedy=False, n_samples=5, max_new=220, bs=16)
zs_rows = []
for txts, (_, r) in zip(outs, rows_e.iterrows()):
    for s_i, raw in enumerate(txts):
        m = EXPL_RE.search(raw)
        expl = m.group(1).strip() if m else raw.strip()
        zs_rows.append({"pair_id": r.pair_id, "family_id": r.family_id,
                        "ep_class": r.ep_class, "sample": s_i,
                        "value_old": r.value_old, "value_new": r.value_new,
                        "new_held": r.new_held, "old_held": r.old_held,
                        "new_hit": bool(val_re[r.value_new].search(expl)),
                        "old_hit": bool(val_re[r.value_old].search(expl)),
                        "explanation": expl})
zs_df = pd.DataFrame(zs_rows)
zs_df.to_parquet(OUT / "eval_zeroshot.parquet")
del zs
torch.cuda.empty_cache()

# ── summary ─────────────────────────────────────────────────────────────────
summary = {"seed": SEED, "eligible": int(len(rows_e)),
           "eligibility_rate": float(rows.eligible.mean()),
           "donor_pair_acc_perm": float(np.mean(donor_ok))}
print("\n=== SFT real-delta arm, by endpoint class ===")
for cls, g in res_real.groupby("ep_class"):
    m, ci = fam_boot(g, "pair_ok")
    print(f"  {cls}: pair {m:.3f} {ci} (n={len(g)})")
    summary[f"sft_pair_{cls}"] = {"acc": m, "ci": ci, "n": int(len(g))}
# per-SIDE accuracy: held-side vs seen-side field accuracy
side = []
for _, r in res_real.iterrows():
    side.append({"family_id": r.family_id, "held": r.old_held, "ok": r.old_ok})
    side.append({"family_id": r.family_id, "held": r.new_held, "ok": r.new_ok})
side = pd.DataFrame(side)
for held, g in side.groupby("held"):
    m, ci = fam_boot(g, "ok")
    lab = "held" if held else "seen"
    print(f"  {lab}-side field accuracy: {m:.3f} {ci} (n={len(g)})")
    summary[f"sft_side_{lab}"] = {"acc": m, "ci": ci, "n": int(len(g))}
summary["sft_parse_rate"] = float(res_real.parse_ok.mean())
summary["sft_pair_overall"] = float(res_real.pair_ok.mean())
summary["perm_pair_own"] = float(res_perm.pair_ok.mean())

# held-side miss substitution table
miss = []
for _, r in res_real.iterrows():
    try:
        c = parse_arrow_transition(r.gen)
        if c.kind == "change":
            if r.old_held and c.old != r.value_old:
                miss.append((r.value_old, c.old))
            if r.new_held and c.new != r.value_new:
                miss.append((r.value_new, c.new))
    except Exception:
        pass
subs = pd.Series([f"{a}->{b}" for a, b in miss]).value_counts()
print("\nheld-side miss substitutions (true->generated):")
print(subs.head(12).to_string() if len(subs) else "  (none)")
summary["held_miss_substitutions"] = subs.head(20).to_dict()

print("\n=== zero-shot arm (string match, mention rate) ===")
for held, g in zs_df.groupby("new_held"):
    per = g.groupby("pair_id").new_hit.mean()
    lab = "held" if held else "seen"
    print(f"  new-value mention, {lab} new: {per.mean():.3f} (n={g.pair_id.nunique()} rows)")
    summary[f"zs_new_{lab}"] = float(per.mean())
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("\nDONE")
