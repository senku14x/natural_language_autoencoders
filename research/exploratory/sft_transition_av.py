"""Work item 2, step 2: LoRA SFT of the released kitft L20 AV to emit the
`old -> new` arrow-transition caption from edit-site deltas, plus the
non-negotiable shuffled-delta control run and the dev evaluation.

Two identical training runs (real / shuffled-Delta), same seed, same data
order; only the Delta assignment differs. Eval arms on dev eligible
raw/preamble change rows (frozen Stage A eligibility):
  1. real model    x real Delta        (the result; also per-epoch curve)
  2. shuffled model x real Delta       (Delta-ignoring floor, empirical)
  3. real model    x permuted Delta    (activation dependence at eval; also
                                        scored against the DONOR's labels)

Hyperparameters frozen in 2026-07-19_sft_predictions.md before running.
Usage: python sft_transition_av.py [--smoke]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/data"))
import nla_inference as NLA  # noqa: E402
from ctf_data.rich_captions import parse_arrow_transition  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

OUT = REPO / "research/data/artifacts/v1/sft_transition"
PLOTS = REPO / "research/plots"
HF = os.environ["HF_HOME"]
AV_PATH = str(next((Path(HF) / "hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots").iterdir()))
SEED = 20260719
EPOCHS = 1 if args.smoke else 3
BATCH = 8 if args.smoke else 32
LR = 2e-5
WARMUP_FRAC = 0.05
MAX_NEW = 16
torch.manual_seed(SEED)

# ── data ────────────────────────────────────────────────────────────────────
tr = pd.read_parquet(OUT / "train_rows.parquet")
dv = pd.read_parquet(OUT / "dev_rows.parquet")
d_tr = (np.load(OUT / "train_h_edit_cf.npy") - np.load(OUT / "train_h_edit_base.npy"))
d_dv = (np.load(OUT / "dev_h_edit_cf.npy") - np.load(OUT / "dev_h_edit_base.npy"))
el_t = tr.eligible.to_numpy()
el_d = dv.eligible.to_numpy()
tr_e, d_tr_e = tr[el_t].reset_index(drop=True), d_tr[el_t]
dv_e, d_dv_e = dv[el_d].reset_index(drop=True), d_dv[el_d]
if args.smoke:
    tr_e, d_tr_e = tr_e.iloc[:64], d_tr_e[:64]
    dv_e, d_dv_e = dv_e.iloc[:64], d_dv_e[:64]
print(f"train eligible {len(tr_e)} | dev eligible {len(dv_e)}")

rng = np.random.default_rng(SEED)
perm_train = rng.permutation(len(tr_e))
coinc = np.mean((tr_e.value_old.to_numpy() == tr_e.value_old.to_numpy()[perm_train])
                & (tr_e.value_new.to_numpy() == tr_e.value_new.to_numpy()[perm_train]))
perm_dev = rng.permutation(len(dv_e))
print(f"shuffle coincidence (same ordered transition): train {coinc:.4f}, "
      f"dev {np.mean((dv_e.value_old.to_numpy()==dv_e.value_old.to_numpy()[perm_dev]) & (dv_e.value_new.to_numpy()==dv_e.value_new.to_numpy()[perm_dev])):.4f}")

# ── AV prompt (sidecar-driven, two-step render->encode) ─────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
av_tok = AutoTokenizer.from_pretrained(AV_PATH)
meta = yaml.safe_load(open(Path(AV_PATH) / "nla_meta.yaml"))
INJ_ID = meta["tokens"]["injection_token_id"]
INJ_L = meta["tokens"]["injection_left_neighbor_id"]
INJ_R = meta["tokens"]["injection_right_neighbor_id"]
INJ_SCALE = float(meta["extraction"]["injection_scale"])
content = meta["prompt_templates"]["av"].format(injection_char=meta["tokens"]["injection_char"])
rendered = av_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)
prompt_ids = av_tok(rendered, add_special_tokens=False).input_ids
mpos = [i for i, t in enumerate(prompt_ids) if t == INJ_ID]
assert len(mpos) == 1 and prompt_ids[mpos[0]-1] == INJ_L and prompt_ids[mpos[0]+1] == INJ_R
MARK = mpos[0]
T = len(prompt_ids)
EOS = av_tok.eos_token_id
embed_scale = NLA.resolve_embed_scale(AV_PATH)
print(f"AV prompt T={T}, marker at {MARK}, eos {EOS}, embed_scale {embed_scale}")

cap_ids_tr = [av_tok(c, add_special_tokens=False).input_ids + [EOS]
              for c in tr_e.caption_arrow_transition]
maxc = max(len(c) for c in cap_ids_tr)
print(f"caption token lengths: max {maxc}, "
      f"median {int(np.median([len(c) for c in cap_ids_tr]))}")


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(AV_PATH, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda")
    return m


def build_prompt_embeds(model):
    with torch.no_grad():
        pe = (model.get_input_embeddings()(
            torch.tensor(prompt_ids, device="cuda").unsqueeze(0)) * embed_scale)
    return pe.float().cpu()[0]  # [T, d]


def make_batch(model, prompt_embeds, deltas, cap_ids_list, embed_layer):
    """deltas [B,d] float32 cpu; returns inputs_embeds, attention_mask, labels."""
    B = len(cap_ids_list)
    Lc = max(len(c) for c in cap_ids_list)
    emb = prompt_embeds.unsqueeze(0).repeat(B, 1, 1).clone()  # [B,T,d] fp32
    vs = NLA.normalize_activation(torch.as_tensor(deltas, dtype=torch.float32), INJ_SCALE)
    emb[:, MARK, :] = vs
    cap = torch.full((B, Lc), EOS, dtype=torch.long)
    lab = torch.full((B, T + Lc), -100, dtype=torch.long)
    msk = torch.zeros(B, T + Lc, dtype=torch.long)
    msk[:, :T] = 1
    for i, c in enumerate(cap_ids_list):
        cap[i, :len(c)] = torch.tensor(c)
        lab[i, T:T+len(c)] = torch.tensor(c)
        msk[i, T:T+len(c)] = 1
    with torch.no_grad():
        cap_emb = embed_layer(cap.to("cuda")) * embed_scale
    full = torch.cat([emb.to("cuda", torch.bfloat16), cap_emb.to(torch.bfloat16)], dim=1)
    return full, msk.to("cuda"), lab.to("cuda")


def train_run(arm, deltas):
    print(f"\n=== SFT run: {arm} ===", flush=True)
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    model = fresh_model()
    prompt_embeds = build_prompt_embeds(model)
    cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    embed_layer = model.get_input_embeddings()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, eps=1e-6)
    steps_per_epoch = math.ceil(len(tr_e) / BATCH)
    total = steps_per_epoch * EPOCHS
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(
            math.pi * max(0, s - warm) / max(1, total - warm)))))
    losses, curve = [], []
    order_rng = np.random.default_rng(SEED + 1)
    step = 0
    for ep in range(EPOCHS):
        model.train()
        idx = order_rng.permutation(len(tr_e))
        for lo in range(0, len(idx), BATCH):
            sel = idx[lo:lo + BATCH]
            emb, msk, lab = make_batch(model, prompt_embeds, deltas[sel],
                                       [cap_ids_tr[i] for i in sel], embed_layer)
            out = model(inputs_embeds=emb, attention_mask=msk, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach()))
            if step % 25 == 0:
                print(f"  ep{ep} step {step}/{total} loss {out.loss:.4f}", flush=True)
            step += 1
        if arm == "real":
            m = evaluate(model, prompt_embeds, d_dv_e, dv_e)
            curve.append({"epoch": ep + 1, **{k: m[k] for k in
                          ("old_acc", "new_acc", "pair_acc", "parse_ok")}})
            print(f"  [epoch {ep+1}] dev pair_acc {m['pair_acc']:.3f} "
                  f"old {m['old_acc']:.3f} new {m['new_acc']:.3f}", flush=True)
    model.save_pretrained(OUT / f"adapter_{arm}")
    return model, prompt_embeds, losses, curve


@torch.no_grad()
def evaluate(model, prompt_embeds, deltas, rows, gen_bs=64):
    model.eval()
    texts = []
    for lo in range(0, len(rows), gen_bs):
        B = min(gen_bs, len(rows) - lo)
        emb = prompt_embeds.unsqueeze(0).repeat(B, 1, 1).clone()
        vs = NLA.normalize_activation(
            torch.as_tensor(deltas[lo:lo+B], dtype=torch.float32), INJ_SCALE)
        emb[:, MARK, :] = vs
        emb = emb.to("cuda", torch.bfloat16)
        msk = torch.ones(B, T, dtype=torch.long, device="cuda")
        out = model.generate(inputs_embeds=emb, attention_mask=msk,
                             do_sample=False, max_new_tokens=MAX_NEW,
                             pad_token_id=EOS)
        texts += av_tok.batch_decode(out, skip_special_tokens=True)
    old_ok, new_ok, pair_ok, parse_ok, gen = [], [], [], [], []
    for txt, (_, r) in zip(texts, rows.iterrows()):
        s = txt.strip().split("\n")[0].strip()
        try:
            c = parse_arrow_transition(s)
            p_old, p_new = (c.old, c.new) if c.kind == "change" else (None, None)
            ok = True
        except Exception:
            p_old = p_new = None
            ok = False
        old_ok.append(p_old == r.value_old)
        new_ok.append(p_new == r.value_new)
        pair_ok.append((p_old == r.value_old) and (p_new == r.value_new))
        parse_ok.append(ok)
        gen.append(s)
    res = rows[["pair_id", "semantic_id", "family_id", "stratum",
                "value_old", "value_new"]].copy()
    res["gen"], res["old_ok"], res["new_ok"], res["pair_ok"], res["parse_ok"] = \
        gen, old_ok, new_ok, pair_ok, parse_ok

    def fam_boot(col, d=res, nboot=2000):
        per = d.groupby("family_id")[col].mean()
        v = per.to_numpy()
        b = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(nboot)]
        return float(np.mean(d[col])), [float(np.percentile(b, 2.5)),
                                        float(np.percentile(b, 97.5))]
    m = {}
    for col in ["old_ok", "new_ok", "pair_ok", "parse_ok"]:
        mean, ci = fam_boot(col)
        m[col.replace("_ok", "_acc") if col != "parse_ok" else "parse_ok"] = mean
        m[f"{col}_ci"] = ci
    for st, g in res.groupby("stratum"):
        for col in ["old_ok", "new_ok", "pair_ok"]:
            mean, ci = fam_boot(col, g)
            m[f"{col.replace('_ok','_acc')}_{st}"] = mean
            m[f"{col}_{st}_ci"] = ci
    m["_rows"] = res
    return m


# ── runs ────────────────────────────────────────────────────────────────────
results = {}
real_model, real_pe, real_losses, real_curve = train_run("real", d_tr_e)
results["real_x_real"] = evaluate(real_model, real_pe, d_dv_e, dv_e)
results["real_x_perm"] = evaluate(real_model, real_pe, d_dv_e[perm_dev], dv_e)
# donor-label scoring for the permuted arm (mechanism check)
donor = dv_e.iloc[perm_dev].reset_index(drop=True)
rp = results["real_x_perm"]["_rows"]
rp["donor_pair_ok"] = [(g == f"{o} -> {n}") for g, o, n
                       in zip(rp.gen, donor.value_old, donor.value_new)]
print(f"\nreal x permuted-delta: own-label pair {rp.pair_ok.mean():.3f}, "
      f"DONOR-label pair {rp.donor_pair_ok.mean():.3f}")
del real_model
torch.cuda.empty_cache()

shuf_model, shuf_pe, shuf_losses, _ = train_run("shuffled", d_tr_e[perm_train])
results["shuf_x_real"] = evaluate(shuf_model, shuf_pe, d_dv_e, dv_e)
del shuf_model
torch.cuda.empty_cache()

# ── analytic delta-ignoring floor ───────────────────────────────────────────
mode_pair = tr_e.groupby(["value_old", "value_new"]).size().idxmax()
floor_pair = float(np.mean((dv_e.value_old == mode_pair[0])
                           & (dv_e.value_new == mode_pair[1])))
floor_old = float((dv_e.value_old == tr_e.value_old.mode()[0]).mean())
floor_new = float((dv_e.value_new == tr_e.value_new.mode()[0]).mean())
print(f"\nanalytic floors: pair {floor_pair:.4f} (mode {mode_pair}), "
      f"old {floor_old:.4f}, new {floor_new:.4f}")

# ── dump ────────────────────────────────────────────────────────────────────
summary = {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
           "train_eligible": len(tr_e), "dev_eligible": len(dv_e),
           "shuffle_coincidence_train": float(coinc),
           "analytic_floor": {"pair": floor_pair, "old": floor_old, "new": floor_new,
                              "mode_pair": list(mode_pair)},
           "real_curve": real_curve,
           "real_final_loss_mean_last50": float(np.mean(real_losses[-50:])),
           "shuf_final_loss_mean_last50": float(np.mean(shuf_losses[-50:])),
           "donor_pair_acc_real_x_perm": float(rp.donor_pair_ok.mean())}
for arm, m in results.items():
    rows = m.pop("_rows")
    rows.to_parquet(OUT / f"eval_{arm}.parquet")
    summary[arm] = m
json.dump(summary, open(OUT / ("sft_summary_smoke.json" if args.smoke
                               else "sft_summary.json"), "w"), indent=1)
np.save(OUT / "loss_real.npy", np.array(real_losses))
np.save(OUT / "loss_shuffled.npy", np.array(shuf_losses))

print("\n=== FINAL (dev, family-bootstrap 95% CI) ===")
for arm in ["real_x_real", "shuf_x_real", "real_x_perm"]:
    m = summary[arm]
    print(f"{arm}: pair {m['pair_acc']:.3f} {m['pair_ok_ci']} | "
          f"old {m['old_acc']:.3f} | new {m['new_acc']:.3f} | parse {m['parse_ok']:.3f}")
    for st in ("S", "N"):
        if f"pair_acc_{st}" in m:
            print(f"   {st}: pair {m[f'pair_acc_{st}']:.3f} {m[f'pair_ok_{st}_ci']}",
                  end="")
    print()
print("DONE")
