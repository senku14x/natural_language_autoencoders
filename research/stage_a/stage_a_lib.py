"""Stage A shared machinery: model loading, batched forwards, L20 extraction,
final-position patching, and metrics.

Conventions (frozen; see research/CLAUDE.md and the v1 decision doc):
  - layer 20 activation = output of decoder block 20 = HF hidden_states[21]
  - read/patch site = final prompt position (verified == len(input_ids)-1)
  - batches are LEFT-padded so the final position is index -1 for every row;
    position_ids are derived from the attention mask (RoPE correctness)
  - weights bf16, metrics fp32; stored logits fp16
"""

import numpy as np
import torch

MODEL_DIR_ENV = "STAGE_A_MODEL_DIR"
LAYER = 20  # block index; hook on model.model.layers[20] output
DEVICE = "cuda"


def load_model_and_tokenizer(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()
    model.requires_grad_(False)
    cfg = model.config
    assert cfg.num_hidden_layers == 28 and cfg.hidden_size == 3584, (
        f"unexpected architecture: layers={cfg.num_hidden_layers} d={cfg.hidden_size}"
    )
    return model, tok


def left_pad_batch(id_lists, pad_id, pad_to=None):
    """[list[int]] -> input_ids, attention_mask, position_ids (left-padded)."""
    n = len(id_lists)
    L = pad_to or max(len(x) for x in id_lists)
    assert all(len(x) <= L for x in id_lists)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    mask = torch.zeros((n, L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, L - len(x):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(DEVICE), mask.to(DEVICE), pos.to(DEVICE)


# Canonical batch shape for ALL Stage A forwards. Forward results are a
# deterministic function of (input, batch shape); different shapes differ by
# up to ~0.6 logit / ~2.2 h20-norm units (measured, level0.json). Fixing one
# shape makes every base/cf/patched comparison exactly shape-consistent.
CANON_B = 64
CANON_L = 131  # max prompt length in the frozen dataset


@torch.no_grad()
def canonical_forward(model, id_lists, pad_id, patch=None):
    """Forward in canonical-shape chunks of exactly (CANON_B, CANON_L).

    Short final chunks are filled by repeating the last row (results dropped).
    Returns (final_logits fp32 [N, V], h20_final fp32 [N, d]).
    """
    logits_out, h_out = [], []
    with FinalPosPatcher(model) as p:
        for lo in range(0, len(id_lists), CANON_B):
            chunk = list(id_lists[lo:lo + CANON_B])
            n_real = len(chunk)
            while len(chunk) < CANON_B:
                chunk.append(chunk[-1])
            chunk_patch = None
            if patch is not None:
                chunk_patch = patch[lo:lo + n_real]
                if n_real < CANON_B:
                    fill = chunk_patch[-1:].expand(CANON_B - n_real, -1)
                    chunk_patch = torch.cat([chunk_patch, fill])
            ids, mask, pos = left_pad_batch(chunk, pad_id, pad_to=CANON_L)
            p.patch = chunk_patch
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                        logits_to_keep=1)
            logits_out.append(out.logits[:n_real, -1, :].float().cpu())
            h_out.append(p.captured[:n_real].cpu())
    return torch.cat(logits_out), torch.cat(h_out)


class FinalPosPatcher:
    """Forward hook on layers[LAYER]: adds patch vectors at sequence index -1.

    patch: [B, d] fp32 tensor or None (extraction-only mode). The hook also
    captures the (pre-patch) block-20 output at index -1 into .captured.
    """

    def __init__(self, model):
        self.block = model.model.layers[LAYER]
        self.patch = None
        self.captured = None
        self._handle = None

    def __enter__(self):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.captured = h[:, -1, :].detach().float().clone()
            if self.patch is None:
                return output
            patched = (h[:, -1, :].float() + self.patch.to(h.device)).to(h.dtype)
            h = h.clone()
            h[:, -1, :] = patched
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h

        self._handle = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False


@torch.no_grad()
def forward_final(model, id_lists, pad_id, patch=None, patcher=None):
    """One batched forward. Returns (final_logits fp32 [B, V], h20_final fp32 [B, d]).

    patch: optional [B, d] fp32 added at block-20 output, final position.
    patcher: reuse an entered FinalPosPatcher (hook already registered).
    """
    ids, mask, pos = left_pad_batch(id_lists, pad_id)

    def run(p):
        p.patch = patch
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    logits_to_keep=1)
        return out.logits[:, -1, :].float(), p.captured

    if patcher is not None:
        return run(patcher)
    with FinalPosPatcher(model) as p:
        return run(p)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ── metrics ──────────────────────────────────────────────────────────────────

def js_divergence(logits_a, logits_b):
    """Jensen-Shannon divergence (base-2, in [0,1]) between softmax rows. fp32."""
    pa = torch.log_softmax(logits_a, dim=-1)
    pb = torch.log_softmax(logits_b, dim=-1)
    a, b = pa.exp(), pb.exp()
    m = 0.5 * (a + b)
    logm = m.clamp_min(1e-30).log()
    kl_am = (a * (pa - logm)).sum(-1)
    kl_bm = (b * (pb - logm)).sum(-1)
    return (0.5 * (kl_am + kl_bm) / np.log(2.0)).clamp(min=0.0)


def topk_overlap(logits_a, logits_b, k=10):
    ta = logits_a.topk(k, dim=-1).indices
    tb = logits_b.topk(k, dim=-1).indices
    out = torch.zeros(logits_a.shape[0])
    for i in range(logits_a.shape[0]):
        out[i] = len(set(ta[i].tolist()) & set(tb[i].tolist())) / k
    return out
