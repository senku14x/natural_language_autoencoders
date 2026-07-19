"""Shared full-sequence patch/extract forward (used by the site-decomposition
and distractor edit-site runs)."""

import torch

from stage_a_lib import CANON_B, CANON_L, FinalPosPatcher, left_pad_batch


class SeqPatcher(FinalPosPatcher):
    """Adds a full [B, CANON_L, d] fp32 tensor to the block-20 output."""

    def __enter__(self):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.captured = h.detach().float().cpu().clone()
            if self.patch is None:
                return output
            h = (h.float() + self.patch.to(h.device)).to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + tuple(output[1:])
            return h

        self._handle = self.block.register_forward_hook(hook)
        return self


def seq_forward_factory(model, pad_id):
    @torch.no_grad()
    def seq_forward(id_lists, patch_full=None):
        """Canonical-shape forward with optional [N, CANON_L, d] additive patch.
        Returns (final logits fp32 [N, V], full h20 fp32 [N, CANON_L, d])."""
        lg_out, h_out = [], []
        with SeqPatcher(model) as p:
            for lo in range(0, len(id_lists), CANON_B):
                chunk = list(id_lists[lo:lo + CANON_B])
                n_real = len(chunk)
                while len(chunk) < CANON_B:
                    chunk.append(chunk[-1])
                cp = None
                if patch_full is not None:
                    cp = patch_full[lo:lo + n_real]
                    if n_real < CANON_B:
                        cp = torch.cat([cp, cp[-1:].expand(CANON_B - n_real, -1, -1)])
                ids, mask, pos = left_pad_batch(chunk, pad_id, pad_to=CANON_L)
                p.patch = cp
                out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                            logits_to_keep=1)
                lg_out.append(out.logits[:n_real, -1, :].float().cpu())
                h_out.append(p.captured[:n_real])
        return torch.cat(lg_out), torch.cat(h_out)

    return seq_forward
