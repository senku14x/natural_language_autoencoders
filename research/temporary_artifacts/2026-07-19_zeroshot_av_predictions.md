# Registered predictions — zero-shot AV read of edit-site deltas

**Logged BEFORE running any AV inference.** Exploratory experiment; not Stage B, not a verbalizer result. Timestamp: 2026-07-19, before AV checkpoint was ever queried.

## What is being tested

Hand the *released* AV (`kitft/nla-qwen2.5-7b-L20-av`, no fine-tuning) the edit-site L20 activation delta and ask, by pure string matching, whether its free-form explanation mentions the **new** value. The interface rescales every injected vector to L2=150 (magnitude destroyed), so only direction is read.

## Priors this is anchored to

- oakhu & ryan_greenblatt: the Qwen2.5-7B NLA is "noisier than changes to problem constants" and does **not** detect problem-constant changes; a mean-direction "rock" beats it. → strong prior that reading a subtle value-swap **difference** will be at or near the floor.
- Anurin: zero-shot diff-of-means verbalization shows only **qualitative** signal, and only for strong steering directions (anger/pirate/Russian); SAE-feature directions mostly invisible. A single value-token swap is a weak, narrow direction.
- Chalnev / general: the decoder preserves structure but **substitutes entities** — so even a "hit" may be structural, and false-positive value mentions from the decoder prior are expected.
- Interface: differences are OOD (the AV was trained on natural document activations, not differences); raw/preamble edit positions (74–98) are **above** the position-50 training floor, so position is not the confound here — style and difference-ness are.

## Predictions (point estimates + rationale)

1. **Real-delta new-value mention rate: ~15%** (wide uncertainty; I would not be surprised by anything in 5–25%). The edit-site delta is largely (new-token − old-token) content propagated to L20; some new-value direction may survive rescaling, but the AV was never trained to read differences, and the Qwen2.5 NLA is the noisiest of the released family.

2. **Shuffled-delta new-value mention rate: ~3%** — this is the false-positive floor (a real delta from a *different* pair should name *that* pair's value, not this one's). Random-Gaussian arm should be similar or lower (~1–3%), possibly with more degenerate/CJK output.

3. **AV(h_cf) alone naming the value: YES, and at the highest rate of any arm (~40%).** Reading a *state* at the value-token position is far easier than reading a *difference*; the edit token IS the new value's position, so h_cf there should encode it most directly. Still degraded well below 100% by OOD prompt style. Symmetrically, AV(h_base) should name the **old** value at a comparable rate and the new value only at the floor — i.e. pointwise differencing (arm4 names new, arm5 names old) may "work" without the delta ever being read, which would itself be an informative (boring) outcome.

## The number that matters

**real-delta new-value rate − shuffled-delta new-value rate.** Predicted small but positive (~ +10pp), with a real chance of ~0 (a clean null), which given the oakhu prior would be unsurprising and still informative. Pair is the bootstrap unit.

## Registered interpretation rules (before seeing data)

- A null (real ≈ shuffled) is **ambiguous** between "the delta is unreadable by this AV" and "the instrument is insensitive" — the h_cf positive-control arm is what disambiguates: if arm 4 can't name a *state* either, the instrument is simply insensitive here (consistent with oakhu) and the delta null says little; if arm 4 names states but arm 1 doesn't beat shuffled, that isolates difference-reading as the failure.
- Any positive real−shuffled gap is at most decodability-of-a-hint evidence; it is **not** a verbalizer result and licenses nothing about captions or gates.
