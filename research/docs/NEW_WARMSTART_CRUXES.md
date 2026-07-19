# New warm-start: cruxes, plausible fixes, and what to probe for

**Date:** 2026-07-19 · **Status:** research memo for the next-iteration AV warm-start (more diverse dataset, multiple case types). Grounded in the 2026-07-19 exploratory arc (Stage A → zero-shot AV → old-value probe → position sweep → consequence/held-out). Not part of the frozen v1 commitment; a design aid.

## 0. What we actually established (so the warm-start isn't built on a wrong premise)

On Qwen2.5-7B-Instruct, L20, the controlled binding task, dev split:

1. **The edit-site delta is the two value-token identities.** old and new value are each linearly decodable (~0.95 colors / 0.87 names), the basis is **position-invariant** (transfer 0.92–1.00 across edit-pos 3→122), and it is **general** — *unseen* values are consistently represented (0.89 leave-family-out 1-NN). The released AV, zero-shot, names the new value 74% (vs 0% shuffled/random) — token-identity readout ("Final token X").
2. **The delta is behaviorally blind.** It cannot tell a consequential edit from an inert one (target-vs-distractor AUC ≈ 0.5 at the edit site *and* at the final position, every layer, both query orders). The consequence exists only as **answer-change magnitude at the final position, layers 24–28** (norm-AUC → 0.99), i.e. answer formation.
3. **The AV interface destroys magnitude** (`injection_scale` rescales every injected vector to L2 = 150; AR loss is direction-only). So the one place the consequence lives (late-final magnitude) is exactly what the AV cannot see.

**Consequence for the new warm-start:** a more diverse dataset will make the AV a *better, more general transition reader* (more values, positions, phrasings → robust `old → new`). It will **not**, by itself, create the ability to read a behavioral *consequence* or emit a grounded `NO_CHANGE`, because that information is not present as a readable direction at the read site — no amount of caption diversity conjures a variable the model doesn't represent there.

---

## 1. The cruxes (load-bearing, ranked)

**C1 — Identity vs consequence.** The channel reads *what changed*, not *whether it matters*. Diverse data cannot fix this at the edit site; it's an information-availability fact of the site, not a data-coverage gap. Everything that distinguishes "a real difference-verbalizer" from "a token-transition namer" hinges here.

**C2 — The magnitude channel is severed.** Consequence = answer-change magnitude (late-final), but the AV normalizes magnitude away. So even reading the right site, the AV can't access the signal. `NO_CHANGE` / "how big a change" are structurally ungroundable through the current interface.

**C3 — Reader-is-a-lookup on templated families (DPI bound).** With deterministic captions the delta→caption map is a lookup over (entity, old, new). Held-out *values* now look readable (they're consistently represented), but that still only certifies a *general transition code*, not that language carries anything a probe/lookup doesn't. The language-earns-its-keep claim lives only in **derived-relation** families (answer ≠ edited token) and paraphrase/edit-semantics.

**C4 — Warm-start can manufacture plausible-but-ungrounded captions.** turntrout: an implausibly-initialized NLA still hit near-control reconstruction while emitting overwhelmingly implausible claims; a "good enough" warm-start makes the *format* right without guaranteeing *grounding*. A more fluent, more diverse warm-start raises this risk, not lowers it.

**C5 — Confounds scale with diversity.** More case types = more chances for text-inversion (the caption paraphrasing visible prompt text), decoder-prior leakage (the AV completing from its own weights), and per-format prior differences. The zero-shot AV already wraps the read token in confabulated prose; diversity adds surface area for that.

**C6 — Metric/measurement traps (we hit two today).** Unit-normalizing away the operative magnitude signal (EXP-1); scoring against the wrong target space (EXP-2, input-embedding retrieval → false 0.00). Both would have flipped the conclusion if unchecked.

---

## 2. Plausible fixes (mapped to cruxes)

- **C1 (read the consequence):** three routes, in increasing honesty —
  1. *Read a site that carries it.* Late-final layers (L24–28) hold the answer change — but only as **magnitude/argmax-flip**, so pair with C2. Expect this to collapse to answer-token promotion (the capped v1 outcome).
  2. *Give the reader the query* (query-first / the availability 2×2, backlog Ext-A). The consequence is query-relative; supply the query in-state so relevance can be represented before the read. Test whether query-first states become target/distractor separable (today they did **not** at the edit site — so this needs a later read position, not the edit token).
  3. *Derived-relation families* (answer ≠ edited token; e.g. edit a rule, query an inheriting entity). Here "names the changed token" and "reads the consequence" finally diverge and there is no token-promotion shortcut. **This is the highest-value new case type for the diverse dataset.**
- **C2 (magnitude):** add an explicit scalar magnitude/΄no-change bit channel (backlog Ext-E) *or* derive `NO_CHANGE` from a distributional readout (JS/argmax-flip) the reader is given directly — not from the injected vector's norm (which is gone). Amendment 1's magnitude-language ban stands; this is a separate channel, not caption words.
- **C3 (earn its keep):** evaluate on held-out **values** with a decoder whose output space *includes* them (today's 0.00 retrieval was a closed-vocabulary artifact — unseen values are representationally there); and prioritize derived families where a probe and the caption can diverge.
- **C4 (grounding):** keep exact structured grading; per-format no-activation and shuffled-delta floors; error-semantics arm (wrong/reverse/shuffled captions must mis-steer correspondingly); rehearsal (mix a little native-distribution NLA text) rather than paraphrase to fight distribution collapse.
- **C5 (confounds):** same-tokens-different-state pairs (crossed cells already give this), answers-absent-from-text by construction, cross-family/no-activation baselines run *every time*.
- **C6 (metrics):** always report direction **and** magnitude separately; validate the decoder target space with a decoder-free control (1-NN / clustering); AUC/margin not sampled strings; family bootstrap; register the metric before reading the number.

---

## 3. What to probe for (concrete checklist for the new build)

Run these as the go/no-go battery *before* trusting any caption metric:

1. **Value-identity decodability** at the read site (expect high, position-invariant). Positive control: `new ~ h_cf`, `old ~ h_base`. — *sanity the instrument.*
2. **Consequence DIRECTION** = target-vs-distractor AUC on the **unit-normalized** delta at the read site/layer. This is the number that decides "difference-reader vs token-namer." Today: 0.5 everywhere. *If a new site/design doesn't move this above 0.5, the caption's consequence field is ungrounded.*
3. **Consequence MAGNITUDE** = same on delta **norm** (expect late-final only). Report separately from #2; do not let it stand in for #2.
4. **Held-out VALUES**, two ways: (a) decoder-free 1-NN/cluster consistency (representation present?), (b) a trained reader *naming* them with an inclusive output space (readable?). Don't score generalization with a closed-vocab classifier.
5. **Derived-relation families:** exact accuracy for the *derived* answer transition (entity not named in the edit), vs a probe on the same delta, vs value-transition-mean and output-direction baselines. This is where language could finally beat a lookup.
6. **NO_CHANGE / distractor grounding:** can the system emit `NO_CHANGE` for behaviorally-inert edits, and is it grounded in something the reader can access (not the erased magnitude)? Report the distractor `NO_CHANGE` rate against the token-identity temptation to name the (still-present) edited value.
7. **Controls, every format:** no-activation, shuffled-delta, reverse, matched-norm random, output-token-direction. Real−shuffled gap is the load-bearing number.
8. **Interface re-checks** (they bite silently): sidecar-driven injection; two-step `apply_chat_template(tokenize=False)→encode` (one-step NFKC eats the ㈎ injection token on transformers 5.14.1 — verified); position above the ~10-token / position-50 floor; per-forward canonical batch shape (results are shape-dependent to ~2 norm units).

---

## 4. One-line orientation for the next session

The delta is a **general, position-invariant, token-identity code for the value transition** — good enough that a diverse warm-start will read `old → new` robustly, including unseen values. It is **behaviorally blind**: the consequence lives only as late-final answer-change magnitude, which the AV interface erases. So the next real progress is not more transition data — it is **derived-relation families** (to make consequence-reading distinguishable from token-naming) and a **magnitude/availability channel** (to ground `NO_CHANGE`), each with the direction-vs-magnitude and decoder-free controls that caught today's two measurement traps.

---

## 5. Addendum (2026-07-19, post-SFT arc) — status of the cruxes and checklist

The −Δ / trivial-decoder / SFT / test_value / blindness-re-check arc (see
`temporary_artifacts/2026-07-19_SESSION_SUMMARY_negdelta_lens_sft.md`) updates
this memo as follows:

- **§0.1 prediction confirmed at the reader level.** A LoRA SFT of the released
  AV reads `old -> new` at 0.997 (shuffled-Δ 0.000) and **names held-out
  values** (0.94–1.00 per class; misses are surface garbles, never color
  substitutions). The "a decoder just can't NAME them" limitation was specific
  to closed-vocabulary decoders, as suspected — the AV itself is the inclusive
  decoder and it names them. `test_value` is now **spent** for reader-level
  questions.
- **C1/§0.2 (blindness) strengthened, with a new trap logged.** A nonlinear
  (GBM) re-check surfaced an apparent query_first-only signal (0.66) that is
  an `edit_pos` template artifact (metadata-only baseline 0.87;
  position-matched GBM at chance both query orders). The null now rests on a
  nonlinear, position-controlled footing. **New standing rule:** any
  discriminator on this dataset must position-match (or covariate-control
  `edit_pos`) under query_first and report the metadata-only baseline
  (`temporary_artifacts/2026-07-19_blindness_recheck_report.md`).
- **C3 narrowed to its final form.** Held-out values no longer carry any of
  the language-earns-its-keep burden — they generalize at both representation
  and reader level. The claim remains capped at a (now value-general)
  token-identity code; the entire remaining burden sits on **derived-relation
  families** and paraphrase/edit semantics.
- **New methods fact for C6:** the value code is in *neither* weight-derived
  basis (unembedding top-5 ≤0.04 on deltas AND raw states; input-embedding
  retrieval already null) — never use W_U or embedding readouts as sensitivity
  controls at this site; and variance-ranked PCA positive controls do not
  certify low-variance sensitivity (use planted-direction checks).
- **§3 checklist status:** item 1 ✓ (again, via SFT); item 2 ✓ strengthened
  (nonlinear + position-matched, still 0.5); item 4 ✓ both halves (4a 1-NN
  0.89 prior arc; 4b reader-naming this arc); item 7 ✓ for the transition
  channel (shuffled/permuted/no-Δ-strategy floors all clean). Items 3
  (magnitude channel), 5 (derived families), 6 (NO_CHANGE grounding) remain
  open and are the frontier.
