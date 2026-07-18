"""End-to-end generation pipeline (tokenizer-only).

Order of operations (spec-mandated):
  1. tokenizer audit (gates everything; abort if <16 colors survive)
  2. pool partition (seeded)  — split membership decided here,
  3. family sampling           before any caption is rendered
  4. variant expansion + tokenization + per-pair invariants (family-atomic drop)
  5. causal-masking crossing check, dataset-level invariants (abort on failure)
  6. parquet + manifest + audit table + rejection table + stdout summary
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from . import audit as audit_mod
from . import invariants as inv
from . import writer
from .inventory import (COLOR_CANDIDATES, NAME_CANDIDATES, CITY_CANDIDATES,
                        NONCE_ENTITIES, PREAMBLE_TEXT, preamble_collisions)
from .pairs import (build_family, build_null_family, expand_variants)
from .splits import build_pools, pool_sanity

MIN_COLORS = 16
_CHAT_SENTINEL = "9f3d1c7b2a6e4f80CONTENT_SENTINEL"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer
    tcfg = cfg["tokenizer"]
    revision = tcfg["revision"]
    if not revision or revision == "FILL_ME":
        raise SystemExit("config tokenizer.revision must be an explicit commit SHA")
    local = tcfg.get("local_path")
    if local:
        # offline path: files fetched at the pinned revision on another machine
        # and placed here; provenance = revision (claimed) + file sha256s
        # recorded in the manifest for later hub verification.
        tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(tcfg["model_id"], revision=revision)
    return tok, revision


def tokenizer_file_hashes(cfg: dict) -> dict:
    """sha256 of local tokenizer files (empty when loading from the hub)."""
    import hashlib
    local = cfg["tokenizer"].get("local_path")
    if not local:
        return {}
    out = {}
    for p in sorted(Path(local).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def chat_template_ids(tok, content: str) -> list[int]:
    """apply_chat_template(tokenize=True) normalized to a flat id list across
    transformers API generations (v4 list / v5 dict of Encoding or nested list)."""
    ref = tok.apply_chat_template([{"role": "user", "content": content}],
                                  tokenize=True, add_generation_prompt=True)
    ids = ref["input_ids"] if isinstance(ref, dict) else ref
    if len(ids) and hasattr(ids[0], "ids"):
        return list(ids[0].ids)
    if len(ids) and isinstance(ids[0], list):
        return list(ids[0])
    return list(ids)


def chat_wrap_parts(tok) -> tuple[str, str]:
    """Constant chat-template prefix/suffix around the user content."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _CHAT_SENTINEL}],
        tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(_CHAT_SENTINEL)
    return prefix, suffix


class Sampler:
    """Deterministic family sampler. Cycles shuffled transition pools so unique
    ordered transitions are maximized before repetition."""

    def __init__(self, rng, pools):
        self.rng, self.pools = rng, pools
        self._cursors: dict[str, int] = {}
        self._lists: dict[str, list] = {}

    def _cycle(self, key: str, items: list):
        if key not in self._lists:
            idx = self.rng.permutation(len(items))
            self._lists[key] = [items[i] for i in idx]
            self._cursors[key] = 0
        lst = self._lists[key]
        i = self._cursors[key]
        self._cursors[key] = (i + 1) % len(lst)
        return lst[i]

    def _scan(self, key: str, items: list, ok):
        """Next item from the cycle satisfying predicate; None if a full pass fails."""
        for _ in range(len(items)):
            cand = self._cycle(key, items)
            if ok(cand):
                return cand
        return None

    def s_family(self, axis: str, template_id: str):
        p = self.pools
        if axis == "test_entity":
            ents = list(self.rng.choice(len(p.heldout_nonces), 2, replace=False))
            e1, e2 = p.heldout_nonces[ents[0]], p.heldout_nonces[ents[1]]
        else:
            ents = list(self.rng.choice(len(p.train_nonces), 2, replace=False))
            e1, e2 = p.train_nonces[ents[0]], p.train_nonces[ents[1]]
        t1_pool_key, t1_pool = {
            "test_transition": ("s_ho_tr", p.heldout_color_pairs),
            "test_value": ("s_ho_val", p.heldout_value_pairs),
        }.get(axis, ("s_train_tr", p.train_color_pairs))
        A, B = self._cycle(t1_pool_key, t1_pool)
        pair2 = self._scan("s_train_tr2", p.train_color_pairs,
                           lambda pr: len({pr[0], pr[1], A, B}) == 4)
        if pair2 is None:
            return None
        d, dp = pair2
        return (e1, e2), A, B, d, dp

    def n_family(self, axis: str, template_id: str):
        p = self.pools
        t1_pool_key, t1_pool = {
            "test_transition": ("n_ho_tr", p.heldout_name_pairs),
            "test_name": ("n_ho_name", p.heldout_name_value_pairs),
        }.get(axis, ("n_train_tr", p.train_name_pairs))
        A, B = self._cycle(t1_pool_key, t1_pool)
        c = self._cycle("cities1", p.cities)
        cp = self._scan("cities2", p.cities, lambda x: x != c)
        return (None, None), A, B, c, cp


def sample_semantics(cfg: dict, rng, pools) -> list:
    sampler = Sampler(rng, pools)
    semantics = []
    axis_of = lambda split: "iid" if split in ("train", "dev") else split.replace("test_", "")
    for stratum in ("S", "N"):
        quotas = cfg["strata"][stratum]["families"]
        for split, n in quotas.items():
            axis = axis_of(split)
            for b in range(n):
                if split == "test_context":
                    ctx = pools.context_templates[stratum]
                    tid = ctx[b % len(ctx)]
                else:
                    tt = pools.train_templates[stratum]
                    tid = tt[b % len(tt)]
                args = (sampler.s_family(split, tid) if stratum == "S"
                        else sampler.n_family(split, tid))
                if args is None:
                    continue
                ents, A, B, d, dp = args
                semantics.extend(build_family(
                    stratum=stratum, template_id=tid, split=split, split_axis=axis,
                    entities=ents, A=A, B=B, d=d, dp=dp, bundle_idx=b))
        n_null = cfg["cells"]["null_aa_semantic"].get(stratum, 0)
        for b in range(n_null):
            tt = pools.train_templates[stratum]
            tid = tt[b % len(tt)]
            args = sampler.s_family("iid", tid) if stratum == "S" else sampler.n_family("iid", tid)
            if args is None:
                continue
            ents, A, _, d, _ = args
            semantics.extend(build_null_family(
                stratum=stratum, template_id=tid, split="train",
                entities=ents, A=A, d=d, bundle_idx=b))
    return semantics


def tokenize_rows(rows, tok, chat_prefix: str, chat_suffix: str, pools,
                  preamble_text: str, batch_size: int = 1024):
    """Fill prompts, input_ids, positions and decoded tokens on VariantRows."""
    from dataclasses import replace as dc_replace

    from .pairs import render_contents

    texts, owners = [], []  # owners: (row, which) which in {base, cf, probe}
    for row in rows:
        s = row.semantic
        if row.prompt_format == "chat":
            row.base_prompt = chat_prefix + row.base_content + chat_suffix
            row.cf_prompt = chat_prefix + row.cf_content + chat_suffix
        else:
            row.base_prompt = row.base_content
            row.cf_prompt = row.cf_content
        texts.append(row.base_prompt); owners.append((row, "base"))
        texts.append(row.cf_prompt); owners.append((row, "cf"))
        if s.cell == "NULL_AA":
            # locate the would-be edit position via a probe counterfactual with
            # a different slot1 value; the probe is discarded after the diff
            pool = pools.train_colors if s.stratum == "S" else pools.train_names
            pv = next(v for v in pool if v not in (s.value_old, s.base_vals[1]))
            probe_sem = dc_replace(s, cf_vals=(pv, s.base_vals[1]))
            pt = preamble_text if row.preamble else ""
            _, probe_c = render_contents(probe_sem, row.query_order, pt)
            probe_text = (chat_prefix + probe_c + chat_suffix
                          if row.prompt_format == "chat" else probe_c)
            texts.append(probe_text); owners.append((row, "probe"))

    encoded = []
    for i in range(0, len(texts), batch_size):
        encoded.extend(tok(texts[i:i + batch_size], add_special_tokens=False)["input_ids"])

    probe_ids = {}
    for (row, which), ids in zip(owners, encoded):
        if which == "base":
            row.base_input_ids = ids
        elif which == "cf":
            row.cf_input_ids = ids
        else:
            probe_ids[row.pair_id] = ids

    for row in rows:
        b, c = row.base_input_ids, row.cf_input_ids
        row.final_pos = len(b) - 1
        if row.semantic.cell == "NULL_AA":
            pids = probe_ids.get(row.pair_id, [])
            diffs = ([i for i, (x, y) in enumerate(zip(b, pids)) if x != y]
                     if len(pids) == len(b) else [])
            row.edit_pos = diffs[0] if len(diffs) == 1 else -1
        else:
            if len(b) == len(c):
                diffs = [i for i, (x, y) in enumerate(zip(b, c)) if x != y]
                row.edit_pos = diffs[0] if len(diffs) == 1 else -1
            else:
                row.edit_pos = -1
        if 0 <= row.edit_pos < len(b):
            row.edit_token_decoded_base = tok.decode([b[row.edit_pos]])
            cf_src = c if row.semantic.cell != "NULL_AA" else b
            row.edit_token_decoded_cf = tok.decode([cf_src[row.edit_pos]])
        row.final_token_decoded = tok.decode([b[-1]])
    return rows


def run(config_path: str, out_dir: str | None = None, print_records: int = 5) -> dict:
    cfg = load_config(config_path)
    out = Path(out_dir or cfg["out_dir"])
    rng = np.random.default_rng(cfg["seed"])

    tok, revision = load_tokenizer(cfg)

    # ---- 1. audit ----
    audits = audit_mod.audit_values(tok, {
        "color": COLOR_CANDIDATES, "name": NAME_CANDIDATES,
        "city": CITY_CANDIDATES, "nonce": NONCE_ENTITIES,
    })
    colors = [a.value for a in audit_mod.survivors(audits, "color")]
    names = [a.value for a in audit_mod.survivors(audits, "name")]
    cities = [a.value for a in audit_mod.survivors(audits, "city")]
    nonces = [a.value for a in audit_mod.survivors(audits, "nonce")]
    writer.write_json(audit_mod.audit_table_rows(audits), out / "values_audit.json")

    audit_summary = {
        "colors_survived": len(colors), "colors_total": len(COLOR_CANDIDATES),
        "names_survived": len(names), "names_total": len(NAME_CANDIDATES),
        "cities_survived": len(cities), "cities_total": len(CITY_CANDIDATES),
        "color_survivors": colors, "name_survivors": names, "city_survivors": cities,
    }
    if len(colors) < MIN_COLORS:
        writer.write_json(audit_summary, out / "AUDIT_GATE_FAILED.json")
        raise SystemExit(
            f"AUDIT GATE: only {len(colors)} colors survive single-token filter "
            f"(<{MIN_COLORS}). Stopping per spec — see {out}/values_audit.json")

    collisions = (preamble_collisions(PREAMBLE_TEXT, colors)
                  + preamble_collisions(PREAMBLE_TEXT, names)
                  + preamble_collisions(PREAMBLE_TEXT, cities))
    if collisions:
        raise SystemExit(f"preamble collides with answer values: {collisions}")

    # ---- 2. pools ----
    pools = build_pools(
        rng, colors=colors, names=names, cities=cities, nonces=nonces,
        holdout_cfg=cfg["holdout"],
        s_templates=cfg["strata"]["S"]["templates"],
        s_context_templates=cfg["strata"]["S"]["context_templates"],
        n_templates=cfg["strata"]["N"]["templates"],
        n_context_templates=cfg["strata"]["N"]["context_templates"])
    pool_problems = pool_sanity(pools)
    if pool_problems:
        raise SystemExit(f"pool sanity failed: {pool_problems}")

    # ---- 3./4. sample + tokenize ----
    semantics = sample_semantics(cfg, rng, pools)
    rows = []
    for sem in semantics:
        rows.extend(expand_variants(sem, PREAMBLE_TEXT))
    chat_prefix, chat_suffix = chat_wrap_parts(tok)
    rows = tokenize_rows(rows, tok, chat_prefix, chat_suffix, pools, PREAMBLE_TEXT)

    # our constant-wrap chat splice must equal transformers' own rendering
    chat_row = next((r for r in rows if r.prompt_format == "chat"), None)
    if chat_row is not None:
        ref_ids = chat_template_ids(tok, chat_row.base_content)
        if ref_ids != list(chat_row.base_input_ids):
            raise SystemExit("chat splice != apply_chat_template — template drift")

    answer_ids = {}
    for kind in ("color", "name", "city"):
        for a in audit_mod.survivors(audits, kind):
            answer_ids[a.value] = a.leading_space_id
    audited_single = set(answer_ids)

    rej = inv.RejectionTable()
    attempted: dict[tuple, int] = {}
    bad_families = set()
    for row in rows:
        key = (row.semantic.template_id, row.semantic.cell)
        attempted[key] = attempted.get(key, 0) + 1
        problems = inv.check_variant_row(row, answer_ids=answer_ids,
                                         audited_single_token=audited_single)
        for p in problems:
            rej.add(family_id=row.semantic.family_id,
                    template_id=row.semantic.template_id,
                    stratum=row.semantic.stratum, cell=row.semantic.cell,
                    pair_id=row.pair_id, reason=p)
        if problems:
            bad_families.add(row.semantic.family_id)
    kept = [r for r in rows if r.semantic.family_id not in bad_families]

    # preamble position floor
    min_fp = cfg["preamble"]["min_final_pos"]
    for row in kept:
        if row.preamble and row.final_pos < min_fp:
            raise SystemExit(
                f"preamble row {row.pair_id} final_pos={row.final_pos} < {min_fp}; "
                "preamble too short — adjust PREAMBLE_TEXT")

    # ---- 5. crossings + dataset invariants ----
    by_key = {(r.semantic.semantic_id, r.query_order, r.prompt_format, r.preamble): r
              for r in kept}
    crossing_problems = inv.check_crossings(by_key)
    if crossing_problems:
        raise SystemExit(f"causal-masking check failed: {crossing_problems[:10]}")

    records = [r.to_record(answer_ids) for r in kept]
    ds_problems = inv.check_dataset(records, pools)
    if ds_problems:
        raise SystemExit(f"dataset invariants failed: {ds_problems[:10]}")

    # ---- 6. write ----
    rej_summary = rej.summary()
    rej_summary["rates"] = {
        f"{t}/{c}": {
            "attempted_rows": attempted.get((t, c), 0),
            "rejected_rows": sum(1 for x in rej.rows
                                 if x.template_id == t and x.cell == c),
        }
        for (t, c) in sorted(attempted)}
    rej_summary["note"] = ("families are dropped atomically: one violating row "
                           "rejects every row of its family")

    pools_summary = {
        "train_colors": pools.train_colors, "heldout_colors": pools.heldout_colors,
        "train_names_n": len(pools.train_names), "heldout_names": pools.heldout_names,
        "train_nonces": pools.train_nonces, "heldout_nonces": pools.heldout_nonces,
        "heldout_color_transitions_unordered": sorted(
            {tuple(sorted(p)) for p in pools.heldout_color_pairs}),
        "heldout_name_transitions_unordered": sorted(
            {tuple(sorted(p)) for p in pools.heldout_name_pairs}),
        "cities_unpartitioned": True,
    }

    preamble_rows_fp = [r.final_pos for r in kept if r.preamble]
    extra = {
        "audit_summary": audit_summary,
        "tokenizer_local_file_sha256": tokenizer_file_hashes(cfg),
        "preamble": {
            "text": PREAMBLE_TEXT,
            "raw_token_count": len(tok.encode(PREAMBLE_TEXT, add_special_tokens=False)),
            "min_final_pos_required": min_fp,
            "min_final_pos_observed": min(preamble_rows_fp) if preamble_rows_fp else None,
            "status": "PROVISIONAL — awaiting user sign-off",
        },
        "chat_template": {"prefix": chat_prefix, "suffix": chat_suffix},
        "caption_labels_emitted": ["transition", "NO_CHANGE"],
        "caption_labels_parser_valid_but_unused": ["NOT_IDENTIFIABLE_FROM_THIS_STATE"],
        "variant_axes": {"query_order": ["query_last", "query_first"],
                         "prompt_format": ["raw", "chat"],
                         "preamble": [False, True]},
    }
    manifest = writer.build_manifest(
        config=cfg, tokenizer_sha=revision, records=records,
        rejections_summary=rej_summary, pools_summary=pools_summary, extra=extra)

    writer.write_parquet(records, out / "pairs.parquet")
    writer.write_json(manifest, out / "manifest.json")
    writer.write_json([vars(x) for x in rej.rows], out / "rejections.json")

    # ---- stdout summary ----
    print_summary(manifest, records, print_records, rng)
    return {"manifest": manifest, "records": records, "out": str(out)}


def print_summary(manifest, records, n_samples, rng):
    c = manifest["counts"]
    print("=" * 72)
    print(f"dataset: {manifest['dataset']}   seed={manifest['seed']}")
    print(f"tokenizer: {manifest['model_id']} @ {manifest['tokenizer_revision_sha'][:12]}")
    print(f"rows={c['rows']}  semantic_pairs={c['semantic_pairs']}  "
          f"families={c['families']}  unique_ordered_transitions={c['unique_ordered_transitions']}")
    print(f"caption labels: {c['by_caption_label']}")
    print("-" * 72)
    print("per-cell rows:")
    for k, v in c["by_cell"].items():
        print(f"  {k:>16}: {v}")
    print("per-split rows:")
    for k, v in c["by_split"].items():
        print(f"  {k:>16}: {v}")
    print("per-stratum rows:", c["by_stratum"])
    print("-" * 72)
    rj = manifest["rejections"]
    print(f"rejections: total={rj['total']}")
    if rj["by_reason"]:
        for k, v in rj["by_reason"].items():
            print(f"  {k}: {v}")
    print("-" * 72)
    aud = manifest["audit_summary"]
    print(f"audit: colors {aud['colors_survived']}/{aud['colors_total']}, "
          f"names {aud['names_survived']}/{aud['names_total']}, "
          f"cities {aud['cities_survived']}/{aud['cities_total']}")
    pre = manifest["preamble"]
    print(f"preamble: {pre['raw_token_count']} tokens, min final_pos observed "
          f"{pre['min_final_pos_observed']} (required ≥{pre['min_final_pos_required']}) "
          f"[{pre['status']}]")
    print("=" * 72)
    idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
    print(f"{len(idx)} randomly sampled records (not selected):")
    for i in idx:
        r = records[i]
        print("-" * 72)
        for k, v in r.items():
            if k in ("base_input_ids", "cf_input_ids"):
                print(f"  {k}: len={len(v)} {v}")
            else:
                print(f"  {k}: {v!r}")
