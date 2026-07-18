"""Parquet + manifest emission.

input_ids are stored as Arrow `list<int32>` columns — typed arrays with
Arrow's internal offsets buffer, not JSON lists. The manifest documents the
layout (dtype/encoding) so the GPU stage reads them without guessing.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _int32_list(col):
    return pa.array(col, type=pa.list_(pa.int32()))


def write_parquet(records: list[dict], path: Path) -> None:
    if not records:
        raise ValueError("no records to write")
    cols = {k: [r[k] for r in records] for k in records[0]}
    arrays, names = [], []
    for name, values in cols.items():
        if name in ("base_input_ids", "cf_input_ids"):
            arrays.append(_int32_list(values))
        elif name in ("behavioral_screen", "oracle_patch_metrics", "delta_norms"):
            # explicit null placeholder columns for the GPU stage
            arrays.append(pa.array(values, type=pa.string()))
        else:
            arrays.append(pa.array(values))
        names.append(name)
    table = pa.Table.from_arrays(arrays, names=names)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def config_hash(config: dict) -> str:
    """Hash of the dataset-defining config. tokenizer.local_path is excluded —
    it is an access mechanism, not a dataset parameter, and must not make the
    same dataset hash differently across machines."""
    cfg = json.loads(json.dumps(config))  # deep copy
    cfg.get("tokenizer", {}).pop("local_path", None)
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(*, config: dict, tokenizer_sha: str, records: list[dict],
                   rejections_summary: dict, pools_summary: dict,
                   extra: dict) -> dict:
    def count_by(key):
        out = {}
        for r in records:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))

    transitions = sorted({
        f"{r['answer_old']}->{r['answer_new']}"
        for r in records if r["answer_old"] != r["answer_new"]})
    semantic_ids = {r["semantic_id"] for r in records}
    families = {r["family_id"] for r in records}

    return {
        "dataset": "counterfactual_difference_nla_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config["tokenizer"]["model_id"],
        "tokenizer_revision_sha": tokenizer_sha,
        "intended_model_revision_sha": config["tokenizer"]["revision"],
        "code_commit": git_commit(),
        "config_hash": config_hash(config),
        "seed": config["seed"],
        "counts": {
            "rows": len(records),
            "semantic_pairs": len(semantic_ids),
            "families": len(families),
            "unique_ordered_transitions": len(transitions),
            "by_cell": count_by("cell"),
            "by_split": count_by("split"),
            "by_split_axis": count_by("split_axis"),
            "by_stratum": count_by("stratum"),
            "by_caption_label": {
                "transition": sum(1 for r in records
                                  if r["answer_old"] != r["answer_new"]),
                "no_change": sum(1 for r in records
                                 if r["answer_old"] == r["answer_new"]),
            },
        },
        "unique_ordered_transitions_list": transitions,
        "rejections": rejections_summary,
        "pools": pools_summary,
        "token_storage": {
            "format": "parquet arrow list<int32>",
            "note": "typed arrays with Arrow offsets buffers; not JSON lists",
            "columns": ["base_input_ids", "cf_input_ids"],
        },
        **extra,
    }


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))
