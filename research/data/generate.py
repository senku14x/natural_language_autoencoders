#!/usr/bin/env python
"""CLI entry point for Counterfactual Difference NLA v1 dataset generation.

Tokenizer-only — never loads model weights. Everything is reproducible from
the config file + seed; the tokenizer revision must be an explicit commit SHA.

Usage:
  python research/data/generate.py --config research/data/configs/smoke.yaml
  python research/data/generate.py --config research/data/configs/v1.yaml \
      --stage audit            # audit only, no dataset build
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctf_data import pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: config out_dir)")
    ap.add_argument("--stage", choices=["audit", "generate"], default="generate",
                    help="'audit' runs only the tokenizer audit gate")
    ap.add_argument("--print-records", type=int, default=5,
                    help="random full records printed in the summary")
    args = ap.parse_args()

    if args.stage == "audit":
        cfg = pipeline.load_config(args.config)
        out = Path(args.out_dir or cfg["out_dir"])
        tok, _ = pipeline.load_tokenizer(cfg)
        from ctf_data import audit as audit_mod
        from ctf_data.inventory import (COLOR_CANDIDATES, NAME_CANDIDATES,
                                        CITY_CANDIDATES, NONCE_ENTITIES)
        from ctf_data import writer
        audits = audit_mod.audit_values(tok, {
            "color": COLOR_CANDIDATES, "name": NAME_CANDIDATES,
            "city": CITY_CANDIDATES, "nonce": NONCE_ENTITIES})
        writer.write_json(audit_mod.audit_table_rows(audits), out / "values_audit.json")
        for kind in ("color", "name", "city", "nonce"):
            surv = audit_mod.survivors(audits, kind)
            print(f"{kind}: {len(surv)}/{len(audits[kind])} survive: "
                  f"{[a.value for a in surv]}")
        return 0

    pipeline.run(args.config, args.out_dir, args.print_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
