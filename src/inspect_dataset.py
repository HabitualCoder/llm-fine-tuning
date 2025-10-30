"""
Quick inspection utility for the OpenCoder dataset (or any Hugging Face dataset).

Examples (PowerShell):
  python -m src.inspect_dataset --dataset OpenCoder-LLM/opc-sft-stage2 --split train --limit 3
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="OpenCoder-LLM/opc-sft-stage2")
    p.add_argument("--config", type=str, default="", help="Dataset config name (e.g., educational_instruct)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--limit", type=int, default=5, help="Print first N samples")
    return p.parse_args()


def detect_language(sample: Dict[str, Any]) -> str:
    for k in ("language", "lang", "code_language"):
        if k in sample and isinstance(sample[k], str):
            return sample[k]
    # heuristic: not reliable; return unknown
    return "unknown"


def main():
    args = parse_args()
    # Some datasets require a config; if none provided, print available configs and exit.
    if not args.config:
        try:
            from datasets import get_dataset_config_names

            configs = get_dataset_config_names(args.dataset)
            if configs:
                print("This dataset requires a config. Available configs:")
                for c in configs:
                    print(" -", c)
                print("\nRe-run with:")
                print(f"  python -m src.inspect_dataset --dataset {args.dataset} --config <one-of-the-above> --split {args.split} --limit {args.limit}")
                return
        except Exception:
            pass
        # If we can't list configs, try default load (may still fail)
    ds = load_dataset(args.dataset, args.config if args.config else None, split=args.split)

    print("Columns:", ds.column_names)
    print("Num rows:", len(ds))

    # language distribution, if available
    lang_counter = Counter()
    for i in range(min(2000, len(ds))):  # sample up to 2k for speed
        lang_counter[detect_language(ds[i])] += 1
    print("\nLanguage distribution (sampled):")
    for lang, cnt in lang_counter.most_common(20):
        print(f"  {lang}: {cnt}")

    print("\nSample rows:")
    for i in range(min(args.limit, len(ds))):
        row = ds[i]
        # Avoid printing huge fields entirely
        preview = {k: (v[:300] + "…" if isinstance(v, str) and len(v) > 300 else v) for k, v in row.items()}
        print(f"[{i}]", preview)


if __name__ == "__main__":
    main()
