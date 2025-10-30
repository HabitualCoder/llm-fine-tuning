"""
Create a Modelfile for Ollama from a merged HF model or a Hugging Face repo.

Examples (PowerShell):
  # From a merged local directory
  python -m src.export_to_ollama `
    --source_path models/qwen2.5-coder-0.5b-opc-sft-lora/merged `
    --ollama_model_name my-qwen-coder-ft

  # From a Hugging Face repo
  python -m src.export_to_ollama `
    --hf_repo your-username/qwen2.5-coder-opc-sft `
    --ollama_model_name my-qwen-coder-ft
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_repo", type=str, default="", help="hf repo like username/repo-name to use in FROM hf://…")
    p.add_argument("--source_path", type=str, default="", help="Local merged model directory to reference in Modelfile")
    p.add_argument("--ollama_model_name", type=str, default="my-qwen-coder-ft")
    p.add_argument("--output", type=str, default="Modelfile")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.hf_repo and not args.source_path:
        raise SystemExit("Provide --hf_repo or --source_path")

    if args.hf_repo:
        source_line = f"FROM hf://{args.hf_repo}"
    else:
        # Ollama supports FROM path: for local directory models as well
        p = Path(args.source_path).resolve()
        if not p.exists():
            raise SystemExit(f"source_path does not exist: {p}")
        source_line = f"FROM {p.as_posix()}"

    modelfile = f"""
{source_line}

# Optional: you can customize generation parameters here
PARAMETER temperature 0.2
PARAMETER top_p 0.95

# Keep a simple template, Qwen supports chat templates internally
TEMPLATE """
{{ .Prompt }}
"""
""".strip()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(modelfile + "\n")

    print(f"Wrote {args.output}.")
    print("Next steps:")
    print(f"  ollama create {args.ollama_model_name} -f {args.output}")
    print(f"  ollama run {args.ollama_model_name}")


if __name__ == "__main__":
    main()
