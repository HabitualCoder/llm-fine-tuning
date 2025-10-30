"""
Quick evaluation for the (fine-tuned) Qwen2.5-Coder model.

Examples (PowerShell):
  # Evaluate merged full model
  python -m src.evaluate --model_path models/qwen2.5-coder-0.5b-opc-sft-lora/merged

  # Evaluate LoRA adapter (needs base model)
  python -m src.evaluate --base_model Qwen/Qwen2.5-Coder-0.5B `
                         --adapter_path models/qwen2.5-coder-0.5b-opc-sft-lora
"""

from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="", help="Path to a merged full model directory")
    p.add_argument("--base_model", type=str, default="", help="Base model id if using LoRA adapter")
    p.add_argument("--adapter_path", type=str, default="", help="Path to LoRA adapter directory")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


def load_model_and_tokenizer(args):
    if args.model_path:
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, device_map="auto")
        return mdl, tok
    elif args.base_model and args.adapter_path:
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # for generation with past kv
        base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code, device_map="auto")
        mdl = PeftModel.from_pretrained(base, args.adapter_path)
        return mdl, tok
    else:
        raise ValueError("Provide either --model_path for merged model or --base_model + --adapter_path for LoRA.")


def sample_prompts() -> List[List[dict]]:
    # Chat-style prompts across multiple languages
    return [
        [
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome."},
        ],
        [
            {"role": "user", "content": "In C++, implement a function to compute gcd(a, b) using Euclid's algorithm."},
        ],
        [
            {"role": "user", "content": "Write a JavaScript function that flattens a nested array."},
        ],
        [
            {"role": "user", "content": "In Java, write a method to reverse a linked list."},
        ],
    ]


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()

    prompts = sample_prompts()
    for i, messages in enumerate(prompts):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print("\n=== Prompt", i, "===")
        print(messages[0]["content"])
        print("--- Generation ---")
        print(generated)


if __name__ == "__main__":
    main()
