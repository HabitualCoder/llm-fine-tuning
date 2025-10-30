"""
Fine-tune Qwen2.5-Coder-0.5B on OpenCoder-LLM/opc-sft-stage2 using LoRA (SFT).

Usage (PowerShell):
	# Activate venv first
	# . .\\.venv\\Scripts\\Activate.ps1

  # CPU torch (slow) or install CUDA torch separately per your GPU
  # pip install -r requirements.txt

  python -m src.fine_tune `
	--base_model Qwen/Qwen2.5-Coder-0.5B `
	--dataset OpenCoder-LLM/opc-sft-stage2 `
	--output_dir models/qwen2.5-coder-0.5b-opc-sft-lora `
	--max_seq_length 2048 `
	--per_device_train_batch_size 2 `
	--gradient_accumulation_steps 8 `
	--learning_rate 2e-4 `
	--num_train_epochs 1 `
	--save_merged False

Notes:
- This script formats samples via tokenizer.apply_chat_template so it works across many chat-like schemas.
- It tries to auto-detect fields in the dataset (messages, prompt/response, instruction/output, question/solution).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
)
import inspect
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-Coder on OpenCoder dataset")
	p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-0.5B")
	p.add_argument("--dataset", type=str, default="OpenCoder-LLM/opc-sft-stage2")
	p.add_argument("--dataset_config", type=str, default="", help="Dataset config name (e.g., educational_instruct)")
	p.add_argument("--dataset_split", type=str, default="train")
	p.add_argument("--output_dir", type=str, default="models/qwen2.5-coder-0.5b-opc-sft-lora")
	p.add_argument("--max_seq_length", type=int, default=2048)
	p.add_argument("--per_device_train_batch_size", type=int, default=2)
	p.add_argument("--gradient_accumulation_steps", type=int, default=8)
	p.add_argument("--learning_rate", type=float, default=2e-4)
	p.add_argument("--weight_decay", type=float, default=0.0)
	p.add_argument("--warmup_ratio", type=float, default=0.03)
	p.add_argument("--num_train_epochs", type=float, default=1.0)
	p.add_argument("--logging_steps", type=int, default=10)
	p.add_argument("--save_steps", type=int, default=1000)
	p.add_argument("--eval_steps", type=int, default=0)
	p.add_argument("--max_train_samples", type=int, default=0, help="If >0, subsample the dataset for quick runs")
	p.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
	p.add_argument("--fp16", action="store_true", help="Use float16 if available")
	p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
	p.add_argument("--packing", action="store_true", help="Pack multiple samples into each sequence")
	p.add_argument("--save_merged", action="store_true", help="Merge LoRA into base and save full model at the end")
	p.add_argument("--push_to_hub", action="store_true", help="Push model and tokenizer to Hugging Face Hub")
	p.add_argument("--hub_model_id", type=str, default="", help="Hub repo to push (e.g., username/qwen2.5-coder-opc-sft)")
	p.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True for model/tokenizer")
	p.add_argument("--seed", type=int, default=42)
	return p.parse_args()


def ensure_tokenizer(tokenizer):
	# Set padding and EOS tokens as needed
	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"  # right padding for training
	return tokenizer


def sample_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
	"""Try to normalize a dataset row into chat messages [{role, content}, ...]."""
	# 1) Already chat-formatted
	if "messages" in sample and isinstance(sample["messages"], list):
		msgs = []
		for m in sample["messages"]:
			role = m.get("role") or m.get("speaker") or m.get("from") or "user"
			content = m.get("content") or m.get("text") or m.get("value") or ""
			if not isinstance(content, str):
				content = json.dumps(content, ensure_ascii=False)
			if role not in {"system", "user", "assistant"}:
				# Map unknown senders conservatively to 'user'
				role = "user"
			msgs.append({"role": role, "content": content})
		return msgs

	# 2) instruction-style
	for inst_key, out_key in [("instruction", "output"), ("prompt", "response"), ("question", "answer"), ("query", "response")]:
		if inst_key in sample and out_key in sample:
			user = str(sample[inst_key])
			assistant = str(sample[out_key])
			return [
				{"role": "user", "content": user},
				{"role": "assistant", "content": assistant},
			]

	# 3) code/problem specific fallbacks
	if "problem" in sample and ("solution" in sample or "code" in sample):
		assistant = sample.get("solution") or sample.get("code") or ""
		return [
			{"role": "user", "content": str(sample["problem"])},
			{"role": "assistant", "content": str(assistant)},
		]

	# 4) Last resort: treat whole sample as a single user turn and expect assistant empty
	return [
		{"role": "user", "content": json.dumps(sample, ensure_ascii=False)},
		{"role": "assistant", "content": ""},
	]


def format_with_template(tokenizer, sample: Dict[str, Any]) -> str:
	messages = sample_to_messages(sample)
	# Build a full conversation string using the model's chat template
	try:
		return tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=False,
		)
	except Exception:
		# Fallback: simple concatenation
		chunks = []
		for m in messages:
			chunks.append(f"<{m['role']}>: {m['content']}\n")
		return "\n".join(chunks)


def main():
	args = parse_args()

	# Speed-up on NVIDIA Ampere+ (e.g., RTX 30xx): enable TF32
	if torch.cuda.is_available():
		try:
			torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
			torch.set_float32_matmul_precision("high")
			print("Enabled TF32 matmul for faster training on NVIDIA GPUs.")
		except Exception:
			pass

	# Disable mixed precision flags on CPU to avoid runtime issues
	if not torch.cuda.is_available():
		args.fp16 = False
		args.bf16 = False

	dtype = None
	if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
		dtype = torch.bfloat16
	elif args.fp16 and torch.cuda.is_available():
		dtype = torch.float16

	print("Loading tokenizer and model…")
	tokenizer = AutoTokenizer.from_pretrained(
		args.base_model,
		trust_remote_code=args.trust_remote_code,
		use_fast=True,
	)
	tokenizer = ensure_tokenizer(tokenizer)

	model = AutoModelForCausalLM.from_pretrained(
		args.base_model,
		trust_remote_code=args.trust_remote_code,
		torch_dtype=dtype if dtype is not None else None,
		device_map="auto",
	)

	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()

	print("Loading dataset…", args.dataset)
	ds = load_dataset(args.dataset, args.dataset_config if args.dataset_config else None, split=args.dataset_split)
	if args.max_train_samples and args.max_train_samples > 0:
		ds = ds.select(range(min(args.max_train_samples, len(ds))))

	# Map to 'text' field using the model's chat template
	def map_fn(batch):
		texts = [format_with_template(tokenizer, s) for s in batch]
		return {"text": texts}

	# Datasets map with batched=False requires dict per item; we'll use batched=True with list of dicts
	ds = ds.map(lambda s: {"text": format_with_template(tokenizer, s)}, batched=False, desc="Formatting with chat template")

	print("Configuring LoRA…")
	lora_config = LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=(
			[
				"q_proj",
				"k_proj",
				"v_proj",
				"o_proj",
				"gate_proj",
				"up_proj",
				"down_proj",
			]
		),
	)

	print("Setting up training args…")
	# Build kwargs and filter by TrainingArguments signature for version compatibility
	ta_kwargs = dict(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		num_train_epochs=args.num_train_epochs,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		fp16=args.fp16,
		bf16=args.bf16,
		dataloader_pin_memory=torch.cuda.is_available(),
		lr_scheduler_type="cosine",
		optim="adamw_torch",
		report_to=["none"],
		seed=args.seed,
	)
	# Only include evaluation knobs if the installed transformers supports them
	try:
		sig = inspect.signature(TrainingArguments.__init__)
		params = set(sig.parameters.keys())
		if "evaluation_strategy" in params and args.eval_steps and args.eval_steps > 0:
			ta_kwargs["evaluation_strategy"] = "steps"
		elif "evaluation_strategy" in params:
			ta_kwargs["evaluation_strategy"] = "no"
		if "eval_steps" in params and args.eval_steps and args.eval_steps > 0:
			ta_kwargs["eval_steps"] = args.eval_steps
	except Exception:
		pass

	# Filter kwargs to avoid unexpected keyword errors
	try:
		sig = inspect.signature(TrainingArguments.__init__)
		ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig.parameters}
	except Exception:
		pass

	training_args = TrainingArguments(**ta_kwargs)

	print("Creating SFTTrainer…")
	# Build SFTTrainer kwargs dynamically for TRL version compatibility
	sft_sig = None
	try:
		sft_sig = inspect.signature(SFTTrainer.__init__)
		sft_params = set(sft_sig.parameters.keys())
	except Exception:
		sft_params = set()

	# Some TRL versions don't accept peft_config; pre-wrap model as fallback
	if "peft_config" not in sft_params:
		try:
			model = get_peft_model(model, lora_config)
			lora_cfg_for_trainer = None
		except Exception:
			lora_cfg_for_trainer = lora_config
	else:
		lora_cfg_for_trainer = lora_config

	sft_kwargs = {
		"model": model,
		"train_dataset": ds,
		"args": training_args,
		"max_seq_length": args.max_seq_length,
		"packing": args.packing,
		"dataset_text_field": "text",
	}
	# Pass tokenizer or processing_class depending on TRL version
	if "tokenizer" in sft_params:
		sft_kwargs["tokenizer"] = tokenizer
	elif "processing_class" in sft_params:
		sft_kwargs["processing_class"] = tokenizer
	# Pass peft_config if supported
	if lora_cfg_for_trainer is not None and "peft_config" in sft_params:
		sft_kwargs["peft_config"] = lora_cfg_for_trainer

	# Filter kwargs strictly to avoid unexpected keyword errors
	sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_params}

	trainer = SFTTrainer(**sft_kwargs)

	print("Starting training…")
	trainer.train()

	print("Saving LoRA adapter and tokenizer…")
	os.makedirs(args.output_dir, exist_ok=True)
	trainer.model.save_pretrained(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)

	# Optionally merge LoRA into base weights and save full model (useful for Ollama and general usage)
	if args.save_merged:
		print("Merging LoRA adapters into base model…")
		try:
			merged = trainer.model.merge_and_unload()
		except AttributeError:
			# Some PEFT versions require calling get_peft_model first or accessing base model
			merged = get_peft_model(model, lora_config).merge_and_unload()
		merged_dir = os.path.join(args.output_dir, "merged")
		os.makedirs(merged_dir, exist_ok=True)
		merged.save_pretrained(merged_dir, safe_serialization=True)
		tokenizer.save_pretrained(merged_dir)
		print(f"Merged model saved to: {merged_dir}")

		if args.push_to_hub and args.hub_model_id:
			print("Pushing merged model to Hugging Face Hub…")
			merged.push_to_hub(args.hub_model_id)
			tokenizer.push_to_hub(args.hub_model_id)
	else:
		if args.push_to_hub and args.hub_model_id:
			print("Pushing LoRA adapter to Hugging Face Hub…")
			trainer.model.push_to_hub(args.hub_model_id)
			tokenizer.push_to_hub(args.hub_model_id)

	print("Done.")


if __name__ == "__main__":
	main()

