# LLM Fine-Tuning Notes (Qwen2.5-Coder + OpenCoder)

These notes consolidate everything we discussed: environment setup on Windows, dataset inspection, LoRA SFT training, merging/exporting, evaluation, Ollama integration, Hugging Face publishing, troubleshooting, and interview-ready talking points.

Repo: `llm-fine-tuning`

- Model: `Qwen/Qwen2.5-Coder-0.5B`
- Dataset: `OpenCoder-LLM/opc-sft-stage2` (configs: `educational_instruct`, `evol_instruct`, `mceval_instruct`, `package_instruct`)
- Approach: LoRA SFT (parameter-efficient fine-tuning), optional merge for deployment


## Quick glossary

- SFT (Supervised Fine-Tuning): Train with input/output pairs to steer model behavior.
- LoRA: Train low-rank adapters on top of the base model for efficiency; small checkpoints, fast training.
- Packing: Concatenate multiple samples into sequences to improve training efficiency (fewer paddings).
- Chat template: `tokenizer.apply_chat_template` converts message dicts into the model’s prompt format.
- Merge: Combine LoRA adapters into the base weights to get a single deployable model.


## Environment (Windows + PowerShell)

Recommended: Python 3.10–3.11; venv; CUDA-enabled PyTorch (for NVIDIA GPUs).

1) Create and activate venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
# If you need CUDA build of torch (CUDA 12.4 wheels)
# pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

3) Verify CUDA

```powershell
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
		print('Device:', torch.cuda.get_device_name(0))
PY
```

Tip: On a laptop GPU like RTX 3050, enable TF32 in training (done in code) and prefer fp16. Expect better throughput with shorter sequences and packing.


## Project structure (key files)

- `src/fine_tune.py`: Main LoRA SFT trainer (robust to transformer/TRL versions)
- `src/inspect_dataset.py`: Inspect dataset columns/configs and sample rows
- `src/evaluate.py`: Quick interactive generation and small checks
- `src/export_to_ollama.py`: Create an Ollama Modelfile from local/HF model
- `models/`: Training outputs (LoRA adapters and/or merged model)
- `data/`: Misc data notes (dataset itself is streamed/cached by `datasets`)


## Dataset inspection

The dataset requires a config. Examples: `educational_instruct`, `evol_instruct`, etc.

List configs and show samples:

```powershell
python -m src.inspect_dataset `
	--dataset OpenCoder-LLM/opc-sft-stage2 `
	--config educational_instruct `
	--split train `
	--limit 3
```

Outputs include:
- Columns, row count (e.g., ~118k rows in some configs)
- A language heuristic
- Sample rows (instruction/code/output fields)


## Training (LoRA SFT)

Minimal example (GPU recommended):

```powershell
python -m src.fine_tune `
	--base_model Qwen/Qwen2.5-Coder-0.5B `
	--dataset OpenCoder-LLM/opc-sft-stage2 `
	--dataset_config educational_instruct `
	--output_dir models/qwen2.5-coder-0.5b-opc-sft-lora `
	--max_seq_length 1024 `
	--per_device_train_batch_size 1 `
	--gradient_accumulation_steps 16 `
	--learning_rate 2e-4 `
	--num_train_epochs 1 `
	--gradient_checkpointing `
	--fp16 `
	--trust_remote_code
```

Notes:
- The script formats samples into chat messages and applies the model’s chat template automatically.
- It enables TF32 on CUDA; mixed precision toggles are guarded (no fp16/bf16 on CPU).
- It dynamically filters `TrainingArguments` and `SFTTrainer` kwargs to handle lib version differences.
- LoRA targets typical projection modules (q, k, v, o, gate, up, down).

Time savers (pick 1–3):
- Reduce `--max_seq_length` (e.g., 896 or 768)
- Use `--packing` to reduce padding
- Cap steps with `--max_train_samples N` or `--num_train_epochs 0.2` or `--save_steps` for early artifacts
- Smaller batch + higher `--gradient_accumulation_steps` to fit memory

Short, demo-friendly run (≈ 1–3 hours on mid GPU, varies):

```powershell
python -m src.fine_tune `
	--base_model Qwen/Qwen2.5-Coder-0.5B `
	--dataset OpenCoder-LLM/opc-sft-stage2 `
	--dataset_config educational_instruct `
	--output_dir models/qwen2.5-coder-0.5b-opc-sft-lora `
	--max_seq_length 896 `
	--per_device_train_batch_size 2 `
	--gradient_accumulation_steps 8 `
	--learning_rate 2e-4 `
	--max_train_samples 20000 `
	--packing `
	--num_train_epochs 1 `
	--gradient_checkpointing `
	--fp16 `
	--trust_remote_code
```


## Saving, merging, and outputs

By default, the script saves LoRA adapters under `--output_dir`.

- To save a merged full model, pass `--save_merged`. This merges LoRA into the base model and writes a standard HF model folder you can load with `transformers` or Ollama.

Example:

```powershell
python -m src.fine_tune `
	--base_model Qwen/Qwen2.5-Coder-0.5B `
	--dataset OpenCoder-LLM/opc-sft-stage2 `
	--dataset_config educational_instruct `
	--output_dir models/qwen2.5-coder-0.5b-opc-sft-lora `
	--max_seq_length 896 `
	--per_device_train_batch_size 2 `
	--gradient_accumulation_steps 8 `
	--learning_rate 2e-4 `
	--max_train_samples 20000 `
	--packing `
	--num_train_epochs 1 `
	--gradient_checkpointing `
	--fp16 `
	--save_merged `
	--trust_remote_code
```

Artifacts you may see:
- `adapter_config.json`, `adapter_model.safetensors` (LoRA)
- `config.json`, `model.safetensors`, `tokenizer.json`, etc. (merged full model)


## Evaluation (quick check)

Run simple generation tests on either merged model or base + adapter:

```powershell
python -m src.evaluate `
	--model_or_path models/qwen2.5-coder-0.5b-opc-sft-lora `
	--use_merged  # omit to use base + adapter
```

The script formats prompts via chat template and prints outputs across a few languages/domains.


## Export to Ollama

Create a Modelfile pointing to your merged model (local path or HF repo). The helper writes the Modelfile and prints the commands to run.

```powershell
python -m src.export_to_ollama `
	--model_path models/qwen2.5-coder-0.5b-opc-sft-lora `
	--model_name qwen2.5-coder-0.5b-opc-sft
```

Then build and run in Ollama:

```powershell
ollama create qwen2.5-coder-0.5b-opc-sft -f .\Modelfile
ollama run qwen2.5-coder-0.5b-opc-sft
```


## Publish to Hugging Face (optional)

1) Login

```powershell
huggingface-cli login
```

2) During training, enable push:

```powershell
python -m src.fine_tune `
	... `
	--push_to_hub `
	--hub_model_id <your-username>/qwen2.5-coder-0.5b-opc-sft
```

Alternatively, after training:
- Use `huggingface_hub` Python APIs to upload, or
- `git lfs` push the saved model directory to a new HF repo.


## Troubleshooting and tips

- CUDA not available:
	- Ensure torch CUDA wheels installed for your CUDA version (e.g., cu124).
	- Verify with the CUDA check snippet above; restart PowerShell if needed.

- Very slow training / pin_memory warning on CPU:
	- Install CUDA torch and run on GPU; `dataloader_pin_memory` is enabled only when CUDA is present.

- Transformers/TRL arg mismatches:
	- Script introspects signatures and filters unexpected args for both `TrainingArguments` and `SFTTrainer`.
	- If you update libraries and get new warnings, re-run; the script should adapt.

- Dataset requires config:
	- Always pass `--dataset_config`; use `src/inspect_dataset.py` to list and preview.

- OOM (out-of-memory):
	- Lower `--max_seq_length`, increase `--gradient_accumulation_steps`, reduce batch size, enable `--packing`, or use `--max_train_samples`.

- PowerShell line continuation:
	- Use backticks ` for continuation. Avoid Bash heredocs; prefer here-strings or inline.


## Interview-ready concepts and talking points

Be ready to explain the following concisely and with trade-offs:

1) Why LoRA for SFT?
	 - Efficiency: trains a small number of parameters; faster and cheaper; small adapter artifacts; easy to merge or swap.

2) Data shaping and chat templates
	 - Normalizing dataset into messages and using `apply_chat_template` avoids brittle hardcoding across models; ensures system/user/assistant role formatting is correct.

3) Packing sequences
	 - Improves GPU utilization by reducing padding; slight complication in loss masking but handled by TRL SFTTrainer.

4) Precision and performance
	 - Use fp16 on CUDA; enable TF32 on Ampere for speed; gradient checkpointing to reduce memory; accumulation to simulate larger batch sizes.

5) Evaluation stance
	 - Quick smoke tests + simple tasks; for rigorous evals use code-specific benchmarks (e.g., HumanEval, MBPP) or task-focused canaries.

6) Deployment paths
	 - Adapter-only vs merged: adapters are lightweight for experimentation; merged is convenient for serving (Ollama, vLLM, HF Inference). Show both.

7) Reproducibility
	 - Pin deps, fix random seeds, record dataset config/splits/sha; log key hyperparams and training hash in README/notes.

8) Safety and licensing
	 - Acknowledge dataset license and output use; avoid leaking sensitive data; discuss guardrails downstream (filters, prompts, moderation).


## Handy CLI flag reference

- `--dataset_config`: Required config name for `OpenCoder-LLM/opc-sft-stage2`.
- `--max_seq_length`: 768–1024 typical for 0.5B models; lower for speed.
- `--per_device_train_batch_size` + `--gradient_accumulation_steps`: Tune for memory and effective batch.
- `--packing`: Enable sample packing to reduce padding overhead.
- `--fp16`/`--bf16`: Prefer fp16 on consumer GPUs; bf16 on supported GPUs.
- `--gradient_checkpointing`: Reduce memory at some compute overhead.
- `--max_train_samples` / `--num_train_epochs` / `--max_steps`: Bound training time.
- `--save_merged`: Write a fully merged model ready for serving.
- `--push_to_hub` + `--hub_model_id`: Publish to Hugging Face.


## What’s next

- Run a bounded training job to get a demonstrable artifact quickly (hours, not days).
- Merge and export to Ollama; validate local inference.
- Optionally push to HF Hub for a portfolio-ready link.
- Iterate on data curation and hyperparams for quality boosts.


---

Session context highlights

- CUDA verified on RTX 3050 Laptop GPU; training switched from CPU (very slow) to GPU.
- Dataset configs handled; inspection script shows columns and samples.
- Training script robust to library versions; pin_memory toggled only on CUDA.
- Short-run recipes provided for faster iteration; full run possible but time-consuming.

