"""Protein generation GRPO baseline — no asyncflow / rhapsody / dragon.

Uses a custom ``rollout_func`` to implement the original EPGF generation
strategy: at every token step, sample N candidates, score each partial
sequence with ProteinSequenceScorer, and select the next token via
softmax-weighted sampling over the bio scores.

DDP / multi-GPU compatibility
------------------------------
``rollout_func`` is DDP-safe here because generation is entirely local to
each rank — there is no coordination with external generator tasks.  Each
rank calls EPGF on its own copy of the model for its own batch shard.
Gradient sync happens in the backward pass as normal.

The one required care: TRL wraps the model in DDP/FSDP before training
starts, so inside ``rollout_func`` we call
``trainer.accelerator.unwrap_model(trainer.model)`` to get the raw
model before calling ``generate()`` and ``compute_transition_scores()``,
both of which are not defined on the DDP wrapper.

Two reward modes:
  1. Composition-only (default): ProteinSequenceScorer — no external tools.
  2. Structural (opt-in): ColabFold → Foldseek top-hit probability.
     Enable by setting COLABFOLD_PATH, FOLDSEEK_PATH, FOLDSEEK_DB env vars.
     In DDP each rank folds/scores its own shard independently.

Usage
-----
    # single GPU
    python protein_generation/baseline.py

    # multi-GPU, one node
    accelerate launch --multi_gpu protein_generation/baseline.py

    # multi-node (2 nodes × 8 GPUs)
    accelerate launch --num_machines 2 --num_processes 16 \\
        protein_generation/baseline.py
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ProteinModel import ProteinSequenceScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[
        logging.FileHandler("protein_baseline.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("protein_baseline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_HOME        = os.environ.get("HF_HOME", "")
COLABFOLD_PATH = os.environ.get("COLABFOLD_PATH", "")  # "/path/to/colabfold_batch"
FOLDSEEK_PATH  = os.environ.get("FOLDSEEK_PATH", "")   # "/path/to/foldseek"
FOLDSEEK_DB    = os.environ.get("FOLDSEEK_DB",    "")  # "/path/to/afdb50"
STORAGE_DIR    = os.environ.get("STORAGE_DIR", "./protein_baseline_storage")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

USE_STRUCTURAL_REWARD = bool(COLABFOLD_PATH and FOLDSEEK_PATH and FOLDSEEK_DB)

# ---------------------------------------------------------------------------
# Superfamilies
# ---------------------------------------------------------------------------

SUPERFAMILIES: dict[str, str] = {
    "CheY-like superfamily":                                           "chey",
    "Tetratricopeptide-like helical domain superfamily":               "tphd",
    "S-adenosyl-L-methionine-dependent methyltransferase superfamily": "sammt",
    "Thioredoxin-like superfamily":                                    "trx",
}

# ---------------------------------------------------------------------------
# Model / EPGF constants
# ---------------------------------------------------------------------------

MODEL_PATH = "GreatCaptainNemo/ProLLaMA"
LORA_DIR   = "prolora"

PROMPT_TEMPLATE = "[Generate by superfamily] Superfamily=<{superfamily}> Seq=<"

NUM_CANDIDATES  = 8     # N candidates sampled per EPGF step
MAX_SEQ_LEN     = 500   # abandon trajectory after this many generated tokens
BIO_THRESHOLD   = 0.55  # minimum ProteinSequenceScorer score to keep a candidate

# EPGF temperature schedule (anneals over the course of one sequence)
INITIAL_TEMPERATURE = 1.0
FINAL_TEMPERATURE   = 0.001
DECAY_RATE          = 0.1

# ---------------------------------------------------------------------------
# EPGF generation
# ---------------------------------------------------------------------------

def _softmax(x: list[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.array(x) / temperature
    e = np.exp(arr - np.max(arr))
    return e / e.sum()


def epgf_generate_sequence(
    prompt: str,
    model,
    tokenizer,
    num_candidates: int = NUM_CANDIDATES,
    max_seq_len: int = MAX_SEQ_LEN,
    bio_threshold: float = BIO_THRESHOLD,
) -> Optional[dict]:
    """Generate one protein sequence from ``prompt`` using EPGF.

    At each token step:
      1. Sample ``num_candidates`` one-token extensions.
      2. Filter by sequence length and ``ProteinSequenceScorer`` bio score.
      3. Select the winning token via softmax-weighted sampling over bio scores
         (temperature decays across steps).

    Returns a dict with keys ``prompt_id``, ``tokens``, ``logp`` suitable
    for the ``rollout_func`` output format, or ``None`` if all candidates
    were filtered out on the first step.

    ``logp`` is a list of per-token log probabilities.  Because
    ``max_new_tokens=1`` per call, each entry is the log prob of the single
    chosen token — exactly the per-token format TRL expects for GRPO.
    """
    original_prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    gen_cfg = GenerationConfig(
        max_new_tokens=1,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=1.0,
        num_return_sequences=num_candidates,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    temperature = INITIAL_TEMPERATURE
    accumulated_tokens: list[int]  = []
    accumulated_logp:   list[float] = []

    current_prompt = prompt
    while True:
        inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
            output_scores=True,
            return_dict_in_generate=True,
        )

        candidates     = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        # transition_scores shape: [num_candidates, 1] (one new token per call)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_probs = transition_scores.sum(dim=1).cpu().numpy().tolist()

        # Sort by log prob descending; keep top half + any that look complete
        ranked = sorted(
            zip(candidates, outputs.sequences, log_probs),
            key=lambda x: x[2],
            reverse=True,
        )
        top_half   = ranked[: max(1, len(ranked) // 2)]
        top_half_set = set(id(item) for item in top_half)
        complete   = [item for item in ranked
                      if item[0].strip().endswith(">") and id(item) not in top_half_set]
        pool = top_half + complete

        # Filter by length and bio score
        bio_scores:       list[float] = []
        filtered_cands:   list[str]   = []
        filtered_tokens:  list        = []
        filtered_logp:    list[float] = []

        for cand, token_ids, lm_score in pool:
            if len(token_ids) > max_seq_len:
                continue
            seq = cand.split("Seq=<")[-1].split(">")[0].strip()
            try:
                bio_score = ProteinSequenceScorer(seq).get_comprehensive_score()
            except Exception:
                continue
            if bio_score < bio_threshold:
                continue
            bio_scores.append(bio_score)
            filtered_cands.append(cand)
            filtered_tokens.append(token_ids)
            filtered_logp.append(lm_score)

        if not filtered_cands:
            return None

        weights    = _softmax(bio_scores, temperature=temperature)
        temperature = max(FINAL_TEMPERATURE, temperature * DECAY_RATE)

        idx          = np.random.choice(len(filtered_cands), p=weights)
        winner       = filtered_cands[idx]
        winner_token = filtered_tokens[idx][-1].item()
        winner_logp  = filtered_logp[idx]

        accumulated_tokens.append(winner_token)
        accumulated_logp.append(winner_logp)

        if winner.endswith(">"):
            return {
                "prompt_id": original_prompt_ids,
                "tokens":    accumulated_tokens,
                "logp":      accumulated_logp,
            }
        else:
            current_prompt = winner


# ---------------------------------------------------------------------------
# rollout_func — runs EPGF on each rank's local model
# ---------------------------------------------------------------------------

def build_rollout_func(
    num_generations: int,
    max_seq_len:     int   = MAX_SEQ_LEN,
    bio_threshold:   float = BIO_THRESHOLD,
):
    """Return a rollout_func for GRPOTrainer that generates via EPGF.

    DDP notes
    ---------
    * Called independently on each accelerate rank for that rank's local
      batch shard — no cross-rank coordination is needed.
    * ``trainer.accelerator.unwrap_model(trainer.model)`` retrieves the raw
      model before any DDP/FSDP wrapper.  Both ``model.generate()`` and
      ``model.compute_transition_scores()`` must be called on the unwrapped
      model; neither is defined on the DDP wrapper class.
    * The model is switched to eval / no_grad for generation and restored
      to training mode afterwards so the subsequent backward pass is not
      affected.
    """

    def rollout_func(prompts: list[str], trainer: GRPOTrainer, **kwargs):
        model     = trainer.accelerator.unwrap_model(trainer.model)
        tokenizer = trainer.processing_class
        eos_id    = tokenizer.eos_token_id

        prompt_ids_out:     list = []
        completion_ids_out: list = []
        logprobs_out:       list = []

        was_training = model.training
        model.eval()

        with torch.no_grad():
            for prompt in prompts:
                generated = 0
                attempts  = 0
                max_attempts = num_generations * 5

                while generated < num_generations and attempts < max_attempts:
                    result = epgf_generate_sequence(
                        prompt, model, tokenizer,
                        max_seq_len=max_seq_len,
                        bio_threshold=bio_threshold,
                    )
                    attempts += 1
                    if result is None:
                        continue

                    prompt_ids_out.append(result["prompt_id"])
                    completion_ids_out.append(result["tokens"] + [eos_id])
                    logprobs_out.append(result["logp"] + [0.0])
                    generated += 1

                # Pad with empty completions if EPGF never found a valid sequence
                while generated < num_generations:
                    prompt_ids_out.append(
                        tokenizer(prompt, add_special_tokens=True).input_ids
                    )
                    completion_ids_out.append([eos_id])
                    logprobs_out.append([0.0])
                    generated += 1

        if was_training:
            model.train()

        return {
            "prompt_ids":     prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs":       logprobs_out,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Sequence extraction helper
# ---------------------------------------------------------------------------

def _extract_sequence(text: str) -> str:
    if "Seq=<" in text:
        return text.split("Seq=<")[-1].split(">")[0].strip()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", text.upper())


# ---------------------------------------------------------------------------
# Reward: composition-based (DDP-safe, no external tools)
# ---------------------------------------------------------------------------

def sequence_composition_reward(prompts, completions, **kwargs) -> list[float]:
    """Score completions with ProteinSequenceScorer.

    Called independently on each rank's local batch shard — no inter-process
    communication required.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        seq  = _extract_sequence(text)
        try:
            score = ProteinSequenceScorer(seq).get_comprehensive_score() if seq else 0.0
        except Exception:
            score = 0.0
        rewards.append(float(score))
    return rewards


# ---------------------------------------------------------------------------
# Reward: structural via ColabFold + Foldseek (optional)
# ---------------------------------------------------------------------------

def _write_fasta(sequences: list[str], path: str) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{i}\n{seq}\n")


def _fold_sequences(sequences: list[str], work_dir: str) -> str:
    fasta_path = os.path.join(work_dir, "sequences.fasta")
    out_dir    = os.path.join(work_dir, "folded")
    os.makedirs(out_dir, exist_ok=True)
    _write_fasta(sequences, fasta_path)
    result = subprocess.run(
        [COLABFOLD_PATH, fasta_path, out_dir, "--num-recycle", "1"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.warning(f"ColabFold exited {result.returncode}: {result.stderr[:300]}")
    return out_dir


def _run_foldseek(pdb_dir: str, db: str, work_dir: str) -> dict[str, float]:
    out_file = os.path.join(work_dir, "fs_out.txt")
    tmp_dir  = os.path.join(work_dir, "fs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run(
        [
            FOLDSEEK_PATH, "easy-search",
            pdb_dir, db, out_file, tmp_dir,
            "--format-output", "query,target,lddt,prob",
            "--exhaustive-search",
        ],
        capture_output=True, text=True, timeout=600,
    )
    hits: dict[str, list[float]] = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                m = re.match(r"^(\d+)_", parts[0])
                seq_id = m.group(1) if m else parts[0]
                try:
                    hits.setdefault(seq_id, []).append(float(parts[3]))
                except ValueError:
                    pass
    return {sid: max(probs) for sid, probs in hits.items()}


def build_structural_reward_func(superfamily: str):
    db = os.path.join(FOLDSEEK_DB, SUPERFAMILIES.get(superfamily, ""))

    def sequence_structural_reward(prompts, completions, **kwargs) -> list[float]:
        seqs = [
            _extract_sequence(c[0]["content"] if isinstance(c, list) else c)
            for c in completions
        ]
        if not any(seqs):
            return [0.0] * len(seqs)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                pdb_dir = _fold_sequences(seqs, tmp)
                scores  = _run_foldseek(pdb_dir, db, tmp)
            except Exception as e:
                logger.warning(f"Structural reward failed for {superfamily}: {e}")
                return [0.0] * len(seqs)
        return [scores.get(str(i), 0.0) for i in range(len(seqs))]

    sequence_structural_reward.__name__ = (
        f"structural_reward_{SUPERFAMILIES.get(superfamily, 'unknown')}"
    )
    return sequence_structural_reward


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_or_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map: Optional[str] = None,
):
    """Load ProLLaMA + LoRA adapter, resuming from a checkpoint if one exists.

    ``device_map=None`` lets accelerate control device placement for DDP.
    Pass ``device_map="auto"`` only for single-process inference.
    """
    # ProLLaMA uses the Llama-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map=device_map
    )

    has_checkpoint = (
        lora_id
        and os.path.isdir(lora_id)
        and any(Path(lora_id).iterdir())
    )

    if has_checkpoint:
        logger.info(f"Resuming from LoRA checkpoint: {lora_id}")
        model = PeftModel.from_pretrained(base, lora_id, is_trainable=True)
    else:
        logger.info(f"No checkpoint — initialising fresh LoRA on {base_model_id}")
        lora_cfg = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class ArchiveCallback(TrainerCallback):
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            step_dir = os.path.join(self.storage_dir, f"step-{state.global_step}")
            Path(step_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Step {state.global_step} complete — archived to {step_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_path: str       = MODEL_PATH,
    lora_dir: str         = LORA_DIR,
    max_steps: int        = 100,
    save_steps: int       = 25,
    num_generations: int  = 4,
    per_device_batch: int = 2,
    grad_accum: int       = 4,
    max_completion_length: int = 512,
    max_prompt_length: int     = 128,
):
    model, tokenizer = load_model_or_checkpoint(model_path, lora_dir, device_map=None)

    formatted_data = [
        {"prompt": PROMPT_TEMPLATE.format(superfamily=sf)}
        for sf in SUPERFAMILIES.keys()
    ]
    dataset = Dataset.from_list(formatted_data)

    if USE_STRUCTURAL_REWARD:
        logger.info("Structural reward enabled (ColabFold + Foldseek)")
        reward_funcs = [build_structural_reward_func(sf) for sf in SUPERFAMILIES]
    else:
        logger.info("Using composition reward (ProteinSequenceScorer)")
        reward_funcs = [sequence_composition_reward]

    rollout_func = build_rollout_func(num_generations=num_generations)

    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=1.0,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        generation_batch_size=per_device_batch,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=1,
        output_dir=lora_dir,
        run_name="protein-baseline",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        rollout_func=rollout_func,
        reward_funcs=reward_funcs,
        callbacks=[ArchiveCallback(STORAGE_DIR)],
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training")
    trainer.train()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
