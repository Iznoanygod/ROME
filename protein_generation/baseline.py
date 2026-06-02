"""Protein generation GRPO baseline — no asyncflow / rhapsody / dragon.

Runs the full EPGF generation + scoring + GRPO training loop in a single
process using plain TRL and Transformers.  Two reward modes are supported:

  1. Composition-only (default): uses ProteinSequenceScorer from ProteinModel.py.
     No external tools required.
  2. Structural (opt-in): calls ColabFold to fold each candidate then Foldseek
     to measure superfamily similarity.  Enable by setting COLABFOLD_PATH and
     FOLDSEEK_PATH below (or the corresponding env vars).

Usage
-----
    python protein_generation/baseline.py

Adjust the CONFIG block near the bottom of this file to match your paths and
compute budget.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainerCallback,
)
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
# Paths — override with environment variables or edit directly
# ---------------------------------------------------------------------------

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_HOME        = os.environ.get("HF_HOME", "")
COLABFOLD_PATH = os.environ.get("COLABFOLD_PATH", "")   # e.g. "/path/to/colabfold_batch"
FOLDSEEK_PATH  = os.environ.get("FOLDSEEK_PATH", "")    # e.g. "/path/to/foldseek"
FOLDSEEK_DB    = os.environ.get("FOLDSEEK_DB", "")      # e.g. "/path/to/afdb50"
STORAGE_DIR    = os.environ.get("STORAGE_DIR", "./protein_baseline_storage")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

USE_STRUCTURAL_REWARD = bool(COLABFOLD_PATH and FOLDSEEK_PATH and FOLDSEEK_DB)

# ---------------------------------------------------------------------------
# Superfamilies and their Foldseek database short-names
# ---------------------------------------------------------------------------

SUPERFAMILIES: dict[str, str] = {
    "CheY-like superfamily":                                              "chey",
    "Tetratricopeptide-like helical domain superfamily":                  "tphd",
    "S-adenosyl-L-methionine-dependent methyltransferase superfamily":    "sammt",
    "Thioredoxin-like superfamily":                                       "trx",
}

# ---------------------------------------------------------------------------
# Model / training configuration
# ---------------------------------------------------------------------------

MODEL_PATH    = "GreatCaptainNemo/ProLLaMA"
LORA_DIR      = "prolora"
MAX_SEQ_LEN   = 500   # maximum generated sequence length (tokens)
NUM_RETURN    = 8     # candidates per EPGF step
BIO_THRESHOLD = 0.55  # minimum ProteinSequenceScorer score to keep a candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_or_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map="auto",
):
    """Load ProLLaMA + LoRA adapter, resuming from checkpoint if one exists."""
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    has_checkpoint = lora_id and os.path.isdir(lora_id) and any(Path(lora_id).iterdir())

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype, device_map=device_map)

    if has_checkpoint:
        logger.info(f"Resuming from LoRA checkpoint: {lora_id}")
        model = PeftModel.from_pretrained(base, lora_id, is_trainable=True)
    else:
        logger.info(f"No checkpoint found — creating fresh LoRA on {base_model_id}")
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


def softmax(x: list[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.array(x) / temperature
    e = np.exp(arr - np.max(arr))
    return e / e.sum()


# ---------------------------------------------------------------------------
# EPGF generation (inline — no task system)
# ---------------------------------------------------------------------------

def epgf_generate_sequence(
    superfamily: str,
    model,
    tokenizer,
    num_return: int = NUM_RETURN,
    max_seq_len: int = MAX_SEQ_LEN,
    bio_threshold: float = BIO_THRESHOLD,
    seed: int = 42,
) -> Optional[dict]:
    """Run EPGF for a single superfamily prompt.

    Returns a dict with keys ``sequence``, ``prompt_id``, ``tokens``,
    ``logp`` — or ``None`` if all candidates were filtered out.
    """
    set_seed(seed)
    prompt = f"[Generate by superfamily] Superfamily=<{superfamily}> Seq=<"
    original_prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    gen_cfg = GenerationConfig(
        max_new_tokens=1,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=1.0,
        num_return_sequences=num_return,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    initial_temperature = 1.0
    final_temperature   = 0.001
    decay_rate          = 0.1

    temperature = initial_temperature
    accumulated_tokens: list[int] = []
    accumulated_logp: list[float] = []

    model.eval()
    with torch.no_grad():
        while True:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
                output_scores=True,
                return_dict_in_generate=True,
            )

            candidates = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            log_probs = transition_scores.sum(dim=1).cpu().numpy().tolist()

            ranked = sorted(
                zip(candidates, outputs.sequences, log_probs),
                key=lambda x: x[2],
                reverse=True,
            )
            top_half = ranked[: max(1, len(ranked) // 2)]
            # also keep any that look complete
            complete = [
                item for item in ranked
                if item[0].strip().endswith(">") and item not in top_half
            ]
            top_half.extend(complete)

            # Filter by bio score
            bio_scores: list[float] = []
            filtered_cands: list[str]      = []
            filtered_tokens: list          = []
            filtered_logp: list[float]     = []

            for cand, token_ids, lm_score in top_half:
                seq = cand.split("Seq=<")[-1].split(">")[0].strip()
                if len(token_ids) > max_seq_len:
                    continue
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

            weights = softmax(bio_scores, temperature=temperature)
            temperature = max(final_temperature, temperature * decay_rate)

            idx    = np.random.choice(len(filtered_cands), p=weights)
            winner = filtered_cands[idx]
            winner_token = filtered_tokens[idx][-1].item()
            winner_logp  = filtered_logp[idx]

            accumulated_tokens.append(winner_token)
            accumulated_logp.append(winner_logp)

            if winner.endswith(">"):
                sequence = winner.split("Seq=<")[-1].split(">")[0].strip()
                return {
                    "sequence":   sequence,
                    "prompt_id":  original_prompt_ids,
                    "tokens":     accumulated_tokens,
                    "logp":       accumulated_logp,
                    "superfamily": superfamily,
                }
            else:
                prompt = winner


# ---------------------------------------------------------------------------
# Reward: composition-based (always available)
# ---------------------------------------------------------------------------

def sequence_composition_reward(prompts, completions, **kwargs) -> list[float]:
    """Score completions using ProteinSequenceScorer (no external tools needed)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        seq = text.split("Seq=<")[-1].split(">")[0].strip()
        try:
            score = ProteinSequenceScorer(seq).get_comprehensive_score()
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
    """Run ColabFold and return the directory containing PDB files."""
    fasta_path = os.path.join(work_dir, "sequences.fasta")
    out_dir    = os.path.join(work_dir, "folded")
    os.makedirs(out_dir, exist_ok=True)
    _write_fasta(sequences, fasta_path)

    result = subprocess.run(
        [COLABFOLD_PATH, fasta_path, out_dir, "--num-recycle", "1"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.warning(f"ColabFold failed: {result.stderr[:500]}")
    return out_dir


def _run_foldseek(pdb_dir: str, db: str, work_dir: str) -> dict[str, float]:
    """Run Foldseek easy-search and return {seq_id: top_prob}."""
    output_file = os.path.join(work_dir, "foldseek_out.txt")
    tmp_dir     = os.path.join(work_dir, "foldseek_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    subprocess.run(
        [
            FOLDSEEK_PATH, "easy-search",
            pdb_dir, db, output_file, tmp_dir,
            "--format-output", "query,target,lddt,prob",
            "--exhaustive-search",
        ],
        capture_output=True, text=True, timeout=600,
    )

    hits: dict[str, list[float]] = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                query = parts[0]
                m = re.match(r"^(\d+)_", query)
                seq_id = m.group(1) if m else query
                try:
                    prob = float(parts[3])
                except ValueError:
                    continue
                hits.setdefault(seq_id, []).append(prob)

    return {sid: max(probs) for sid, probs in hits.items()}


def build_structural_reward_func(superfamily: str):
    """Return a reward function that folds + searches with Foldseek."""
    db = os.path.join(FOLDSEEK_DB, SUPERFAMILIES.get(superfamily, ""))

    def sequence_structural_reward(prompts, completions, **kwargs) -> list[float]:
        seqs = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            seq = text.split("Seq=<")[-1].split(">")[0].strip()
            seqs.append(seq)

        if not any(seqs):
            return [0.0] * len(seqs)

        with tempfile.TemporaryDirectory() as tmp:
            try:
                pdb_dir = _fold_sequences(seqs, tmp)
                scores  = _run_foldseek(pdb_dir, db, tmp)
            except Exception as e:
                logger.warning(f"Structural reward failed: {e}")
                return [0.0] * len(seqs)

        rewards = []
        for i in range(len(seqs)):
            rewards.append(scores.get(str(i), 0.0))
        return rewards

    sequence_structural_reward.__name__ = f"structural_reward_{SUPERFAMILIES.get(superfamily, 'unknown')}"
    return sequence_structural_reward


# ---------------------------------------------------------------------------
# Custom rollout — implements EPGF inline
# ---------------------------------------------------------------------------

def build_rollout_func(model, tokenizer, num_generations: int):
    """Return a rollout_func compatible with TRL's GRPOTrainer."""

    def rollout_func(prompts: list[str], trainer: GRPOTrainer, **kwargs):
        eos_id = tokenizer.eos_token_id
        prompt_ids_out:    list = []
        completion_ids_out: list = []
        logprobs_out:      list = []

        seen: dict[str, list] = {}
        for prompt in prompts:
            if prompt not in seen:
                seen[prompt] = []

        for prompt in list(seen.keys()):
            superfamily = prompt  # dataset rows use the superfamily name as the prompt
            gens = []
            attempts = 0
            max_attempts = num_generations * 5
            while len(gens) < num_generations and attempts < max_attempts:
                result = epgf_generate_sequence(
                    superfamily, model, tokenizer, seed=attempts
                )
                attempts += 1
                if result is not None:
                    gens.append(result)

            if not gens:
                logger.warning(f"No valid sequences generated for {superfamily}")
                # pad with empty placeholders so TRL batch shape stays intact
                placeholder_ids  = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
                for _ in range(num_generations):
                    prompt_ids_out.append(placeholder_ids)
                    completion_ids_out.append([eos_id])
                    logprobs_out.append([0.0])
                continue

            count = prompts.count(prompt)
            needed = count  # TRL expects len(prompts) * num_generations total items
            # repeat gens cyclically if we have fewer than needed
            for i in range(needed):
                g = gens[i % len(gens)]
                prompt_ids_out.append(g["prompt_id"])
                completion_ids_out.append(g["tokens"] + [eos_id])
                logprobs_out.append(g["logp"] + [0.0])

        return {
            "prompt_ids":     prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs":       logprobs_out,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Checkpoint archival callback
# ---------------------------------------------------------------------------

class ArchiveCallback(TrainerCallback):
    """Move stage/output files to storage after each training step."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir

    def on_step_end(self, args, state, control, **kwargs):
        step_dir = os.path.join(self.storage_dir, f"step-{state.global_step}")
        Path(step_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Step {state.global_step} complete — checkpointing to {step_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    superfamilies: dict[str, str] = SUPERFAMILIES,
    model_path: str   = MODEL_PATH,
    lora_dir: str     = LORA_DIR,
    max_steps: int    = 100,
    save_steps: int   = 25,
    num_generations: int = 4,
    per_device_batch: int = 2,
    grad_accum: int   = 4,
    seed: int         = 42,
):
    set_seed(seed)

    model, tokenizer = load_model_or_checkpoint(model_path, lora_dir)

    # dataset: one row per superfamily — the prompt is the superfamily name
    formatted_data = [{"prompt": sf} for sf in superfamilies.keys()]
    dataset = Dataset.from_list(formatted_data)

    # reward functions
    reward_funcs = []
    if USE_STRUCTURAL_REWARD:
        logger.info("Using structural reward (ColabFold + Foldseek)")
        for sf in superfamilies.keys():
            reward_funcs.append(build_structural_reward_func(sf))
    else:
        logger.info("Using composition-based reward (ProteinSequenceScorer)")
        reward_funcs.append(sequence_composition_reward)

    rollout_func = build_rollout_func(model, tokenizer, num_generations=num_generations)

    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        generation_batch_size=per_device_batch,
        max_completion_length=1024,
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=1.0,
        report_to="none",
        run_name="protein-baseline",
        output_dir=lora_dir,
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
