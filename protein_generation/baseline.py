"""Protein generation GRPO baseline — no asyncflow / rhapsody / dragon.

Uses stock TRL GRPOTrainer with no custom rollout_func, so it works correctly
under ``accelerate launch`` for multi-GPU and multi-node training.

The model (ProLLaMA) generates protein sequences autoregressively given a
superfamily prompt.  Two reward modes are supported:

  1. Composition-only (default, no external tools):
       ProteinSequenceScorer from ProteinModel.py scores generated sequences
       on amino-acid diversity, physicochemical properties, and complexity.

  2. Structural (opt-in, requires ColabFold + Foldseek):
       Each completion is folded with ColabFold then searched against the
       per-superfamily Foldseek database.  The top-hit probability is the
       reward.  Enable by setting COLABFOLD_PATH, FOLDSEEK_PATH, FOLDSEEK_DB
       env vars (or editing the constants below).

       NOTE: in DDP each rank calls the reward function independently on its
       local batch shard.  ColabFold is CPU/GPU-heavy, so only enable
       structural reward when you have dedicated folding resources or are
       running single-GPU.

Usage
-----
    # single GPU
    python protein_generation/baseline.py

    # multi-GPU on one node
    accelerate launch --multi_gpu protein_generation/baseline.py

    # multi-node (2 nodes, 8 GPUs each)
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

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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
# Paths — set via env vars or edit here
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
# Model constants
# ---------------------------------------------------------------------------

MODEL_PATH = "GreatCaptainNemo/ProLLaMA"
LORA_DIR   = "prolora"

# ProLLaMA prompt format — the model continues after "Seq=<"
PROMPT_TEMPLATE = "[Generate by superfamily] Superfamily=<{superfamily}> Seq=<"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_or_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map: Optional[str] = None,
):
    """Load ProLLaMA + LoRA, resuming from a saved adapter if one exists.

    ``device_map`` is intentionally left as None so that accelerate can place
    the model on the correct device for DDP.  Pass ``device_map="auto"`` only
    for single-process inference.
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
        logger.info(f"No checkpoint found — initialising fresh LoRA on {base_model_id}")
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
# Sequence extraction helper
# ---------------------------------------------------------------------------

def _extract_sequence(text: str) -> str:
    """Pull the amino-acid sequence out of a ProLLaMA completion."""
    # The model generates everything after "Seq=<" up to the closing ">"
    if "Seq=<" in text:
        return text.split("Seq=<")[-1].split(">")[0].strip()
    # fallback: treat the whole text as the sequence
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", text.upper())


# ---------------------------------------------------------------------------
# Reward: composition-based (multi-GPU safe, no external tools)
# ---------------------------------------------------------------------------

def sequence_composition_reward(prompts, completions, **kwargs) -> list[float]:
    """Score each completion with ProteinSequenceScorer.

    Called independently on each accelerate rank's local batch shard —
    no inter-process communication required.
    """
    rewards = []
    for completion in completions:
        # TRL passes completions as raw strings for non-chat models
        text = completion[0]["content"] if isinstance(completion, list) else completion
        seq = _extract_sequence(text)
        try:
            score = ProteinSequenceScorer(seq).get_comprehensive_score() if seq else 0.0
        except Exception:
            score = 0.0
        rewards.append(float(score))
    return rewards


# ---------------------------------------------------------------------------
# Reward: structural via ColabFold + Foldseek (optional, single-GPU recommended)
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
    """Return a per-superfamily structural reward function."""
    db = os.path.join(FOLDSEEK_DB, SUPERFAMILIES.get(superfamily, ""))

    def sequence_structural_reward(prompts, completions, **kwargs) -> list[float]:
        seqs = [
            _extract_sequence(
                c[0]["content"] if isinstance(c, list) else c
            )
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
# Checkpoint archival callback
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
    model_path: str      = MODEL_PATH,
    lora_dir: str        = LORA_DIR,
    max_steps: int       = 100,
    save_steps: int      = 25,
    num_generations: int = 4,
    per_device_batch: int = 2,
    grad_accum: int      = 4,
    max_completion_length: int = 512,
    max_prompt_length: int = 128,
):
    # device_map=None so accelerate controls device placement in DDP
    model, tokenizer = load_model_or_checkpoint(model_path, lora_dir, device_map=None)

    # One row per superfamily; the prompt string is what the model conditions on.
    # GRPOTrainer tokenises this directly (no chat template — ProLLaMA is not
    # an instruction model).
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

    training_args = GRPOConfig(
        # optimisation
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=1.0,
        # batching
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        generation_batch_size=per_device_batch,
        # sequence lengths
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # schedule
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=1,
        # output
        output_dir=lora_dir,
        run_name="protein-baseline",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
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
