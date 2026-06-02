"""Protein generation GRPO baseline — no asyncflow / rhapsody / dragon.

Uses stock TRL GRPOTrainer with no custom rollout_func, so it works correctly
under ``accelerate launch`` for multi-GPU and multi-node training.

EPGF-style constrained generation is implemented as a HuggingFace
``LogitsProcessor``, which is called at every token step and is fully
DDP-compatible (each rank applies it independently to its local shard).

The constraints applied per-token are:
  1. Vocabulary masking  — only valid amino-acid tokens and the closing ">"
     can be sampled; everything else is set to -inf.
  2. Bio-score gating    — after each token, the partial sequence is scored
     with ProteinSequenceScorer.  If the score falls below ``bio_threshold``
     the processor forces an immediate ">" (sequence end), pruning the
     trajectory early rather than letting it continue into bad chemistry.
  3. Length enforcement  — once the generated portion exceeds ``max_seq_len``
     tokens the processor forces ">".

The one EPGF aspect that does NOT map to a LogitsProcessor is the
"generate N candidates, score all, pick best" selection step.  That is
replaced by GRPO's ``num_generations``: TRL generates multiple completions
per prompt and the policy gradient selects better sequences via the reward
signal rather than at generation time.

Two reward modes:
  1. Composition-only (default): ProteinSequenceScorer — no external tools.
  2. Structural (opt-in): ColabFold → Foldseek top-hit probability.
     Enable by setting COLABFOLD_PATH, FOLDSEEK_PATH, FOLDSEEK_DB env vars.
     In DDP each rank scores its own shard independently; only recommended
     for single-GPU or when dedicated folding resources are available.

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

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
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
# Model constants
# ---------------------------------------------------------------------------

MODEL_PATH = "GreatCaptainNemo/ProLLaMA"
LORA_DIR   = "prolora"

PROMPT_TEMPLATE = "[Generate by superfamily] Superfamily=<{superfamily}> Seq=<"

# EPGF constraint knobs
MAX_SEQ_LEN    = 500   # force ">" after this many generated tokens
BIO_THRESHOLD  = 0.55  # force ">" when partial sequence scores below this

# ---------------------------------------------------------------------------
# EPGF LogitsProcessor
# ---------------------------------------------------------------------------

class EPGFLogitsProcessor(LogitsProcessor):
    """Per-token EPGF constraints as a HuggingFace LogitsProcessor.

    DDP-safe: called independently on each rank for its local batch shard.
    No inter-process communication is needed.

    Parameters
    ----------
    tokenizer :
        The model tokenizer, used to decode partial sequences and to look up
        the token IDs for amino acids and ">".
    prompt_length : int
        Number of tokens in the prompt (input_ids up to this index are the
        prompt; everything after is the generated sequence so far).
    max_seq_len : int
        Force sequence termination after this many generated tokens.
    bio_threshold : float
        Force sequence termination if ``ProteinSequenceScorer`` scores the
        current partial sequence below this value.
    """

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        max_seq_len: int = MAX_SEQ_LEN,
        bio_threshold: float = BIO_THRESHOLD,
    ):
        self.tokenizer     = tokenizer
        self.prompt_length = prompt_length
        self.max_seq_len   = max_seq_len
        self.bio_threshold = bio_threshold

        vocab_size = tokenizer.vocab_size

        # Precompute set of token IDs that are allowed during generation.
        # For Llama-2 tokeniser each single letter is its own token, but we
        # check encode() to be safe (some tokenisers prepend a space token).
        allowed: set[int] = set()
        for aa in self.AMINO_ACIDS:
            for tid in tokenizer.encode(aa, add_special_tokens=False):
                if tid < vocab_size:
                    allowed.add(tid)
        # The closing ">" signals end-of-sequence.
        for tid in tokenizer.encode(">", add_special_tokens=False):
            if tid < vocab_size:
                allowed.add(tid)

        self._allowed_ids = list(allowed)

        # Precompute the "only allow '>'" mask — reused when forcing termination.
        end_ids: set[int] = set()
        for tid in tokenizer.encode(">", add_special_tokens=False):
            if tid < vocab_size:
                end_ids.add(tid)
        self._end_ids = list(end_ids)

        # Build the base allowed mask once (shape: [vocab_size])
        self._allowed_mask = torch.full((vocab_size,), float("-inf"))
        for tid in self._allowed_ids:
            self._allowed_mask[tid] = 0.0

        self._end_mask = torch.full((vocab_size,), float("-inf"))
        for tid in self._end_ids:
            self._end_mask[tid] = 0.0

    def _partial_sequence(self, input_ids_row: torch.Tensor) -> str:
        """Decode the generated tokens (after the prompt) as a sequence string."""
        generated = input_ids_row[self.prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Strip any ">" that may have been generated already
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", text.upper())

    def _should_terminate(self, partial_seq: str, gen_len: int) -> bool:
        """Return True if the current trajectory should be forced to end."""
        if gen_len >= self.max_seq_len:
            return True
        if not partial_seq:
            return False
        try:
            score = ProteinSequenceScorer(partial_seq).get_comprehensive_score()
            return score < self.bio_threshold
        except Exception:
            return False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        out = scores.clone()

        for i in range(batch_size):
            gen_len = input_ids.shape[1] - self.prompt_length
            partial  = self._partial_sequence(input_ids[i])

            if self._should_terminate(partial, gen_len):
                # Force ">" — end the sequence now
                out[i] = self._end_mask.to(scores.device)
            else:
                # Allow only valid amino-acid tokens and ">"
                out[i] = out[i] + self._allowed_mask.to(scores.device)

        return out


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

    ``device_map=None`` so accelerate controls device placement for DDP.
    """
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
# Inject EPGFLogitsProcessor into model.generate()
# ---------------------------------------------------------------------------

def patch_model_generate(model, tokenizer, prompt_length: int):
    """Monkey-patch model.generate() to always apply EPGFLogitsProcessor.

    This is transparent to GRPOTrainer — it calls model.generate() as usual
    and the constraint is applied automatically.  Works in DDP because each
    rank patches its own local model instance.
    """
    processor = EPGFLogitsProcessor(tokenizer, prompt_length=prompt_length)
    original_generate = model.generate

    def constrained_generate(*args, **kwargs):
        lp = kwargs.pop("logits_processor", None)
        if lp is None:
            lp = LogitsProcessorList()
        elif not isinstance(lp, LogitsProcessorList):
            lp = LogitsProcessorList(list(lp))
        lp.append(processor)
        kwargs["logits_processor"] = lp
        return original_generate(*args, **kwargs)

    model.generate = constrained_generate
    return model


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

    # Compute the prompt token length so EPGFLogitsProcessor knows where
    # the generated portion starts.  All prompts share the same template
    # structure; we measure on the longest one.
    sample_prompt = max(
        (PROMPT_TEMPLATE.format(superfamily=sf) for sf in SUPERFAMILIES),
        key=len,
    )
    prompt_length = len(tokenizer(sample_prompt, add_special_tokens=True).input_ids)

    # Patch model.generate() to apply EPGF constraints on every call.
    # GRPOTrainer calls model.generate() internally and will pick this up
    # automatically, on every rank, without any changes to the trainer.
    patch_model_generate(model, tokenizer, prompt_length=prompt_length)

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
