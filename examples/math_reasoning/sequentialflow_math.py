"""Math reasoning GRPO training via ROME SequentialFlow.

Adapts the separated reward.py / update.py scripts into the SequentialFlow
abstraction so that generation, scoring, and training are orchestrated by ROME
rather than run in a single monolithic script.

Usage
-----
    python examples/math_reasoning/sequentialflow_math.py
"""

import asyncio
import logging
import os
import re

logging.basicConfig(
    filename="rome_math_sequentialflow.log",
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
)
logger = logging.getLogger("rome_math_sequentialflow")

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("hf_token")
except Exception:
    os.environ.setdefault("HF_TOKEN", "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    os.environ.setdefault("HF_HOME", "/work/nvme/bdyk/apark4/huggingface")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import GenerationConfig
from trl import GRPOConfig

from radical.asyncflow import WorkflowEngine

from rome.config import ModelConfig
from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig
from rome.train.grpo import GRPO
from rome.utils import load_model
from rome.workflow import Workflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_LORA = "32llamalora"

MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 288  # 287 + 1 just in case

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _build_dataset(split: str):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": _extract_hash_answer(x["answer"]),
    })
    return ds


# ---------------------------------------------------------------------------
# Reward functions  (TRL completion format: list of dicts with "content" key)
# ---------------------------------------------------------------------------

_match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

_match_numbers = re.compile(
    SOLUTION_START + r".*?([\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)


def _text(completion) -> str:
    """Extract plain text from a TRL completion (list-of-dicts or raw string)."""
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def match_format_exactly(completions, **kwargs):
    return [
        3.0 if _match_format.search(_text(c)) is not None else 0.0
        for c in completions
    ]


def match_format_approximately(completions, **kwargs):
    scores = []
    for c in completions:
        r = _text(c)
        score = 0.0
        score += 0.5 if r.count(REASONING_START) == 1 else -1.0
        score += 0.5 if r.count(REASONING_END) == 1 else -1.0
        score += 0.5 if r.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if r.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = [_text(c) for c in completions]
    extracted = [
        m.group(1) if (m := _match_format.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        if guess == true_answer:
            scores.append(3.0)
        elif guess.strip() == true_answer.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    scores.append(1.0)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.5)
                else:
                    scores.append(-1.5)
            except Exception:
                scores.append(-1.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    responses = [_text(c) for c in completions]
    extracted = [
        m.group(1) if (m := _match_numbers.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            scores.append(
                1.5 if float(guess.replace(",", "")) == float(true_answer.strip()) else -0.5
            )
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Evaluate function  (runs on the test split each iteration)
# ---------------------------------------------------------------------------

def build_evaluate_func(
    model_config: ModelConfig,
    num_questions: int = 16,
    batch_size: int = 8,
):
    """Return an evaluate function that scores the model on the GSM8K test set."""

    test_ds = _build_dataset("test").shuffle(seed=42).select(range(num_questions))
    test_ds = test_ds.rename_column("answer", "ground_truth")

    async def evaluate(mc: ModelConfig) -> float:
        model, tokenizer = load_model(mc)
        model.eval()

        generation_config = mc.generation_config or GenerationConfig(
            max_new_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

        score_sum = 0.0
        for q_idx, question in enumerate(test_ds):
            messages = [question["prompt"]] * batch_size
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            input_length = inputs.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )

            responses = tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )
            ground_truths = [question["ground_truth"]] * batch_size

            rewards = (
                match_format_exactly(responses)
                + match_format_approximately(responses)
                + check_answer(messages, responses, ground_truths)
                + check_numbers(messages, responses, ground_truths)
            )
            question_reward = sum(rewards) / batch_size
            score_sum += question_reward
            logger.info(
                f"Eval question {q_idx + 1}/{num_questions}: reward={question_reward:.3f}"
            )

        logger.info(f"Eval total reward: {score_sum:.3f} over {num_questions} questions")
        return score_sum

    return evaluate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    generation_cfg = GenerationConfig(
        max_new_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
    )

    model_config = ModelConfig(
        base_model_name=MODEL_NAME,
        lora_name=MODEL_LORA,
        lora_config=lora_cfg,
        generation_config=generation_cfg,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    grpo_cfg = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        max_steps=100,
        save_steps=50,
        max_grad_norm=1.0,
        output_dir=MODEL_LORA,
        overwrite_output_dir=True,
    )

    trainer = GRPO(
        gpus=1,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        grpo_config=grpo_cfg,
    )

    flow_config = SequentialFlowConfig(
        iterations=10,
        num_generators=2,
        num_scorers=2,
        batch_size=8,
    )

    evaluate_func = build_evaluate_func(model_config)

    asyncflow = WorkflowEngine()

    flow = SequentialFlow(
        model_config=model_config,
        trainer=trainer,
        evaluate_func=evaluate_func,
        asyncflow=asyncflow,
        flow_config=flow_config,
    )

    train_dataset = _build_dataset("train")
    await flow.launch(dataset=train_dataset)


if __name__ == "__main__":
    asyncio.run(main())
