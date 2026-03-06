import logging
logging.basicConfig(
    filename='romeupdate.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("romeupdate")
logger.info("update started")
run_name="roserun"
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('hf_token')
    userdata.get('hf')
except:
    os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"
import re
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64
lora_alpha = 64
target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

def load_model_or_latest_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map="auto",
):
    """
    If `output_dir` contains checkpoints for this run, load the latest one.
    Otherwise, load the base model from Hugging Face Hub.
    """

    last_checkpoint = None
    iteration = 0

    # If a LoRA output dir is provided, prefer LoRA checkpoints in that directory
    if lora_id and os.path.isdir(lora_id):
        last_checkpoint = get_last_checkpoint(lora_id)

    # If no LoRA checkpoints found and lora_id is falsy, check whether the
    # base model id refers to a local checkpoint directory containing
    # checkpoints (e.g., when training saved full-model checkpoints).
    if last_checkpoint is None and (not lora_id) and os.path.isdir(base_model_id):
        last_checkpoint = get_last_checkpoint(base_model_id)

    # If we found a local checkpoint (either LoRA or full-model), load tokenizer
    # and model from that checkpoint directory where appropriate.
    if last_checkpoint is not None:
        logger.info(f"Found checkpoint at: {last_checkpoint}")
        # Try loading tokenizer from checkpoint first, fall back to base_model_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(last_checkpoint, padding_side="left", use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Determine whether this checkpoint is a LoRA adapter checkpoint or a full model
        # Heuristic: presence of 'adapter_config.json' indicates a PEFT/LoRA checkpoint
        is_peft = os.path.exists(os.path.join(last_checkpoint, "adapter_config.json"))

        if is_peft:
            logger.info("Loading base model and applying LoRA adapter from checkpoint")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                dtype=dtype,
                device_map=device_map,
            )
            m = re.search(r"checkpoint-(\d+)", last_checkpoint)
            if m:
                iteration = int(m.group(1))
            model = PeftModel.from_pretrained(base_model, last_checkpoint, is_trainable=True)
            loaded_from = last_checkpoint
        else:
            logger.info("Loading full model from checkpoint directory")
            model = AutoModelForCausalLM.from_pretrained(
                last_checkpoint,
                dtype=dtype,
                device_map=device_map,
            )
            m = re.search(r"checkpoint-(\d+)", last_checkpoint)
            if m:
                iteration = int(m.group(1))
            loaded_from = last_checkpoint

        return model, tokenizer, loaded_from, iteration

    # No checkpoints found; load tokenizer from base_model_id and either attach LoRA
    # wrapper (if lora_id provided) or just load base model.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if lora_id:
        # No LoRA checkpoint found: create a new LoRA wrapper around the base model
        logger.info(f"No LoRA checkpoint found in {lora_id}; creating LoRA wrapper on base model {base_model_id}")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        model = get_peft_model(model, lora_config)
        loaded_from = base_model_id
    else:
        # lora_id was false and no checkpoints found — load the base model from hub
        logger.info(f"No checkpoints found; loading base model from hub: {base_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        loaded_from = base_model_id

    return model, tokenizer, loaded_from, iteration

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_lora = "32llamalora"

model, tokenizer, _, iteration = load_model_or_latest_checkpoint(model_name, model_lora)


from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split = "train")
def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})

import re

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores
match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores

max_prompt_length = 287 + 1 # + 1 just in case!

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 50,
    max_grad_norm = 1.0,
    report_to = "trackio", # Can use Weights & Biases
    run_name=f"llama32mathlora-{iteration}",
    output_dir = model_lora,
    overwrite_output_dir=True,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()