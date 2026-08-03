import asyncio
import os
import sys

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend
from rose.metrics import GREATER_THAN_THRESHOLD

from multiprocessing import mp
from peft import LoraConfig
from rome.config import ModelConfig
from rome.train.grpo import GRPO
import re

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

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

async def rome_flow():
    mp.set_start_method("dragon")
    backend = await DragonExecutionBackendV3()
    flow = await WorkflowEngine.create(backend=backend)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=256,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    generation_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=1,
        repetition_penalty=1,
    )
    model_config = ModelConfig(
        base_model_name = "meta-llama/Llama-2-7b-hf",
        model_name = None,
        lora_name = "model_lora",
        lora_config = lora_config,
        generation_config = generation_config,
    )

    grpo_trainer = GRPO(
        gpus=1,
        rewrad_funcs=[
            match_format_approximately, 
            match_format_exactly, 
            check_answer, 
            check_numbers
        ],
        grpo_config = GRPOConfig(
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
            max_steps = 1,
            save_steps = 50,
            max_grad_norm = 1.0,
            report_to = "trackio", # Can use Weights & Biases
            run_name=f"testrun",
            output_dir = "model_lora",
        )
    )

    flow_config = SequentialFlowConfig(
        iterations = 2,
        reward_threshold = 10,
        operator = GREATER_THAN_THRESHOLD,
        num_generators = 2,
        num_scorers = 1,
    )
    seq_flow = SequentialFlow(
        model_config=model_config,
        trainer = grpo_trainer,
        evaluate_func = eval_func,
        asyncflow = flow,
        
    )

    await flow.shutdown()

if __name__ == "__main__":
    asyncio.run(rome_flow())