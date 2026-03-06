import logging
logging.basicConfig(
    filename='romereward.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("romeupdate")
logger.info("reward started")
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
dataset = load_dataset("openai/gsm8k", "main", split = "test")
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
    "ground_truth": extract_hash_answer(x["answer"]),
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
        response = completion
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion
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
    responses =completions

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
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]

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

batch_size=8
num_questions=16
iteration=1

shuffled_dataset = dataset.shuffle(seed=42)
test_questions = shuffled_dataset.select(range(min(num_questions, len(shuffled_dataset))))

score_sum = 0
logger.info(f"Starting evaluation on {len(test_questions)} questions...")

for q_idx, question in enumerate(test_questions):
    messages = [question["prompt"]] * batch_size
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs.shape[1]
    #print("tokenized")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_seq_length,
            do_sample=True,      # sampling
            top_p=0.95,
            temperature=0.8,
            pad_token_id = tokenizer.eos_token_id
            # don't usually mix beam search + sampling;
            # if you want beam search, drop top_p/temperature and set num_beams>1
        )
        
    #print("generated")
    #texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #generated_tokens = outputs[:, input_length:]
    #texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    truncated = outputs[:,input_length:]
    responses = tokenizer.batch_decode(truncated, skip_special_tokens=True)
    ground_truths = [question["ground_truth"]] * batch_size
    #PCA
    mfe_reward = match_format_exactly(responses)
    mfa_reward = match_format_approximately(responses)
    ca_reward = check_answer(messages, responses, ground_truths)
    cn_reward = check_numbers(messages, responses, ground_truths)
    
    logger.info("=" * 80)
    logger.info(f"Question {q_idx + 1}")
    logger.info("FULL PROMPT:")
    logger.info(f"System: {question['prompt'][0]['content']}")
    logger.info(f"User: {question['prompt'][1]['content']}")
    logger.info(f"Ground truth: {question['ground_truth']}")
    logger.info("=" * 80)
    for resp_idx, (response, mfe, mfa, ca, cn) in enumerate(zip(responses, mfe_reward, mfa_reward, ca_reward, cn_reward)):
        logger.info(f"Response {resp_idx + 1} | Total reward: {mfe+mfa+ca+cn} | MFA: {mfa} | MFE: {mfe} | CA: {ca} | CN: {cn} |")
        logger.info(response)
        logger.info("-" * 80)
    total_rewards = mfe_reward+mfa_reward+ca_reward+cn_reward
    total = sum(total_rewards)/batch_size
    score_sum += total
    logger.info(f"Progress: {q_idx + 1}/{len(test_questions)} questions, question_reward={total}")

logger.info("Finished grading...")
logger.info(f"Test reward {score_sum} over {len(test_questions)} questions")
print(score_sum)