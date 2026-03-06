#!/usr/bin/env python3
"""
SFT protein model
Usage: python sft.py --data sequences_rewards.json --model <model_name> --output ./sft_output
"""
import os, re
os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"
import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from trl import SFTTrainer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(data_file):
    """Load sequences and keep only the top fraction by reward score."""
    with open(data_file) as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} sequences")

    # Sort by reward descending
    data.sort(key=lambda x: x['reward'], reverse=True)
    return data


def format_sample(sample, prompt_template):
    """Reconstruct the full prompt+sequence string for SFT."""
    prompt = prompt_template.format(superfamily=sample['superfamily'])
    return {'text': f"{prompt}{sample['sequence']}>"}

def load_llama_or_latest_checkpoint(
    base_model_id: str,
    lora_id: str,
    dtype=torch.bfloat16,
    device_map="auto",
):
    """
    Load base model and optionally attach a local LoRA adapter (directory).
    If `lora_id` is a directory containing a checkpoint, attach it via
    `PeftModel.from_pretrained`. Otherwise, if `args.apply_lora` is set,
    construct a `LoraConfig` and wrap the base model with `get_peft_model`.

    Returns: model, tokenizer, loaded_from, iteration
    """
    last_checkpoint = None

    if lora_id and os.path.isdir(lora_id):
        last_checkpoint = True

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    iteration = 0
    if last_checkpoint is not None:
        logger.info(f"Found LoRA checkpoint")
        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        # Attach LoRA adapter weights
        logger.info("Applying LoRA adapter from checkpoint...")
        model = PeftModel.from_pretrained(base_model, lora_id, is_trainable=True)
        loaded_from = last_checkpoint
    else:
        logger.info(f"No checkpoint found, loading base model: {base_model_id}")
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device_map,
        )
        model = get_peft_model(model, lora_config)
        loaded_from = base_model_id

    return model, tokenizer, loaded_from
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',           required=True, help='JSONL file with sequence and reward fields')
    parser.add_argument('--epochs',         type=int,   default=3)
    parser.add_argument('--lr',             type=float, default=2e-5)
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--grad_accum',     type=int,   default=4)
    args = parser.parse_args()

    # Load and filter
    data = load_data(args.data)

    # Format for SFT — reconstruct full prompt+completion string
    prompt_template = '[Generate by superfamily] Superfamily=<{superfamily}> Seq=<'
    formatted = [format_sample(s, prompt_template) for s in data]
    dataset = Dataset.from_list(formatted)

    print(f"\nDataset sample:\n  {formatted[0]['text'][:120]}...")

    # Load model and tokenizer
    model_path = "GreatCaptainNemo/ProLLaMA"
    lora_id = "prolora"
    model, tokenizer, loaded_from = load_llama_or_latest_checkpoint(
        model_path,
        lora_id,
    )

    training_args = TrainingArguments(
        output_dir=lora_id,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='no',
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        #tokenizer=tokenizer,
        #dataset_text_field='text',
        #max_seq_length=args.max_seq_length,
    )

    print("\nStarting SFT training...")
    trainer.train()
    trainer.save_model(lora_id)
    tokenizer.save_pretrained(lora_id)
    print(f"Model saved to {lora_id}")


if __name__ == '__main__':
    main()