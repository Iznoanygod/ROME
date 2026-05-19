import argparse, asyncio, yaml, random, logging, time, os, itertools, sys

from typing import Union

#from concurrent.futures import ThreadPoolExecutor
#from radical.asyncflow import LocalExecutionBackend
from rhapsody.backends import DragonExecutionBackendV3

from dragon.infrastructure.policy import Policy
from dragon.native.event import Event
from dragon.native.machine import System, Node
from dragon.data.ddict import DDict

import rhapsody
from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger
logger = logging.getLogger(__name__)

import multiprocessing as mp

os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"
foldseek_path = "/work/hdd/bdyk/apark4/foldseek/bin/foldseek"
colabfold_path = "/work/nvme/bdyk/apark4/localcolabfold/.pixi/envs/default/bin/colabfold_batch"
runfold_path = "/work/nvme/bdyk/apark4/ROME/protein_generation/run_fold.sh"
runsingularity_path = "/work/nvme/bdyk/apark4/ROME/protein_generation/run_singularity.sh"
tmp_dir = "/tmp"
colabfold_sif_path = "/work/nvme/bdyk/apark4/ROME/colabfold_1.6.0-cuda12.sif"
fold_cache = "/work/hdd/bdyk/apark4/foldcache"
foldseekdb_path = "/work/hdd/bdyk/apark4/foldseek/afdb50"
storage_path = "/work/hdd/bdyk/apark4/ROME/storagenew"

generation_task_batch_size = 4
prompt_gen_batch_size = 2

def eprint(str):
    sys.stderr.write(str + "\n")
    sys.stderr.flush()

superfamilies = {
    "CheY-like superfamily": "chey",
    "Tetratricopeptide-like helical domain superfamily": "tphd",
    "S-adenosyl-L-methionine-dependent methyltransferase superfamily": "sammt",
    "Thioredoxin-like superfamily": "trx",

    "Ankyrin repeat-containing domain superfamily": "ankr",
    "Immunoglobulin-like domain superfamily": "igld",
    "Nucleotide-binding alpha-beta plait domain superfamily": "nabp",
    "RmlC-like cupin domain superfamily": "rlcd",

    "Leucine-rich repeat domain superfamily": "lrrd",
    "Glycoside hydrolase superfamily": "gchl",
    "WD40-repeat-containing domain superfamily": "wd40",
    "Papain-like cysteine peptidase superfamily": "plcd",

    "Protein kinase-like domain superfamily": "pkld",
    "HAD-like superfamily": "hadl",
    "Winged helix-like DNA-binding domain superfamily": "whdb",
    "Galactose-binding-like domain superfamily": "gbld"
}

def find_gpus():
    nodes_with_gpu = []
    # loop through all nodes Dragon is running on
    for huid in System().nodes:
        node = Node(huid)
        # loop through however many GPUs it may have
        gpu_list = []
        for gpu_id in node.gpus:
            gpu_list.append(gpu_id)
        nodes_with_gpu.append((node.hostname, gpu_list))
    return nodes_with_gpu

async def main():
    init_default_logger(logging.INFO)
    rhapsody.enable_logging(level=logging.DEBUG)
    all_gpus = find_gpus()
    trainer_task_gpu = all_gpus[0]  # assign the whole first node to trainer task
    remaining_nodes_gpus = all_gpus[1:]  # use the rest of the nodes for generation and folding

    class RoundRobinGPU:
        def __init__(self, node_gpu_list):
            self.gpu_list = []
            #assuming all nodes have same gpu count
            # try to round robin by node?
            for node, gpus in node_gpu_list:
                for gpu in gpus:
                    self.gpu_list.append((node, [gpu]))
            self._iterator = itertools.cycle(self.gpu_list)
        def next(self):
            return next(self._iterator)

    gpu_allocator = RoundRobinGPU(remaining_nodes_gpus)

    trainer_task_policy = Policy(
        placement=Policy.Placement.HOST_NAME,
        host_name=trainer_task_gpu[0],
        gpu_affinity=[3],  # assign all GPUs on the trainer node to the trainer task
    )
    nodes = 1
    mp.set_start_method("dragon")
    #print(len(os.sched_getaffinity(0)))
    backend = await DragonExecutionBackendV3(
        #num_workers=nodes * mp.cpu_count(),
        #disable_background_batching=False,
    )
    #backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)

    alloc = System()
    num_nodes = int(alloc.nnodes)
    dict_mem_per_node = 1 * 1024**3 

    # Input superfamily, fills gen_fam_ddict with family and list of sequences
    # gen_fam_ddict[superfamily].append({"prompt_id": original_prompt_id, "tokens":tokens, "logp":logp})
    @flow.function_task
    async def generate_seq_epgf(superfamily, num_seq=64, gen_fam_ddict=None, seq_ddict=None, seed=42):
        #print("test")
        import os, socket, sys
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        import logging
        logger = logging.getLogger(__name__)
        import re
        import math
        import random
        import numpy as np
        import torch
        from ProteinModel import ProteinSequenceScorer
        from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
        from tqdm import tqdm
        import multiprocessing as mp
        #print(torch.cuda.device_count())
        #if gen_fam_ddict is not None:
            #("DDict passed to generate")
        eprint(f"generate_seq_epgf {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        def set_seed(seed=seed) -> None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            random.seed(seed)
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
            from pathlib import Path
            if (lora_id is not None) and os.path.isdir(lora_id) and any(Path(lora_id).iterdir()):
                last_checkpoint = True
        
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            iteration = 0
            if last_checkpoint is not None:
                eprint(f"Found LoRA checkpoint")
                eprint(f"Loading base model: {base_model_id}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    device_map=device_map,
                )
                # Attach LoRA adapter weights
                eprint("Applying LoRA adapter from checkpoint...")
                model = PeftModel.from_pretrained(base_model, lora_id, is_trainable=True)
                loaded_from = last_checkpoint
            else:
                eprint(f"No checkpoint found, loading base model: {base_model_id}")
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
        
        model_path = "GreatCaptainNemo/ProLLaMA"
        lora_id = "prolora"
        model, tokenizer, loaded_from = load_llama_or_latest_checkpoint(
            model_path,
            lora_id,
        )
        generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=1,
            num_return_sequences=8,
            repetition_penalty=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        def softmax(x, temperature=1.0):
            x = np.array(x) / temperature
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
    
        initial_temperature = 1.0
        final_temperature = 0.001
        decay_rate = 0.1
        
        # maximum allowed sequence length (chars / tokens)
        MAX_SEQ_LEN = 500
        
        model.eval()
        
        seq_with_scores = []
        set_seed()
        with torch.no_grad():
            i = 0
            #gennerate num_seq protein sequences
            eprint("good sign")
            while i < num_seq:
                temperature = initial_temperature
                failed = False
                prompt = f'[Generate by superfamily] Superfamily=<{superfamily}> Seq=<'  #you can modify this prompt
                original_prompt_id = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
                logp = []
                tokens = []
                while True:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
        
                    candidates = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                    log_probs = transition_scores.sum(dim=1).cpu().numpy().tolist() # log prob used as score
        
                    candidates_with_scores = [(candidates[idx], outputs.sequences[idx], log_probs[idx]) for idx in range(len(candidates))]
                    candidates_with_scores.sort(key=lambda x: x[2], reverse=True)
        
                    top_half_count = max(1, len(candidates) // 2)
                    top_candidates = candidates_with_scores[:top_half_count]
        
                    for cand, tok, score in candidates_with_scores:
                        if cand.strip().endswith('>') and not any(c == cand and s == score for c, _, s in top_candidates):
                            top_candidates.append((cand, tok, score))
        
                    bio_scores = []
                    filtered_candidates = []
                    filtered_cand_logp  = []
                    filtered_cand_tokens= []
        
                    for cand, token, lm_score in top_candidates:
                        seq = cand.split('Seq=<')[-1].split('>')[0].strip()
                        # skip sequences longer than MAX_SEQ_LEN characters or tokens
                        if len(token) > MAX_SEQ_LEN:
                            continue
                        scorer = ProteinSequenceScorer(seq)
                        bio_score = scorer.get_comprehensive_score()
        
                        if bio_score < 0.55:
                            continue
                        bio_scores.append(bio_score)
                        filtered_candidates.append(cand)
                        filtered_cand_logp.append(lm_score)
                        filtered_cand_tokens.append(token)
                    if len(filtered_candidates) == 0:
                        failed = True
                        break
        
                    bio_scores_softmax = softmax(bio_scores, temperature=temperature)
                    temperature = max(final_temperature, temperature * decay_rate)
                    sampled_idx = np.random.choice(len(filtered_candidates), p=bio_scores_softmax)
                    winner = filtered_candidates[sampled_idx]
                    winner_logp = filtered_cand_logp[sampled_idx]
                    winner_token = filtered_cand_tokens[sampled_idx][-1]
                    if winner.endswith('>'):
                        logp.append(winner_logp)
                        tokens.append(winner_token.item())
                        break
                    else:
                        prompt = winner
                        logp.append(winner_logp)
                        tokens.append(winner_token.item())
        
                if not failed:
                    sequence = winner.split('Seq=<')[-1].split('>')[0].strip()
                    seq_with_scores.append({"prompt_id": original_prompt_id, "sequence":sequence, "tokens":tokens, "logp":logp, "superfamily": superfamily})
                    if gen_fam_ddict is not None:
                        #print("adding to ddict")
                        #gen_fam_ddict[sequence] = {"prompt_id": original_prompt_id, "tokens":tokens, "logp":logp, "superfamily": superfamily}
                        if superfamily not in gen_fam_ddict:
                            gen_fam_ddict[superfamily] = []
                        seq_list = gen_fam_ddict[superfamily]
                        seq_list.append({"sequence": sequence, "prompt_id": original_prompt_id, "tokens":tokens, "logp":logp})
                        gen_fam_ddict[superfamily] = seq_list
                    if seq_ddict is not None:
                        seq_ddict[sequence] = {"prompt_id": original_prompt_id, "tokens":tokens, "logp":logp, "superfamily": superfamily}
                    eprint(f"Generated sequence {sequence}")
                    i += 1
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        eprint(f"finished generating {superfamily}")
        
        return seq_with_scores
    
    # Input generated_family ddict with list of sequences, outputs list of folded struct files, fill ddict with family with list of sequences
    @flow.function_task
    async def fold_sequences(generated_families_ddict, superfamily, staging_dir, output_dir, finished_family_ddict, seed=None, use_singularity=True):
        import os, subprocess, re, shutil, socket, sys, uuid
        file_ruuid = str(uuid.uuid4())
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        import logging
        logger = logging.getLogger(__name__)
        from pathlib import Path
        if seed is None:
            import random
            seed = random.randint(0, 65535)
        eprint(str(len(os.sched_getaffinity(0))))
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        eprint(f"fold_sequences {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        output_path = Path(output_dir)
        staging_path = Path(staging_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir(parents=True, exist_ok=True)
        superfamily_sequences = generated_families_ddict[superfamily]
        input_file = staging_path / "sequences.fasta"
        with open(input_file, 'w') as f:
            for i, seq_info in enumerate(superfamily_sequences):
                sequence = seq_info["sequence"]
                f.write(f'>{i}\n')
                f.write(sequence + '\n')
        if not use_singularity:
            
            eprint("calling runfold")
            colabfold_result = subprocess.run([
                runfold_path,
                input_file,
                staging_dir
            ], capture_output=True, text=True)
            eprint(colabfold_result.stdout)
        else:
            tmp_stage_path = Path(f"/tmp/{file_ruuid}")
            try:
                tmp_stage_path.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                eprint("tmp staging directory already exists, clearing it out")
                shutil.rmtree(tmp_stage_path)
                tmp_stage_path.mkdir(parents=True, exist_ok=False)
            input_file = tmp_stage_path / "sequences.fasta"
            with open(input_file, 'w') as f:
                for i, seq_info in enumerate(superfamily_sequences):
                    sequence = seq_info["sequence"]
                    f.write(f'>{i}\n')
                    f.write(sequence + '\n')
            eprint("calling singularity")
            command = [
                runsingularity_path,
                colabfold_sif_path,
                "sequences.fasta",
                f"{tmp_stage_path}",
                f"{fold_cache}",
                f"{seed}"
            ]
            #eprint(str(command))
            singularity_result = subprocess.run(command, capture_output=True, text=True)
            eprint(singularity_result.stdout)

        #staging_path = Path(staging_dir)
        #staging_path.mkdir(parents=True, exist_ok=True)
    
        pattern = re.compile(r'^(.+)_unrelaxed_rank_001_alphafold2_ptm_model_\d+_seed_000\.pdb$')
        eprint("copying pdb files")
        if not use_singularity:
            for pdb_file in staging_path.glob('*.pdb'):
                if pattern.match(pdb_file.name):
                    shutil.copy(pdb_file, output_path / pdb_file.name)
        else:
            tmp_path = Path(f"/tmp/{file_ruuid}")
            for pdb_file in tmp_path.glob('*.pdb'):
                if pattern.match(pdb_file.name):
                    shutil.copy(pdb_file, output_path / pdb_file.name)
        finished_family_ddict[superfamily] = len(superfamily_sequences)
        eprint(f"finished folding {superfamily}")
        return superfamily
    
    @flow.executable_task
    async def singularity_task(generated_families_ddict, superfamily, staging_dir, output_dir, finished_family_ddict, seed=None):
        import os, subprocess, re, shutil, socket, sys
        
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        import logging
        logger = logging.getLogger(__name__)
        from pathlib import Path
        if seed is None:
            import random
            seed = random.randint(0, 65535)
        eprint(str(len(os.sched_getaffinity(0))))
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        #eprint(f"fold_sequences {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        output_path = Path(output_dir)
        staging_path = Path(staging_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir(parents=True, exist_ok=True)
        superfamily_sequences = generated_families_ddict[superfamily]
    
        input_file = staging_path / "sequences.fasta"
        with open(input_file, 'w') as f:
            for i, seq_info in enumerate(superfamily_sequences):
                sequence = seq_info["sequence"]
                f.write(f'>{i}\n')
                f.write(sequence + '\n')
        #eprint(runsingularity_path + " " + colabfold_sif_path + " sequences.fasta " + staging_dir)
        return runsingularity_path + " " + colabfold_sif_path + " sequences.fasta " + staging_dir
    
    @flow.function_task
    async def post_sing_task(staging_dir, output_dir):
        import os, subprocess, re, shutil, socket, sys
        pattern = re.compile(r'^(.+)_unrelaxed_rank_001_alphafold2_ptm_model_\d+_seed_000\.pdb$')
        eprint("copying pdb files")
        output_path = Path(output_dir)
        staging_path = Path(staging_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir(parents=True, exist_ok=True)
        for pdb_file in staging_path.glob('*.pdb'):
            if pattern.match(pdb_file.name):
                shutil.copy(pdb_file, output_path / pdb_file.name)
        return "done"
    # Input superfamily, directory of folded structs, database, fills seq_ddict with foldseek scores for each sequence
    # seq_ddict[seq] = {
    #     "total_hits": len(matches),
    #     "top_target": top_target,
    #     "top_lddt":   top_lddt,
    #     "top_prob":   top_prob,
    #     "avg_lddt":   avg_lddt,
    #     "avg_prob":   avg_prob,
    # }

    @flow.function_task
    async def score_sequence(superfamily, pdb_input_dir, db, tmp_dir, output_file, scored_ddict, input_fasta_path, seq_ddict = None, format_output='query,target,lddt,prob'):
        # Perform foldseek search on sequences and return hprob scores
        import os, socket, sys, uuid
        file_ruuid = str(uuid.uuid4())
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        eprint(str(len(os.sched_getaffinity(0))))
        import logging
        logger = logging.getLogger(__name__)
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        eprint(f"score_sequence {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        import subprocess
        import csv
        from pathlib import Path
        import re
        if seq_ddict is not None:
            logger.info("DDict passed to fold")
        pdb_input_path = Path(pdb_input_dir)
        tmp_path       = Path(f"{fold_cache}/{file_ruuid}")
        output_path    = Path(output_file)
        tmp_path.mkdir(parents=True, exist_ok=True)
        #eprint(str(tmp_path))
        pdb_count = len(list(pdb_input_path.glob('*.pdb')))
        eprint("running foldseek")
        # Run foldseek
        cmd = [
            foldseek_path, 'easy-search',
            str(pdb_input_path),
            db,
            str(output_path),
            str(tmp_path),
            '--format-output', format_output,
        ]
        #eprint(str(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        eprint("foldseek finished")
        #if result.returncode != 0:
        #    raise RuntimeError(f"Foldseek failed:\n{result.stderr}")
        
        def load_fasta_sequences(fasta_path: str) -> list[str]:
            sequences = []
            current = []
            with open(fasta_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current:
                            sequences.append("".join(current))
                            current = []
                    else:
                        current.append(line)
                if current:
                    sequences.append("".join(current))
            return sequences
        
        # Parse foldseek output and extract hprob scores
        def parse_sequence_id(query_name):
            m = re.match(r'^(\d+)_unrelaxed_rank_001', query_name)
            return m.group(1) if m else query_name
    
        hits = {}
        with open(output_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                query, target, lddt, prob = parts[0], parts[1], float(parts[2]), float(parts[3])
                seq_id = str(parse_sequence_id(query))
                if seq_id not in hits:
                    hits[seq_id]=[]
                hits[seq_id].append((target, lddt, prob))
        for i in range(pdb_count):
            #first have to check if foldseek returned no hits for the sequence. In this case prob=-1, lddt=-1, target=None
            if str(i) not in hits:
                hits[str(i)] = [("None", -1, -1)]
        # create scored sequence list with top hit and average scores
        sequences = load_fasta_sequences(input_fasta_path)
        scored_sequences = []
        for seq_id, matches in sorted(hits.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            total = len(matches)
            top_target, top_lddt, top_hscore = matches[0]  # first hit is top match
            avg_lddt   = sum(m[1] for m in matches) / total
            avg_prob = sum(m[2] for m in matches) / total
            seq = sequences[int(seq_id)]
            scored_sequences.append({
                "sequence":   seq,
                "total_hits": total,
                "top_target": top_target,
                "top_lddt":   top_lddt,
                "top_prob":   top_hscore,
                "avg_lddt":   avg_lddt,
                "avg_prob":   avg_prob,
            })
            scored_ddict[seq] = {
                "total_hits": total,
                "top_target": top_target,
                "top_lddt":   top_lddt,
                "top_prob":   top_hscore,
                "avg_lddt":   avg_lddt,
                "avg_prob":   avg_prob,
            }
            eprint(f"scored sequence {seq} of {superfamily}")
            if seq_ddict is not None:
                # get existing ddict entry for the sequence
                existing_entry = {}
                if seq in seq_ddict:
                    existing_entry = seq_ddict[seq]
                # update the entry with new foldseek scores
                existing_entry.update({
                    "total_hits": len(matches),
                    "top_target": top_target,
                    "top_lddt":   top_lddt,
                    "top_prob":   top_hscore,
                    "avg_lddt":   avg_lddt,
                    "avg_prob":   avg_prob,
                })
                seq_ddict[seq] = existing_entry
        eprint(f"finished scoring {superfamily}")
        return scored_sequences

    trainer_process_template = {"process_template": {"policy": trainer_task_policy}}
    @flow.function_task
    async def grpo_trainer(superfamily_ddict, gen_fam_ddict, folded_ddict, scored_ddict, reset_event, task_description=trainer_process_template):
        import sys
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        #sys.stderr.write("test")
        import logging, os, re, socket
        # Force single-node single-process distributed setup
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        # Unset SLURM variables that confuse accelerate into thinking multinode
        os.environ.pop("SLURM_PROCID", None)
        os.environ.pop("SLURM_NODEID", None)
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_NPROCS", None)
        os.environ.pop("SLURM_STEP_NODELIST", None)
        os.environ.pop("SLURM_JOB_NODELIST", None)
        import numpy as np
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
        from transformers import TrainerCallback
        from datasets import Dataset
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
        from trl import GRPOConfig, GRPOTrainer
        from pathlib import Path
        import shutil
        logger = logging.getLogger(__name__)
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        logger.info(f"grpo_trainer with GPU {cuda_devices} on host {socket.gethostname()}")
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
            from pathlib import Path
            last_checkpoint = None
        
            if lora_id and os.path.isdir(lora_id) and any(Path(lora_id).iterdir()):
                last_checkpoint = True
        
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            iteration = 0
            if last_checkpoint is not None:
                eprint(f"Found LoRA checkpoint")
                eprint(f"Loading base model: {base_model_id}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    device_map=device_map,
                )
                # Attach LoRA adapter weights
                eprint("Applying LoRA adapter from checkpoint...")
                model = PeftModel.from_pretrained(base_model, lora_id, is_trainable=True)
                loaded_from = last_checkpoint
            else:
                eprint(f"No checkpoint found, loading base model: {base_model_id}")
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

        class GRPOCallback(TrainerCallback):
            # Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients
            def on_optimizer_step(self, args, state, control, **kwargs):
                pass

            def on_step_begin(self, args, state, control, **kwargs):
                pass

            def on_step_end(self, args, state, control, **kwargs):
                #set reset event to clear launchers
                eprint("on_step_end")
                
                reset_event.set()
                eprint("reset_event set")
                #need to move all files generated into storage
                for superfamily in superfamily_ddict.keys():
                    shorthand = superfamilies[superfamily]
                    superfamily_storage_path = f"{storage_path}/{shorthand}-{state.global_step}"
                    superfamily_stage_path = f"/work/nvme/bdyk/apark4/ROME/stage-{shorthand}"
                    superfamily_output_path = f"/work/nvme/bdyk/apark4/ROME/output-{shorthand}"
                    Path(superfamily_storage_path).mkdir(parents=True, exist_ok=True)
                    eprint(f"moving files for superfamily {superfamily} to storage")
                    #move contents of stage and output to storage, then clear stage and output for next round
                    for file in os.listdir(superfamily_stage_path):
                        shutil.move(os.path.join(superfamily_stage_path, file), os.path.join(superfamily_storage_path, file))
                    for file in os.listdir(superfamily_output_path):
                        shutil.move(os.path.join(superfamily_output_path, file), os.path.join(superfamily_storage_path, file))
                    eprint(f"finished moving files for superfamily {superfamily}")
                    shutil.rmtree(superfamily_stage_path)
                    shutil.rmtree(superfamily_output_path)
                
                #clear out the dictionarys
                eprint("clearing out dictionaries")
                superfamily_ddict.clear()
                gen_fam_ddict.clear()
                folded_ddict.clear()
                scored_ddict.clear()
                eprint("finished clearing out dictionaries")
                eprint("on_step_end done")
                pass

            def on_substep_end(self, args, state, control, **kwargs):
                #set reset event to clear launchers
                eprint("on_substep_end")
                
                reset_event.set()
                eprint("reset_event set")
                #need to move all files generated into storage
                for superfamily in superfamily_ddict.keys():
                    shorthand = superfamilies[superfamily]
                    superfamily_storage_path = f"{storage_path}/{shorthand}-{state.global_step}"
                    superfamily_stage_path = f"/work/nvme/bdyk/apark4/ROME/stage-{shorthand}"
                    superfamily_output_path = f"/work/nvme/bdyk/apark4/ROME/output-{shorthand}"
                    Path(superfamily_storage_path).mkdir(parents=True, exist_ok=True)
                    eprint(f"moving files for superfamily {superfamily} to storage")
                    #move contents of stage and output to storage, then clear stage and output for next round
                    for file in os.listdir(superfamily_stage_path):
                        shutil.move(os.path.join(superfamily_stage_path, file), os.path.join(superfamily_storage_path, file))
                    for file in os.listdir(superfamily_output_path):
                        shutil.move(os.path.join(superfamily_output_path, file), os.path.join(superfamily_storage_path, file))
                    eprint(f"finished moving files for superfamily {superfamily}")
                    shutil.rmtree(superfamily_stage_path)
                    shutil.rmtree(superfamily_output_path)
                
                #clear out the dictionarys
                eprint("clearing out dictionaries")
                superfamily_ddict.clear()
                gen_fam_ddict.clear()
                folded_ddict.clear()
                scored_ddict.clear()
                eprint("finished clearing out dictionaries")
                eprint("on_substep_end done")
                pass

            def on_train_begin(self, args, state, control, **kwargs):
                pass

            def on_train_end(self, args, state, control, **kwargs):
                pass

        max_prompt_length=2048
        max_seq_length=512
        model_path = "GreatCaptainNemo/ProLLaMA"
        lora_id = "prolora"
        model, tokenizer, loaded_from = load_llama_or_latest_checkpoint(
            model_path,
            lora_id,
        )
        training_args = GRPOConfig(
            learning_rate = 5e-6,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",
            logging_steps = 1,
            # host 1 GPU, 1 prompt per step, 4 generations per prompt. 

            #per_device_train_batch_size = (num_nodes-1)*gpu_count*task_batch_size / host_gpu_count
            per_device_train_batch_size = prompt_gen_batch_size*4, # how many to process at once per gpu
            # Controls hardware side, max number prompt per gpu

            gradient_accumulation_steps = 4, # how many steps to accumulate
            #how many steps accumulate before back pass

            num_generations = prompt_gen_batch_size, # How many generations for each prompt

            generation_batch_size = prompt_gen_batch_size*4,
            save_strategy="no",
            max_completion_length = 1024,
            max_steps = 4,
            save_steps = 4,
            max_grad_norm = 1.0,
            report_to = "none", # Can use Weights & Biases
            run_name=f"prolora-rome",
            output_dir = lora_id,
            #overwrite_output_dir=True,
        )
        eprint("Created grpo config")
        # each process receives: per_device_train_batch_size unique prompts
        # each process must return: per_device_train_batch_size × num_generations completions
        def rollout_func(prompts: list[str], trainer: GRPOTrainer, output_ddict=superfamily_ddict, gen_fam_ddict=gen_fam_ddict, reset_event = reset_event):
            eprint(f"Entered rollout_func with {len(prompts)} prompts")
            reset_event.clear()
            #model = trainer.model
            #tokenizer = trainer.processing_class
            #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            eos_id = trainer.processing_class.eos_token_id
            device = trainer.model.device
            import time
            n_gen = prompt_gen_batch_size
            total_gens = len(prompts)
            prev_prompt=""
            for prompt in list(set(prompts)):
                num = prompts.count(prompt)
                output_ddict[prompt] = num
                eprint(f"Adding {prompt} with {num} iteration to be generated")
                
            # wait for generations to be done
            prompt_ids, completion_ids, logprobs = [], [], []

            for prompt in list(dict.fromkeys(prompts)):
                p_gens=[]
                if prompt in gen_fam_ddict:
                    p_gens = gen_fam_ddict[prompt]
                while len(p_gens) < n_gen:
                    waiting_prompt = f"Prompt {prompt} has {len(p_gens)}/{n_gen} generations completed. Waiting..."
                    if waiting_prompt != prev_prompt:
                        eprint(waiting_prompt)
                        prev_prompt = waiting_prompt
                    time.sleep(1)
                    if prompt in gen_fam_ddict:
                        p_gens = gen_fam_ddict[prompt]
                eprint(f"Prompt {prompt} has completed. Continuing...")
                gens_for_prompt = gen_fam_ddict[prompt]
                for gen_result in gens_for_prompt:
                    original_prompt_id = gen_result["prompt_id"]
                    sequence = gen_result["sequence"]
                    tokens = gen_result["tokens"] + [eos_id]
                    logp = gen_result["logp"] + [0.0]
    
                    prompt_id = original_prompt_id
                    completion_id = tokens
                    logprob = logp
                    
                    prompt_ids.append(prompt_id)
                    completion_ids.append(completion_id)
                    logprobs.append(logprob)
                    eprint(f"Rolling out sequence {sequence}")
            #print("out of rollout")
            #print({
            #    "prompt_ids": len(prompt_ids),
            #    "completion_ids": len(completion_ids),
            #    "logprobs": len(logprobs),
            #})
            eprint("exiting rollout_func")
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
            }
    
        def sequence_reward(prompts, completions, seq_ddict=scored_ddict, **kwargs):
            rewards = []
            prev_prompt=""
            eprint(f"sequence_reward began with {len(completions)} completions")
            import time
            for prompt, completion in zip(prompts, completions):
                seq = completion[:-1] if completion.endswith('>') else completion
                while seq not in seq_ddict.keys():
                    waiting_msg = f"Sequence {seq} not scored yet. Waiting..."
                    if waiting_msg != prev_prompt:
                        eprint(waiting_msg)
                        prev_prompt = waiting_msg
                    time.sleep(1)
                seq_scores = seq_ddict[seq]
                #eprint(seq)
                #eprint(str(seq_scores))
                reward = seq_scores["top_prob"]  # use average foldseek probability as reward signal
                #print(reward)
                eprint(f"Sequence {seq} scored")
                rewards.append(reward)
                
                #rewards.append(1.0)
            eprint("sequence_reward complete")
            #reset_event.set()
            return rewards
        
        # build dataset of prompts from list of sequences
        # dataset is just list of prompts, it is raw superfamily string, rollout_func will handle prompt formatting
        formatted_data = [{"prompt": superfamily} for superfamily in superfamilies.keys()]
        superfamily_dataset = Dataset.from_list(formatted_data)
        eprint("Dataset made")
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            rollout_func=rollout_func,
            reward_funcs = [
                sequence_reward
            ],
            args = training_args,
            train_dataset = superfamily_dataset,
            callbacks = [GRPOCallback()],
        )
        trainer.train()
        
    _generate_terminate = asyncio.Event()
    _folding_terminate = asyncio.Event()
    _scoring_terminate = asyncio.Event()
    _iteration_reset = Event()
    
    async def gen_seq_listen_launch(gen_fam_ddict, fin_fam_ddict, seq_ddict, terminate_event, reset_event):
        eprint("gen_seq_listen_launch started")
        generated_families = []
        gen_task_futures = []
        while not terminate_event.is_set():
            if reset_event.is_set():
                logger.info("gen_seq_listen_launch reset")
                generated_families = []
                await asyncio.sleep(1)
                continue
            superfam_to_gen = []
            task_to_launch = []
            for family, num in gen_fam_ddict.items():
                if family not in generated_families:
                    eprint(f"Adding {family} with {num} to generate queue")
                    superfam_to_gen.append((family, num))
            for family, num in superfam_to_gen:
                gpu_alloc = gpu_allocator.next()
                task_policy_rr = Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=gpu_alloc[0],
                    gpu_affinity=gpu_alloc[1],
                )
                eprint(f"Submitting generation for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                #task_fut = generate_seq_epgf(family, num, gen_fam_ddict=gen_fam_ddict, seq_ddict=seq_ddict, task_description={"process_template": {"policy": task_policy_rr}})
                gen_task = generate_seq_epgf(
                    family, 
                    num_seq=num, 
                    gen_fam_ddict=fin_fam_ddict, 
                    seq_ddict=seq_ddict, 
                    task_description={"process_template": {"policy": task_policy_rr}}
                )
                eprint(f"Submitted generation for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                generated_families.append(family)
                gen_task_futures.append(gen_task)
            #eprint("gen_seq_listen_launch sleeping")
            await asyncio.sleep(1)
        eprint(f"Terminate event set for gen_seq_listen_launch, waiting for generation tasks to complete")
        await asyncio.gather(*gen_task_futures)
        eprint(f"All generation tasks completed for gen_seq_listen_launch")
        return "done"
    
    async def fold_seq_listen_launch(gen_fam_ddict, finished_fam_ddict, terminate_event, reset_event):
        eprint("fold_seq_listen_launch started")
        processed_families = []
        task_futures = []
        while not terminate_event.is_set():
            if reset_event.is_set():
                logger.info("fold_seq_listen_launch reset")
                processed_families = []
                await asyncio.sleep(1)
                continue
            families_to_fold = []
            tasks_to_launch = []
           
           
            for family, seq_list in gen_fam_ddict.items():
                if family not in processed_families and len(seq_list)==prompt_gen_batch_size:
                    eprint(f"Adding {family} with {len(seq_list)} sequences to fold queue")
                    families_to_fold.append(family)
            for family in families_to_fold:
                gpu_alloc = gpu_allocator.next()
                task_policy_rr = Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=gpu_alloc[0],
                    gpu_affinity=gpu_alloc[1],  
                )# async def fold_sequences(generated_families_ddict, superfamily, staging_dir, output_dir, finished_family_ddict, seed=None)
                eprint(f"Submitting folding for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                
                task_fut = fold_sequences(
                    gen_fam_ddict, 
                    family, 
                    f"/work/nvme/bdyk/apark4/ROME/stage-{superfamilies[family]}",
                    f"/work/nvme/bdyk/apark4/ROME/output-{superfamilies[family]}", 
                    finished_fam_ddict, 
                    task_description={"process_template": {"policy": task_policy_rr}}
                )
                
                #sing_fut = singularity_task(
                #    gen_fam_ddict, 
                #    family, 
                #    f"/work/nvme/bdyk/apark4/ROME/stage-{superfamilies[family]}", 
                #    f"/work/nvme/bdyk/apark4/ROME/output-{superfamilies[family]}", 
                #    finished_fam_ddict, 
                #    task_description={"process_template": {"policy": task_policy_rr}}
                #)
                #util_fut = post_sing_task(f"/work/nvme/bdyk/apark4/ROME/stage-{superfamilies[family]}", f"/work/nvme/bdyk/apark4/ROME/output-{superfamilies[family]}", sing_fut)
                #util_fut.add_done_callback(lambda fut, family=family: eprint(f"Post singularity task for {family} completed"))
                eprint(f"Submitted folding for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                processed_families.append(family)
                task_futures.append(task_fut)
            logger.info("fold_seq_listen_launch sleeping")
            await asyncio.sleep(1)
        eprint(f"Terminate event set for fold_seq_listen_launch, waiting for folding tasks to complete")
        await asyncio.gather(*task_futures)
        eprint(f"All folding tasks completed for fold_seq_listen_launch")
        return "done"
    
    #async def score_sequence(superfamily, pdb_input_dir, db, tmp_dir, output_file, seq_ddict = None, format_output='query,target,lddt,prob'):
    async def score_family_listen_launch(fin_fam_ddict, scored_ddict, terminate_event, reset_event, seq_ddict=None):
        eprint("score_family_listen_launch started")
        scored_families = []
        task_futures = []
        while not terminate_event.is_set():
            if reset_event.is_set():
                print("score_family_listen_launch reset")
                scored_families = []
                await asyncio.sleep(1)
                continue
            families_to_score = []
            tasks_to_launch = []
            # if fin_fam has keys, print keys
            for family in fin_fam_ddict.keys():
                if family not in scored_families:
                    eprint(f"Adding {family} to scoring queue")
                    families_to_score.append(family)
            for family in families_to_score:
                gpu_alloc = gpu_allocator.next()
                task_policy_rr = Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=gpu_alloc[0],
                    gpu_affinity=gpu_alloc[1],  
                )
                #async def score_sequence(superfamily, pdb_input_dir, db, tmp_dir, output_file, scored_ddict, input_fasta_path, seq_ddict = None, format_output='query,target,lddt,prob'):
                eprint(f"Submitting scoring for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                task_fut = score_sequence(
                    superfamily=family, 
                    pdb_input_dir=f"/work/nvme/bdyk/apark4/ROME/output-{superfamilies[family]}", 
                    db=f"{foldseekdb_path}/{superfamilies[family]}", 
                    tmp_dir=tmp_dir, 
                    output_file=f"/work/nvme/bdyk/apark4/ROME/output-{superfamilies[family]}/score.txt", 
                    scored_ddict=scored_ddict, 
                    input_fasta_path = f"/work/nvme/bdyk/apark4/ROME/stage-{superfamilies[family]}/sequences.fasta",
                    seq_ddict=seq_ddict,
                    task_description={"process_template": {"policy": task_policy_rr}}
                )
                eprint(f"Submitted scoring for family {family} on {gpu_alloc[0]} with {gpu_alloc[1]}")
                scored_families.append(family)
                task_futures.append(task_fut)
            await asyncio.sleep(1)
        eprint(f"Terminate event set for score_family_listen_launch, waiting for scoring tasks to complete")
        await asyncio.gather(*task_futures)
        eprint(f"All scoring tasks completed for score_family_listen_launch")
        return "done"
    
    generated_sequences_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)

    superfamily_to_generate_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    generated_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    folded_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    scored_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    
    print("Starting listener/Launchers", flush=True)
    gsl_fut = gen_seq_listen_launch( superfamily_to_generate_ddict, generated_families_ddict, generated_sequences_ddict, _generate_terminate, _iteration_reset)
    fsl_fut = fold_seq_listen_launch( generated_families_ddict, folded_families_ddict, _folding_terminate, _iteration_reset)
    sfl_fut = score_family_listen_launch( folded_families_ddict, scored_families_ddict, _scoring_terminate, _iteration_reset, seq_ddict=generated_sequences_ddict)
    listener_fut = asyncio.gather(gsl_fut, fsl_fut, sfl_fut)
    print(f"Submitting grpo_trainer with GPU {trainer_task_gpu[1]} on {trainer_task_gpu[0]}", flush=True)

    trainer = grpo_trainer(superfamily_to_generate_ddict, generated_families_ddict, folded_families_ddict, scored_families_ddict, _iteration_reset, task_description={"process_template": {"policy": trainer_task_policy}})
    await trainer
    print("Trainer finished, shutting down listeners", flush=True)
    _generate_terminate.set()
    _folding_terminate.set()
    _scoring_terminate.set()
    await asyncio.gather(listener_fut)
    print("Listeners shut down, exiting", flush=True)
    #gen_fam_dict={}
    #seq_dict={}
    #finished_family_dict={}
    #scored_dict={}
    #superfamily_dict={}
    #result = generate_seq_epgf("CheY-like superfamily", num_seq=4, gen_fam_ddict=gen_fam_dict, seq_ddict=seq_dict)
    #fold_res = fold_sequences(gen_fam_dict, "CheY-like superfamily", "/work/nvme/bdyk/apark4/ROME/stage", "/work/nvme/bdyk/apark4/ROME/output", finished_family_dict)
    #score_res = score_sequence("CheY-like superfamily", "/work/nvme/bdyk/apark4/ROME/output", "/work/hdd/bdyk/apark4/foldseek/afdb50/chey", "/work/nvme/bdyk/apark4/ROME/tmp", "/work/nvme/bdyk/apark4/ROME/output/score.txt", scored_dict, seq_dict)
    #grpo_trainer(superfamily_dict, gen_fam_dict, seq_dict)

    await flow.shutdown()
    print("Flow shutdown")
if __name__ == '__main__':
    asyncio.run(main())