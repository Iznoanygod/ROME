import argparse, asyncio, yaml, random, logging, time, os, itertools, sys
from more_itertools import batched
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

os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
os.environ["HF_HOME"] = "/work/nvme/bdyk/apark4/huggingface"
foldseek_path = "/work/hdd/bdyk/apark4/foldseek/bin/foldseek"
colabfold_path = "/work/nvme/bdyk/apark4/localcolabfold/.pixi/envs/default/bin/colabfold_batch"
runfold_path = "/work/nvme/bdyk/apark4/ROME/run_fold.sh"
runsingularity_path = "/work/nvme/bdyk/apark4/ROME/run_singularity.sh"
tmp_dir = "/tmp"
colabfold_sif_path = "/work/nvme/bdyk/apark4/ROME/colabfold_1.6.0-cuda12.sif"
fold_cache = "/work/hdd/bdyk/apark4/foldcache"
foldseekdb_path = "/work/hdd/bdyk/apark4/foldseek/afdb50"
storage_path = "/work/hdd/bdyk/apark4/ROME/storage"

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
    #rhapsody.enable_logging(level=logging.DEBUG)
    all_gpus = find_gpus()
    trainer_task_gpu = all_gpus[0]  # assign the whole first node to trainer task
    remaining_nodes_gpus = all_gpus[1:]  # use the rest of the nodes for generation and folding

    class RoundRobinGPU:
        def __init__(self, node_gpu_list):
            self.gpu_list = []
            for node, gpus in node_gpu_list:
                for gpu in list(batched(gpus, 2)):
                    self.gpu_list.append((node, gpu))
            self._iterator = itertools.cycle(self.gpu_list)
        def next(self):
            return next(self._iterator)

    gpu_allocator = RoundRobinGPU(remaining_nodes_gpus)

    trainer_task_policy = Policy(
        placement=Policy.Placement.HOST_NAME,
        host_name=trainer_task_gpu[0],
        gpu_affinity=[0],  # assign all GPUs on the trainer node to the trainer task
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
    async def generate_seq_epgf(superfamily_batch, fam_seq_ddicts, _started_event, _failed_event, _hostname, _terminate_event, seed=42):
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
        
        if socket.gethostname() != _hostname:
            _failed_event.set()
            return -1
        _started_event.set()
        #eprint(f"generate_seq_epgf {superfamily_batch} with GPU {cuda_devices} on host {socket.gethostname()}")
        def set_seed(seed=seed) -> None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            random.seed(seed)
        """
            Load base model and optionally attach a local LoRA adapter (directory).
            If `lora_id` is a directory containing a checkpoint, attach it via
            `PeftModel.from_pretrained`. Otherwise, if `args.apply_lora` is set,
            construct a `LoraConfig` and wrap the base model with `get_peft_model`.
        
            Returns: model, tokenizer, loaded_from, iteration
            """
        def load_llama_or_latest_checkpoint(
            base_model_id: str,
            lora_id: str,
            dtype=torch.bfloat16,
            device_map="auto",
        ):
            last_checkpoint = None
            from pathlib import Path
            if (lora_id is not None) and os.path.isdir(lora_id) and any(Path(lora_id).iterdir()):
                last_checkpoint = True
        
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left", use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            iteration = 0
            if last_checkpoint is not None:
                #eprint(f"Found LoRA checkpoint")
                #eprint(f"Loading base model: {base_model_id}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    device_map=device_map,
                )
                # Attach LoRA adapter weights
                #eprint("Applying LoRA adapter from checkpoint...")
                model = PeftModel.from_pretrained(base_model, lora_id, is_trainable=True)
                loaded_from = last_checkpoint
            else:
                #eprint(f"No checkpoint found, loading base model: {base_model_id}")
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
        
        #seq_with_scores = []
        set_seed(seed)
        with torch.no_grad():
            while not _terminate_event.is_set():
                for superfamily in superfamily_batch:
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
                        #seq_with_scores.append({"prompt_id": original_prompt_id, "sequence":sequence, "tokens":tokens, "logp":logp, "superfamily": superfamily})
                        
                        #if fam_seq_ddict is not None:
                            
                        #    fam_seq_ddict[sequence] = {"prompt_id": original_prompt_id, "tokens":tokens, "logp":logp, "superfamily": superfamily}
                        ddict = fam_seq_ddicts[superfamily]
                        ddict[sequence] = {"prompt_id": original_prompt_id, "tokens":tokens, "logp":logp, "superfamily": superfamily}
                        #eprint(f"Generated sequence {sequence}")
            
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        eprint(f"finished generating {superfamily}")
        
        return seq_with_scores
    
    # Input generated_family ddict with list of sequences, outputs list of folded struct files, fill ddict with family with list of sequences
    @flow.function_task
    async def fold_sequences(sequences, staging_dir, output_dir, finished_seq_ddict, _start_event, _failed_event, _hostname, seed=None, use_singularity=True):
        import os, subprocess, re, shutil, socket, sys
        if socket.gethostname() != _hostname:
            _failed_event.set()
            return -1
        _start_event.set()
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        import logging
        logger = logging.getLogger(__name__)
        from pathlib import Path
        if seed is None:
            import random
            seed = random.randint(0, 65535)
        import uuid
        file_ruuid = str(uuid.uuid4())
        tmp_stage_path = Path(f"/tmp/{file_ruuid}")
        eprint(str(len(os.sched_getaffinity(0))))
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        #eprint(f"fold_sequences {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        output_path = Path(output_dir)
        staging_path = Path(staging_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir(parents=True, exist_ok=True)
        try:
            tmp_stage_path.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            #eprint("tmp staging directory already exists, clearing it out")
            shutil.rmtree(tmp_stage_path)
            tmp_stage_path.mkdir(parents=True, exist_ok=False)
        #superfamily_sequences = generated_families_ddict[superfamily]
        input_file = tmp_stage_path / "sequences.fasta"
        
        with open(input_file, 'w') as f:
            for i, sequence in enumerate(sequences):
                f.write(f'>{i}\n')
                f.write(sequence + '\n')
        if not use_singularity:
            eprint("calling runfold")
            colabfold_result = subprocess.run([
                runfold_path,
                input_file,
                tmp_stage_path,
            ], capture_output=True, text=True)
            eprint(colabfold_result.stdout)
        else:
            #eprint("calling singularity")
            command = [
                runsingularity_path,
                colabfold_sif_path,
                "sequences.fasta",
                f"{tmp_stage_path}",
                f"{fold_cache}",
                f"{seed}"
            ]
            #instead fake it by creating fast pdb
            for i, sequence in enumerate(sequences):
                fake_pdb = tmp_stage_path / f"{i}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                with open(fake_pdb, 'w') as f:
                    f.write("HEADER    FAKE PDB\n")
                    f.write("END\n")
            #eprint(str(command))
            #singularity_result = subprocess.run(command, capture_output=True, text=True)
            #eprint(singularity_result.stdout)

        #staging_path = Path(staging_dir)
        #staging_path.mkdir(parents=True, exist_ok=True)
    
        pattern = re.compile(r'^(.+)_unrelaxed_rank_001_alphafold2_ptm_model_\d+_seed_000\.pdb$')
        #eprint("copying pdb files")
        if not use_singularity:
            for pdb_file in staging_path.glob('*.pdb'):
                if pattern.match(pdb_file.name):
                    seq_name = pdb_file.name.split("_")[0]
                    if seq_name in sequences:
                        shutil.copy(pdb_file, output_path / f"{seq_name[:200]}.pdb")
                        finished_seq_ddict[seq_name] = f"{seq_name[:200]}.pdb"
        else:
            tmp_path = Path(f"/tmp/{file_ruuid}")
            for pdb_file in tmp_path.glob('*.pdb'):
                if pattern.match(pdb_file.name):
                    seq_id = int(pdb_file.name.split("_")[0])
                    seq_name = sequences[seq_id]
                    shutil.copy(pdb_file, output_path / f"{pdb_file.name}")
                    finished_seq_ddict[seq_name] = f"{pdb_file.name}"
        #finished_family_ddict[superfamily] = len(superfamily_sequences)
        #eprint(f"finished folding")
        return sequences
    @flow.function_task
    async def score_sequence(superfamily, sequences, pdb_input_dir, output_file, scored_ddict, format_output='query,target,lddt,prob'):
        # Perform foldseek search on sequences and return hprob scores
        import os, socket, sys
        import uuid
        file_ruuid = str(uuid.uuid4())
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        eprint(str(len(os.sched_getaffinity(0))))
        import logging
        logger = logging.getLogger(__name__)
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        eprint(f"score_sequence {superfamily} with GPU {cuda_devices} on host {socket.gethostname()}")
        if True:
            return
        import subprocess
        import csv
        from pathlib import Path
        import re
        if seq_ddict is not None:
            logger.info("DDict passed to fold")
        pdb_input_path = Path(pdb_input_dir)

        tmp_path       = Path(f"/tmp/seek-{file_ruuid}")
        tmp_stage_path = Path(f"/tmp/seek_stage-{file_ruuid}")
        db = f"{foldseekdb_path}/{superfamilies[superfamily]}"
        output_path    = tmp_path / output_file
        tmp_path.mkdir(parents=True, exist_ok=True)
        #eprint(str(tmp_path))
        #first move all sequence pdb files to tmp_path
        # it is {sequence}.pdb
        for i, sequence in enumerate(sequences):
            pdb_file = pdb_input_path / f"{sequence[:200]}.pdb"
            if pdb_file.exists():
                shutil.copy(pdb_file, tmp_stage_path / f"{i}.pdb")
                
        eprint("running foldseek")
        # Run foldseek
        cmd = [
            foldseek_path, 'easy-search',
            str(tmp_stage_path),
            db,
            str(output_path),
            str(tmp_path),
            '--format-output', format_output,
        ]
        #eprint(str(cmd))
        #result = subprocess.run(cmd, capture_output=True, text=True)
        eprint("foldseek finished")
    
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
        for i in range(len(sequences)):
            #first have to check if foldseek returned no hits for the sequence. In this case prob=-1, lddt=-1, target=None
            if str(i) not in hits:
                hits[str(i)] = [("None", -1, -1)]
        # create scored sequence list with top hit and average scores
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
    async def grpo_trainer(scored_fam_seq_ddict, _started_event, _failed_event, _hostname, _terminate_event, task_description=trainer_process_template):
        import sys
        def eprint(str):
            sys.stderr.write(str + "\n")
            sys.stderr.flush()
        #sys.stderr.write("test")
        import logging, os, re, socket
        if socket.gethostname() != _hostname:
            eprint(f"generate_seq_epgf unexpected host {socket.gethostname()}, expected {_hostname}. Terminating")
            print(f"generate_seq_epgf unexpected host {socket.gethostname()}, expected {_hostname}. Terminating")
            _failed_event.set()
            return -1
        _started_event.set()
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
        eprint(f"grpo_trainer with GPU {cuda_devices} on host {socket.gethostname()}")
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
            overwrite_output_dir=True,
        )
        eprint("Created grpo config")
        # each process receives: per_device_train_batch_size unique prompts
        # each process must return: per_device_train_batch_size × num_generations completions
        def rollout_func(prompts: list[str], trainer: GRPOTrainer, output_ddict=scored_fam_seq_ddict):
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
                
            # wait for generations to be done
            prompt_ids, completion_ids, logprobs = [], [], []

            for prompt in list(dict.fromkeys(prompts)):
                fam_seq_ddict = scored_fam_seq_ddict[prompt]


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
    
        async def sequence_reward(prompts, completions, seq_ddict=scored_ddict, **kwargs):
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
                    await asyncio.sleep(1)
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
        
    _terminate = asyncio.Event()
    _iteration_reset = Event()
    
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


    async def launch_fold_with_retry(sequences_to_fold, folded_seq_ddict):
        gpu_alloc = gpu_allocator.next()
        task_policy_rr = Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=gpu_alloc[0],
            gpu_affinity=gpu_alloc[1],
        )
        
        success_event = Event()
        fail_event = Event()
        while True:
            eprint(f"Submitting folding for batch of {len(sequences_to_fold)} sequences on {gpu_alloc[0]} with {gpu_alloc[1]}")
            task = fold_sequences(
                sequences_to_fold, 
                f"/work/nvme/bdyk/apark4/ROME/stage", 
                f"/work/nvme/bdyk/apark4/ROME/output", 
                folded_seq_ddict,
                success_event,
                fail_event,
                gpu_alloc[0],
                task_description={"process_template": {"policy": task_policy_rr}}
            )
            while not success_event.is_set() and not fail_event.is_set():
                await asyncio.sleep(5)
            
            if success_event.is_set():
                #eprint("colabfold launched")
                return task
            fail_event.clear()
            success_event.clear()

    async def fold_seq_batch_launch(fam_seq_ddict, folded_seq_ddict, batch_size, max_wait_time, terminate_event):
        #eprint("fold_seq_batch_launch started")
        processed_sequences = []
        task_futures = []
        import datetime
        
        while not terminate_event.is_set():
            tasks_to_launch = []
            sequences_to_fold = []
            first_add_time = 0
            while first_add_time == 0 or (datetime.datetime.now() - first_add_time).total_seconds() < max_wait_time:
                for seq_ddict in fam_seq_ddict.values():
                    for sequence in seq_ddict.keys():
                        if sequence not in processed_sequences and sequence not in sequences_to_fold:
                            #eprint(f"Adding sequence {sequence} to fold queue")
                            if len(sequences_to_fold) == 0:
                                first_add_time = datetime.datetime.now()
                            sequences_to_fold.append(sequence)
                            if len(sequences_to_fold) == batch_size:
                                break
                    if len(sequences_to_fold) == batch_size:
                        break
                if len(sequences_to_fold) == batch_size:
                    break
                await asyncio.sleep(1)
            #eprint(f"Batch of {len(sequences_to_fold)} sequences ready for folding")
            task_fut = await launch_fold_with_retry(sequences_to_fold, folded_seq_ddict)
            processed_sequences.extend(sequences_to_fold)
            task_futures.append(task_fut)
            await asyncio.sleep(1)
    
    async def launch_score_with_retry(superfamily, sequences, pdb_input_dir, output_file, scored_ddict, format_output='query,target,lddt,prob'):
        gpu_alloc = gpu_allocator.next()
        task_policy_rr = Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=gpu_alloc[0],
            gpu_affinity=gpu_alloc[1],
        )
        
        success_event = Event()
        fail_event = Event()
        while True:
            eprint(f"Submitting folding for batch of {len(sequences_to_fold)} sequences on {gpu_alloc[0]} with {gpu_alloc[1]}")
#    async def score_sequence(superfamily, sequences, pdb_input_dir, output_file, scored_ddict, format_output='query,target,lddt,prob'):
            task = score_sequence(
                    superfamily=superfamily,
                    sequences=sequences,
                    pdb_input_dir=pdb_input_dir, 
                    output_file=f"scores.txt", 
                    scored_ddict=scored_ddict, 
                    task_description={"process_template": {"policy": task_policy_rr}}
            )
            while not success_event.is_set() and not fail_event.is_set():
                await asyncio.sleep(5)
            
            if success_event.is_set():
                eprint("foldseek launched")
                return task
            fail_event.clear()
            success_event.clear()
    
    async def seek_seq_batch_launch(fam_seq_ddict, folded_seq_ddict, scored_fam_seq_ddict, batch_size, terminate_event):
        # assuing folded_seq_dict filled with seq unlabelled, take sequence, find superfamily from 
        eprint("seek_seq_batch_launch started")
        scored_sequences = []
        task_futures = []
        fam_seq_to_score={}
        while not terminate_event.is_set():
            tasks_to_launch = []
            for superfamily in superfamilies.keys():
                fam_seq_to_score[superfamily] = []
            
            for superfamily, seq_ddict in fam_seq_ddict.items():
                for sequence in folded_seq_ddict.keys():
                    if sequence in seq_ddict.keys() and sequence not in scored_sequences and sequence not in fam_seq_to_score[superfamily]:
                        eprint(f"Adding sequence {sequence} of superfamily {superfamily} to seek queue")
                        fam_seq_to_score[superfamily].append(sequence)
            
            for superfamily, sequences in fam_seq_to_score.items():
                if len(sequences) == 0:
                    continue
                # task fut
                eprint(f"Batch of {len(sequences)} sequences ready for seeking for superfamily {superfamily}")
                task_fut = await launch_score_with_retry(
                    superfamily,
                    sequences, 
                    f"/work/nvme/bdyk/apark4/ROME/output",
                    "scores.txt",
                    scored_fam_seq_ddict
                )
                scored_sequences.extend(sequences)
                task_futures.append(task_fut)
            await asyncio.sleep(1)
                
        
    
    #generated_sequences_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)

    #superfamily_to_generate_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    #generated_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    #folded_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    #scored_families_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    generator_tasks = []
    generator_success_event = Event()
    generator_fail_event = Event()
    superfamily_seq_ddict = {}
    scored_superfamily_seq_ddict = {}
    for superfamily in superfamilies.keys():
        superfamily_seq_ddict[superfamily] = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
        scored_superfamily_seq_ddict[superfamily] = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    folded_seq_ddict = DDict(managers_per_node=1, n_nodes =2, total_mem=1024**3)
    print("Starting sequence generator", flush=True)

    #manually have to check if the task starts on the correct node
    #async def generate_seq_epgf(superfamily_batch, fam_seq_ddicts, _started_event, _failed_event, _hostname, _terminate_event, num_seq=64, seed=42):
    #start 8 tasks each handles two superfamilies, 
    async def launch_generator_with_retry(superfamily_batch, gpu_alloc):
        task_policy_rr = Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=gpu_alloc[0],
            gpu_affinity=gpu_alloc[1],
        )
        success_event = Event()
        fail_event = Event()
        print(f"Submitting generation for {superfamily_batch[0]} on {gpu_alloc[0]}", flush=True)
        while True:
            generator_task = generate_seq_epgf(
                    superfamily_batch,
                    superfamily_seq_ddict,
                    success_event,
                    fail_event,
                    gpu_alloc[0],
                    _terminate,
                    task_description={"process_template": {"policy": task_policy_rr}}
                )
            while not success_event.is_set() and not fail_event.is_set():
                #print(f"Waiting for {superfamily_batch[0]} to start...", flush=True)
                await asyncio.sleep(5)
            
            if success_event.is_set():
                #print(f"Generator for {superfamily_batch[0]} started on {gpu_alloc[0]}", flush=True)
                return generator_task
            
            #print(f"{superfamily_batch[0]} failed, killing...", flush=True)
            #await generator_task
            #print(f"{superfamily_batch[0]} killed, retrying...", flush=True)
            fail_event.clear()
            success_event.clear()
    gen_launch_tasks = [
        launch_generator_with_retry(superfamily_batch, gpu_allocator.next())
        for superfamily_batch in list(batched(superfamilies.keys(), 2))
    ]
    folder_task = asyncio.create_task(
        fold_seq_batch_launch(
            superfamily_seq_ddict,
            folded_seq_ddict,
            batch_size=4,
            max_wait_time=60,
            terminate_event=_terminate))
    seeker_task = asyncio.create_task(
        seek_seq_batch_launch(
            superfamily_seq_ddict,
            folded_seq_ddict,
            scored_superfamily_seq_ddict,
            batch_size=4,
            terminate_event=_terminate))
    generator_tasks = await asyncio.gather(*gen_launch_tasks)
    print("All generators started", flush=True)

   # async def fold_seq_batch_launch(seq_ddict, batch_size, max_wait_time, terminate_event):

    #gsl_fut = gen_seq_listen_launch( superfamily_to_generate_ddict, generated_families_ddict, generated_sequences_ddict, _generate_terminate, _iteration_reset)
    #fsl_fut = fold_seq_listen_launch( generated_families_ddict, folded_families_ddict, _folding_terminate, _iteration_reset)
    #sfl_fut = score_family_listen_launch( folded_families_ddict, scored_families_ddict, _scoring_terminate, _iteration_reset, seq_ddict=generated_sequences_ddict)
    #listener_fut = asyncio.gather(gsl_fut, fsl_fut, sfl_fut)
    await asyncio.gather(*generator_tasks)
    
    print(f"Submitting grpo_trainer with GPU {trainer_task_gpu[1]} on {trainer_task_gpu[0]}", flush=True)
    trainer = grpo_trainer(superfamily_seq_ddict, _terminate)
    #await trainer
    
    print("Trainer finished, shutting down listeners", flush=True)
    _terminate.set()
    #_folding_terminate.set()
    #_scoring_terminate.set()
    await asyncio.gather(*generator_tasks)
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