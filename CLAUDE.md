# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with test deps)
pip install -e ".[test]"

# Run all fast tests (unit + mocked-integration, no model downloads)
pytest -m fast

# Run a single test file
pytest tests/unit/test_trainer.py

# Run a single test by name
pytest tests/unit/test_trainer.py::test_trainer_stores_args

# Run everything including slow end-to-end tests
pytest
```

The test suite is split by marker: `fast` (default for any test not explicitly marked `slow`) covers everything that doesn't need real models or GPUs. Slow tests require `pytest.importorskip` guards and real dependencies.

The `conftest.py` stubs out torch, transformers, trl, peft, radical.asyncflow, dragon, and rose when they aren't installed, so unit tests run on a minimal environment.

## Architecture

ROME is a framework for distributed RL fine-tuning of LLMs. The core abstraction is a **Workflow** that separates *generation*, *scoring*, and *training* into independently scheduled tasks, backed by radical.asyncflow/rhapsody/dragon for distributed execution.

### Key abstractions

**`ModelConfig`** (`rome/config.py`) — dataclass carrying everything needed to load a model: `base_model_name`, optional `model_name`, optional `lora_name`/`lora_config`, `generation_config`, `dtype`, `device_map`, `max_seq_length`.

**`Trainer`** (`rome/trainer.py`) — abstract base. Concrete subclasses: `GRPO` (`rome/train/grpo.py`) and `SFT` (`rome/train/sft.py`, in progress). `reward_funcs` is the list of reward callables passed to the trainer. Functions decorated with `@Workflow.reward_task` are expected to be run as external tasks by the workflow; all others run inline inside the trainer.

**`Workflow`** (`rome/workflow.py`) — abstract base with two static decorators:
- `@Workflow.reward_task` — marks a reward function to be dispatched as a task (sets `func._is_reward_task = True`)
- `@Workflow.evaluate_task` — marks an evaluation function to be dispatched as a task

**`SequentialFlow`** (`rome/flows/sequentialflow.py`) — the main implemented workflow. Runs a sequential RL loop: generate → score → train. Uses `SequentialReinforcementLearner` from rose and coordinates work via Dragon `DDict` (shared distributed dictionary) and `Event` objects. Generator and scorer tasks are launched as asyncflow function tasks; the trainer runs as an `@rl.update_task`; the evaluator runs as `@rl.as_stop_criterion`.

**`load_model` / `reload_lora` / `save_model`** (`rome/utils.py`) — utilities for model I/O. `reload_lora` and `save_model` are currently stubs (empty `pass`).

### GRPO orchestration detail

When `@Workflow.reward_task`-decorated reward functions are present, `GRPO.train()` wraps them with `_reward_func_wrapper`, which reads results from `workflow_ddict[f"reward_{name}_outputs"]` and waits (polls with `asyncio.sleep`) until all `request_ids` are present. The `_default_rollout_func` writes prompts into `workflow_ddict["generation_requests"]` and waits on `workflow_ddict["generator_outputs"]` — this is how the trainer hands off generation to external generator tasks and retrieves results.

### SequentialFlow DDict key conventions

| Key | Written by | Read by |
|---|---|---|
| `generation_requests` | rollout_func | `_generation_schedule` |
| `generator_{i}_input` | `_generation_schedule` | generation_task i |
| `generator_{i}_output` | generation_task i | `_generation_gather` |
| `generator_outputs` | `_generation_gather` | rollout_func, `_scorer_schedule` |
| `reward_{name}_{i}_input` | `_scorer_schedule` | scorer_task i |
| `reward_{name}_{i}_output` | scorer_task i | `_scorer_gather` |
| `reward_{name}_outputs` | `_scorer_gather` | `_reward_func_wrapper` |

### Examples

`examples/math_reasoning/` — GSM8K math reasoning:
- `update.py` / `reward.py` — standalone training and evaluation scripts (no ROME)
- `sequentialflow_math.py` — same use case wired into `SequentialFlow`

`protein_generation/` — protein sequence generation with ProLLaMA:
- `EPGF.py` / `dragon_protein_run.py` — full distributed implementations using rhapsody/dragon
- `baseline.py` — single-script baseline using stock `GRPOTrainer` + `EPGFLogitsProcessor` for constrained generation (multi-GPU safe via `accelerate launch`)
- `ProteinModel.py` — `ProteinSequenceScorer` for composition-based sequence scoring
- `build_dataset.py` / `parse_foldseek.py` — dataset construction from Foldseek output

### Known incomplete areas

- `reload_lora` and `save_model` in `rome/utils.py` are empty stubs
- `SFT` trainer (`rome/train/sft.py`) has a stub `train()` method
- `SFT` and `SequentialFlow` are commented out of `rome/__init__.py` and `rome/train/__init__.py`
- No per-iteration evaluation call in `SequentialFlow.launch()` beyond the stop criterion
- `SequentialFlow.launch()` does not pass a dataset to `trainer.train()`
