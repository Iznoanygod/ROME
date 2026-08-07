# ROME

The RADICAL Optimizer for Model Enhancement (ROME).

## ROME-A

The original ROME exposed self-improvement as a workflow with three components —
model inference, reward/simulation, and model training. Building a new
improvement workflow with ROME was easy; plugging ROME into a workflow you
already had was not.

**ROME-A** makes ROME workflow agnostic. Model improvement becomes a set of
pluggable modules you add to an existing workflow rather than a workflow you
adopt. Each component is a configurable unit, adding a new model or training
algorithm is a single task, and adoption costs a few API calls.

### The three managers

| Component | What it does | API |
| --- | --- | --- |
| **Data Manager** (`rome.data`) | Collects scored outputs from the host workflow and builds them into a training dataset. Handles organization and synchronization of the data across nodes and tasks, so a task anywhere in the workflow can contribute. | `add_training_data`, `get_training_dataset` |
| **Stream Manager** (`rome.stream`) | Runs inference and reward as persistent asynchronous tasks using workflow-supplied code, and reloads the model when it sees a new checkpoint. | `submit`, `get_outputs`, `reload_model` |
| **Training Manager** (`rome.trainer`) | Creates and schedules training tasks on HPC, publishes updated checkpoints back to the workflow, and reports whether training is possible, running or finished. | `start_training`, `get_training_status`, `get_current_model` |

`rome.Manager` wires the three together. The line that closes the loop is the
training manager's checkpoint callback being the stream manager's reload hook:
a completed training round *is* the event that swaps inference onto the new
model.

### Adoption

```python
import rome
from rome.train.mpnn import ProteinMPNNTrainer, impress_corpus_filter

manager = rome.Manager(
    asyncflow,                                  # your existing WorkflowEngine
    data_config=rome.DataConfig(
        min_samples=64,
        filter_func=impress_corpus_filter(),    # only train on confident designs
    ),
    trainer_config=rome.TrainerConfig(trainer=ProteinMPNNTrainer()),
)
await manager.start()

# ... your workflow runs unchanged ...
manager.add_training_data(sequence=seq, pdb_path=pdb, score=plddt)
weights = manager.get_current_model()           # None until the first round

await manager.stop()
```

Training starts automatically once `min_samples` fresh records accumulate, or
on demand via `await manager.start_training()`.

Workflows that also want ROME-A to own inference add stream configs:

```python
manager = rome.Manager(
    asyncflow,
    stream_configs=[
        rome.StreamConfig(name="generate", load_func=my_load, process_func=my_infer,
                          num_streams=4, num_gpus=1),
        rome.StreamConfig(name="score", kind=rome.StreamKind.REWARD,
                          process_func=my_reward, num_streams=2),
    ],
    ...
)
```

Reward-stream outputs feed the data manager automatically.

### Adding a training algorithm

Subclass `TrainTask`, implement one method, declare what it needs:

```python
class MyTrainer(rome.TrainTask):
    def train(self, dataset, output_dir, **kwargs) -> str:
        ...                       # dataset is what the data manager built
        return output_dir         # path the streams will reload from

rome.TrainerConfig(trainer=MyTrainer(gpus=4, nodes=2))
```

A bare `(dataset, output_dir, **kwargs) -> checkpoint_path` function works too —
it is wrapped in a `FunctionTrainer` for you. Two trainers ship with ROME-A:
`rome.train.llm.GRPOTrainer` (TRL/GRPO for LLMs) and
`rome.train.mpnn.ProteinMPNNTrainer` (IMPRESS-R).

### Runtime

ROME-A schedules nothing itself. Training rounds and stream tasks are submitted
to the `radical.asyncflow` `WorkflowEngine` the host workflow passes in, with
per-task resources given as an asyncflow `task_description`. Shared state lives
in a Dragon `DDict`; pass your own via `Manager(..., ddict=...)` and ROME-A will
namespace its keys under `rome|` so nothing collides with the workflow's own.

### Use case: IMPRESS-R

IMPRESS runs backbone → ProteinMPNN → structure prediction → pLDDT/pTM/pAE →
keep/fallback/migrate/drop. It is open loop: each campaign improves the designs,
never the model. IMPRESS-R adds ROME-A so the campaign's own highest-confidence
sequences fine-tune ProteinMPNN mid-campaign, and the improved model returns to
the pipeline. IMPRESS itself runs unchanged.

See `examples/agnostic/impress_r.py` (data + training) and
`examples/agnostic/llm_grpo_streams.py` (all three managers).

## Layout

```
rome/            ROME-A
  manager.py       Manager — wires the three components together
  data.py          Data Manager
  stream.py        Stream Manager
  trainer.py       Training Manager
  train/           trainer tasks (base, llm/GRPO, mpnn/ProteinMPNN)
  utils.py         DDict layout helpers + asyncflow submission
oldrome/         the original ROME flows, kept for reference
examples/        ROME-A adoption examples
protein_generation/  IMPRESS pipeline scripts
```

## Tests

```bash
pip install -e '.[test]'
pytest -m fast      # unit + mocked integration, no GPUs
```
