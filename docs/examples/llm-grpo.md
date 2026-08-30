# LLM self-improvement with all three managers

`examples/agnostic/llm_grpo_streams.py`

The counterpart to the IMPRESS-R examples. Those use only the data and training
halves, because IMPRESS owns its own inference. This one uses the stream manager
too: **generation and scoring run as persistent asynchronous tasks, and they
hot-swap onto each new LoRA adapter the trainer publishes** — without the workflow
orchestrating the swap.

```text
inference stream  ──generations──▶  reward stream  ──scores──▶  Data Manager
         ▲                                                            │
         └──────────── checkpoint ◀── Training Manager ◀──────────────┘
```

```bash
dragon examples/agnostic/llm_grpo_streams.py
```

Needs a GPU, TRL, and access to `meta-llama/Llama-3.2-1B-Instruct`.

## The workflow's own code

Nothing ROME-specific appears in the inference or the reward function. This is
the whole of the model-facing code:

```python
def load_generator(checkpoint_path, ctx):
    config = ModelConfig(base_model_name=BASE_MODEL, lora_name=checkpoint_path,
                         required_gpus=1)
    model, tokenizer = load_model(config)
    return {"model": model, "tokenizer": tokenizer}


def generate(prompts, ctx):
    model, tokenizer = ctx.model["model"], ctx.model["tokenizer"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
```

`load_generator` is called at startup **and again on every published
checkpoint**. Note what it reloads: a LoRA adapter, not a full model. That is why
a stream can swap weights between batches without stalling the campaign on a
multi-gigabyte read — and why `save_model` writes only the adapter when one is
configured.

## Scoring in its own stream

```python
def score(generations):
    scored = []
    for item in generations:
        reward = 0.0
        if any(ch.isdigit() for ch in item["completion"]):
            reward += 1.0
        if len(item["completion"]) < 600:
            reward += 0.5
        scored.append({**item, "score": reward})
    return scored
```

Two things to notice.

**It takes no `ctx`.** ROME-A inspects the signature and only passes the context
when there is somewhere to put it, so an existing reward function is adopted
unchanged.

**Its return value becomes corpus records.** The manager wires a
`StreamKind.REWARD` group's outputs into the data manager automatically, so
`score` returning a list of dicts is all the plumbing there is between scoring and
the training corpus.

Running the reward as its own stream is the point of the split: it gets its own
tasks and its own resources (here `num_gpus=0`, since it is pure Python), so it
can be as slow as it likes without occupying a GPU or blocking generation.

!!! info "Inline rewards versus reward streams"

    `GRPOConfig(reward_funcs=[])` is empty on purpose here. TRL's `reward_funcs`
    are called *inside* the training round; anything expensive or resource-hungry
    belongs in a reward stream instead. Use inline rewards for cheap, pure
    functions of the text; use a stream for anything that needs a model, a
    simulator or a node.

## Wiring it up

```python
manager = rome.Manager(
    flow,
    data_config=rome.DataConfig(min_samples=32, sampling="top_k", shard_size=128),
    trainer_config=rome.TrainerConfig(
        trainer=GRPOTrainer(GRPOConfig(model_config=model_config,
                                       reward_funcs=[], num_generations=4)),
        checkpoint_dir="./rome_checkpoints",
        poll_interval=5.0,
    ),
    stream_configs=[
        rome.StreamConfig(name="generate", kind=rome.StreamKind.INFERENCE,
                          model_path=model_config.lora_name,
                          load_func=load_generator, process_func=generate,
                          num_streams=2, num_gpus=1, batch_size=4),
        rome.StreamConfig(name="score", kind=rome.StreamKind.REWARD,
                          process_func=score,
                          num_streams=2, num_gpus=0, batch_size=8),
    ],
)
```

`sampling="top_k", shard_size=128` means every round trains on the best 128
completions the campaign has produced so far, not the most recent ones — the
corpus is never deleted, so "best so far" keeps meaning something across rounds.

## The driver loop

```python
for round_index in range(20):
    manager.stream.submit_batch(PROMPTS, stream="generate")

    for output in manager.stream.get_outputs(stream="generate"):
        manager.stream.submit(output["result"], stream="score")

    await asyncio.sleep(5.0)
```

That is the entire orchestration: feed generation, forward its completions to
scoring. Scored results reach the corpus on their own, training fires on its own,
and the streams reload on their own.

`model_path=model_config.lora_name` on the inference group points the first load
at the adapter directory, so a run resumes from a previously trained adapter
rather than always starting from the base model.

## What to watch

```text
round 0: corpus 0  | model v0 | NOT_ENOUGH_DATA | streams ['RUNNING', 'RUNNING', 'RUNNING', 'RUNNING']
round 7: corpus 34 | model v0 | WAITING         | streams ['RUNNING', ...]
round 8: corpus 38 | model v1 | NOT_ENOUGH_DATA | streams ['RELOADING', 'RUNNING', ...]
```

A replica showing `RELOADING` is the loop closing: the trainer published v1, the
callback fired, and that replica is loading the new adapter between batches. The
other replicas are still serving from v0 until they reach their own batch
boundary — which is exactly right, and why the campaign never pauses.

## API reference

* [`rome.train.llm`](../api/rome/train/llm.md) — `GRPOTrainer`, `GRPOConfig`,
  `ModelConfig`, `load_model`, `save_model`
* [`rome.stream`](../api/rome/stream.md) — `StreamConfig`, `StreamContext`
