"""Dummy components for exercising ROME without a model.

Everything here has the same shape as a real ROME trainer or stream function
and none of the cost: the trainer sleeps instead of fine-tuning, and the model
emits a placeholder string instead of generating. That makes them useful for

* smoke-testing a new execution backend or allocation, where the question is
  whether tasks are placed and whether the DDict is reachable, not whether the
  model learns;
* demonstrating the closed loop end to end in seconds (see
  ``examples/agnostic/dummy_loop.py``);
* standing in for one half of a real campaign while the other half is built.

The dummy checkpoint is a real file on disk holding the round's version, so a
stream that reloads it genuinely observes the new version rather than being told
about it. A run therefore fails in the same places a real one would --- an
unwritable checkpoint directory, a DDict that is not shared, a stream that never
sees a publication.

Both the trainer and the inference function block on ``time.sleep``. That is
correct rather than sloppy: ROME runs a synchronous trainer inside an
asyncflow task and a synchronous ``process_func`` in a worker thread, so neither
sleep stalls the event loop the other streams are sharing.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from rome.train.base import TrainTask

DEFAULT_TEMPLATE = "model example output [{uuid}]"
"""What a dummy model "generates".

``{uuid}`` is a fresh UUID per output; ``{version}`` and ``{index}`` are
available for callers who want the model version or the position in the batch
visible in the text itself.
"""

CHECKPOINT_FILE = "checkpoint.json"
"""Filename the dummy trainer writes and the dummy model reads back."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DummyTrainer(TrainTask):
    """A training task that pretends to train.

    Sleeps for ``train_seconds`` to stand in for compute, then writes a
    checkpoint recording what the round would have learned from. The sleep is
    the point: it makes a round take long enough that the surrounding
    machinery --- the ``RUNNING`` status, streams continuing to serve while a
    round is in flight, a stop request waiting for the round rather than
    cancelling it --- is actually observable.

    Parameters
    ----------
    train_seconds : float
        How long a round pretends to take.
    gpus, nodes : int
        Resources declared to the workflow engine, exactly as a real trainer
        would. Nothing uses them, but a backend smoke test wants them requested.
    name : str, optional
        Checkpoint directories are named after this. Defaults to ``"dummy"``.
    fail_every : Optional[int]
        Raise on every ``n``-th round. Used to exercise the training manager's
        failure handling; ``None`` (default) never fails.
    """

    def __init__(
        self,
        *,
        train_seconds: float = 2.0,
        gpus: int = 1,
        nodes: int = 1,
        name: Optional[str] = None,
        fail_every: Optional[int] = None,
    ):
        super().__init__(gpus=gpus, nodes=nodes, name=name or "dummy")
        self.train_seconds = train_seconds
        self.fail_every = fail_every
        self.rounds: List[Dict[str, Any]] = []
        """One entry per completed round: what it trained on and for how long."""

    def train(self, dataset, output_dir: str, **kwargs: Any) -> str:
        """Pretend to train on ``dataset``; return the checkpoint directory."""
        version = int(kwargs.get("model_version", len(self.rounds) + 1))
        samples = len(dataset)

        if self.fail_every and version % self.fail_every == 0:
            raise RuntimeError(
                f"{self.name}: simulated training failure on round {version}"
            )

        started = time.time()
        time.sleep(self.train_seconds)      # stand-in for the actual fine-tune
        elapsed = time.time() - started

        write_dummy_checkpoint(output_dir, version=version, samples=samples)
        self.rounds.append({
            "version": version,
            "samples": samples,
            "seconds": elapsed,
            "checkpoint": output_dir,
        })
        return output_dir


def write_dummy_checkpoint(output_dir: str, version: int, samples: int) -> str:
    """Write the placeholder checkpoint a dummy round produces."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    with open(path, "w") as fd:
        json.dump(
            {"version": version, "samples": samples, "trained_at": time.time()},
            fd,
            indent=2,
        )
    return path


def read_dummy_checkpoint(checkpoint_path: Optional[str]) -> Dict[str, Any]:
    """Read a dummy checkpoint, tolerating one that was never written.

    A stream starts before the first training round completes, so "no
    checkpoint yet" is the normal initial state, not an error: it reports
    version 0, which is what the untrained model is.
    """
    if not checkpoint_path:
        return {"version": 0, "samples": 0}
    path = checkpoint_path
    if os.path.isdir(path):
        path = os.path.join(path, CHECKPOINT_FILE)
    try:
        with open(path) as fd:
            return json.load(fd)
    except (OSError, ValueError):
        return {"version": 0, "samples": 0}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class DummyModel:
    """A model that generates a placeholder string.

    Reports the version it was loaded from, read out of the checkpoint file
    rather than passed in, so that a reload is observable from the output alone.

    Parameters
    ----------
    checkpoint_path : Optional[str]
        Where it was loaded from. ``None`` means untrained (version 0).
    template : str
        Output format. ``{uuid}`` is substituted with a fresh UUID per call;
        ``{version}`` and ``{index}`` are also available.
    latency : float
        Seconds one generation pretends to take.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        template: str = DEFAULT_TEMPLATE,
        latency: float = 0.0,
    ):
        self.checkpoint_path = checkpoint_path
        self.template = template
        self.latency = latency
        metadata = read_dummy_checkpoint(checkpoint_path)
        self.version = int(metadata.get("version", 0))
        self.trained_on = int(metadata.get("samples", 0))

    def generate(self, prompt: Any = None, index: int = 0) -> str:
        """Produce one placeholder output."""
        if self.latency:
            time.sleep(self.latency)
        return self.template.format(
            uuid=uuid.uuid4(), version=self.version, index=index, prompt=prompt
        )

    def generate_batch(self, prompts: List[Any]) -> List[str]:
        return [self.generate(p, i) for i, p in enumerate(prompts)]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<DummyModel v{self.version} trained_on={self.trained_on}>"


def dummy_load(
    checkpoint_path: Optional[str],
    ctx: Any = None,
    template: str = DEFAULT_TEMPLATE,
    latency: float = 0.0,
) -> DummyModel:
    """``load_func`` for a dummy inference stream.

    Called once at startup and again every time the training manager publishes
    a checkpoint. Pass ``template`` and ``latency`` through
    ``StreamConfig.load_kwargs``.
    """
    return DummyModel(checkpoint_path, template=template, latency=latency)


def dummy_infer(prompts: List[Any], ctx: Any) -> List[str]:
    """``process_func`` for a dummy inference stream.

    Returns one placeholder output per prompt, generated by whichever model
    version the stream currently holds.
    """
    model = ctx.model
    if model is None:
        # No checkpoint had been published when this replica started, so
        # load_func never ran. An untrained dummy is the right stand-in.
        model = DummyModel()
    return model.generate_batch(prompts)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def dummy_reward(outputs: List[Any], score: float = 1.0) -> List[Dict[str, Any]]:
    """``process_func`` for a dummy reward stream.

    Emits a corpus-shaped record per output, so a manager with this stream
    configured accumulates training data with no scoring code at all. Real
    reward functions vary the score; this one is deliberately constant, so a
    run's behaviour depends only on the plumbing under test.
    """
    return [{"completion": output, "score": score} for output in outputs]


__all__ = [
    "DummyTrainer",
    "DummyModel",
    "dummy_load",
    "dummy_infer",
    "dummy_reward",
    "write_dummy_checkpoint",
    "read_dummy_checkpoint",
    "DEFAULT_TEMPLATE",
    "CHECKPOINT_FILE",
]
