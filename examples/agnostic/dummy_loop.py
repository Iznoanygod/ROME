"""ROME-A end to end with no model: a dummy inference stream and a dummy trainer.

The smallest complete demonstration of the closed loop. An inference stream
serves placeholder outputs, the workflow scores them into the corpus, and once
enough accumulate the training manager runs a round that sleeps instead of
fine-tuning --- and the *same running stream* starts answering from the new
checkpoint, without this script orchestrating the handover.

What to watch in the output: ``model v`` climbs from 0, and it climbs while the
stream keeps serving. The stream is never restarted.

    $ dragon examples/agnostic/dummy_loop.py

Everything real about this run is the machinery: tasks are placed by the
workflow engine, state crosses the DDict, the checkpoint is a file that is
genuinely written and genuinely read back. Only the model and the gradient
steps are fake, which makes this the right thing to run first on a new backend
or a new allocation.
"""

import asyncio
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import rome
from rome.dummy import DummyTrainer, dummy_infer, dummy_load

ROUNDS = 12
PROMPTS_PER_ROUND = 4


async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)
    checkpoint_dir = tempfile.mkdtemp(prefix="rome_dummy_")

    manager = rome.Manager(
        flow,
        # One iteration's worth of prompts is enough for a round, so the
        # first swap lands within a couple of iterations and every second
        # iteration after that triggers another.
        data_config=rome.DataConfig(min_samples=PROMPTS_PER_ROUND),
        trainer_config=rome.TrainerConfig(
            trainer=DummyTrainer(train_seconds=1.0, gpus=1),
            checkpoint_dir=checkpoint_dir,
            poll_interval=0.5,
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer",
                kind=rome.StreamKind.INFERENCE,
                load_func=dummy_load,
                process_func=dummy_infer,
                load_kwargs={"latency": 0.05},   # pretend generation cost
                num_streams=2,
                num_gpus=1,
                batch_size=PROMPTS_PER_ROUND,
                poll_interval=0.05,
            )
        ],
    )
    await manager.start()

    print(f"checkpoints -> {checkpoint_dir}\n")

    try:
        for round_index in range(ROUNDS):
            # Ask the inference stream for a batch of outputs.
            request_ids = manager.stream.submit_batch(
                [f"prompt-{round_index}-{i}" for i in range(PROMPTS_PER_ROUND)]
            )

            # Collect them and score them into the corpus. In a real campaign
            # the score comes from a simulation or a reward model; here it is
            # noise, because nothing downstream depends on its value.
            for request_id in request_ids:
                record = await manager.stream.get_output(request_id, timeout=30.0)
                if record is None:
                    print(f"  request {request_id[:8]} timed out")
                    continue
                manager.add_training_data(
                    completion=record["result"],
                    score=random.random(),
                    produced_by_version=record["model_version"],
                )
                print(f"  v{record['model_version']} | {record['result']}")

            print(
                f"round {round_index}: corpus {manager.data.total_count:3d} "
                f"({manager.data.unconsumed_count} fresh) | "
                f"model v{manager.model_version} | "
                f"{manager.get_training_status().name}\n"
            )

            # A round takes 1s of "compute", so this gives the training
            # manager time to notice the threshold, run, and publish. Nothing
            # here waits on training: the stream keeps serving throughout, and
            # picks up the new checkpoint between batches on its own.
            await asyncio.sleep(1.0)

        print("final report:", manager.report())
    finally:
        # Stops the trainer first, waiting out any in-flight round rather than
        # cancelling it, then drains and stops the streams.
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
