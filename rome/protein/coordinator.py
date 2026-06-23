"""Adaptive coordinator — parent + child pipeline lifecycle (IMPRESS analog).

Owns the live set of :class:`ProteinBindingPipeline` instances. Two channels:

* ``submit_queue`` — new and child pipelines waiting to start
* ``complete_queue`` — pipelines that finished (drained iter_seqs or hit
  ``max_cycles``)

Sub-pipeline spawn requests come from the flow's per-cycle adaptive step
via :meth:`submit_child_pipeline_request`. The coordinator does not run tasks
itself; it hands pipelines to ``ProteinBindingFlow`` which schedules the
inner loop on the asyncflow backend.
"""

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rome.protein.pipeline import ProteinBindingPipeline


class AdaptiveCoordinator:
    def __init__(self):
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._complete_queue: asyncio.Queue = asyncio.Queue()
        self._active: Dict[str, ProteinBindingPipeline] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def submit(self, pipeline: ProteinBindingPipeline) -> None:
        self._active[pipeline.pipeline_id] = pipeline
        await self._submit_queue.put(pipeline)

    async def next_to_run(self) -> ProteinBindingPipeline:
        return await self._submit_queue.get()

    async def mark_complete(self, pipeline_id: str) -> None:
        self._active.pop(pipeline_id, None)
        await self._complete_queue.put(pipeline_id)

    def submit_child_pipeline_request(
        self,
        parent: ProteinBindingPipeline,
        migrated_backbones: Dict[str, Any],
    ) -> ProteinBindingPipeline:
        """Build a child pipeline carrying the migrated backbones forward.

        The child inherits ``previous_scores`` for its backbones (so its
        first-pass criterion can detect immediate degradation against the
        parent's last state) and advances ``sub_order``.
        """
        child_id = f"{parent.pipeline_id}_sub{parent.sub_order + 1}_{uuid.uuid4().hex[:6]}"
        child = ProteinBindingPipeline(
            pipeline_id=child_id,
            base_path=parent.base_path,
            backbones={bid: spec for bid, spec in migrated_backbones.items()},
            is_child=True,
            start_cycle=parent.passes,
            passes=parent.passes,
            sub_order=parent.sub_order + 1,
            seq_rank=parent.seq_rank + 1,
            previous_scores={
                bid: dict(parent.current_scores.get(bid, {}))
                for bid in migrated_backbones
            },
            iter_seqs=dict(migrated_backbones),
        )
        child.set_up_dirs()
        parent.copy_pdbs_into(child.input_path, list(migrated_backbones.keys()))
        return child

    async def drain(self) -> None:
        """Block until no pipelines are active and no submissions are pending."""
        while self._active or not self._submit_queue.empty():
            await asyncio.sleep(0.1)
