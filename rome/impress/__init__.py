"""ROME-under-IMPRESS shim layer.

This subpackage lets ROME plug into an existing IMPRESS workflow without
modifying it. The IMPRESS pipeline (``ProteinBindingPipeline``,
``ImpressManager``, the ``adaptive_fn`` callback) is unchanged; ROME
attaches as an additive layer that:

* sweeps per-pass score CSVs IMPRESS writes (``af_stats_<name>_pass_<N>.csv``)
  into a score-gated training corpus,
* fires a ProteinMPNN fine-tuning round when the corpus crosses a
  configurable threshold,
* tracks ``model_version`` + the current checkpoint path so downstream
  IMPRESS passes can pick up new weights (when the pipeline class is
  extended to consume them).

The minimum-viable shim is **passive**: it only does work when
``wrap_adaptive_fn(...)`` is called. Streaming MPNN generation and
mid-cycle weight hot-reload need a small subclass of IMPRESS's
``ProteinBindingPipeline`` and are deferred — see
``RomeStreamingPipeline`` (TBD).
"""

from rome.impress.shim import CorpusThresholds, RomeShim

__all__ = ["CorpusThresholds", "RomeShim"]
