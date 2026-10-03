"""How DeepSeek-V4.1-Flash plugs into our GLM serving stack on TensorFold 0.6.0.

Determined and tested on CPU: ``topology`` (layer roles), ``pool`` (paged cache families, shared pages), ``state``
(a slot's bounded state, snapshots), ``sessions`` / ``sessdisk`` (RAM and NVMe tiers, resumed == fresh), ``engram``
(the NVMe row reader), ``memory`` (the per-node budget and the run-time floor), ``protocol`` (the ``Forward`` the
serving layer needs from M1), ``plan`` / ``rounds`` / ``batch`` (the batcher: batched == alone), ``stack`` (one call
builds a rank's serving stack), ``encoding`` / ``dsml`` / ``structured`` / ``app`` (the OpenAI server: DeepSeek's
encoding, DSML tool calls, structured output). Contracts still to fill: ``family``, ``engine`` (M1), ``drafting``
(M2). No torch at import of the pure modules."""
