"""CSA2 for DeepSeek-V4.1-Flash on sm_121 (Triton): the FP8 KV row (``rows``), the compressors and row stores
(``compress``), the lightning indexer with the hierarchical candidates (``index``), and sparse + sliding-window
attention with sinks (``attn``). Row-invariant throughout; ``ref`` is the torch reference the tests compare with."""
