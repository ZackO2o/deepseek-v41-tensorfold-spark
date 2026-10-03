# Operations

## What a start does (`scripts/serve.sh start`)

1. Takes a lock (one start at a time; the watchdog stands down while it is held) and clears the stop marker.
2. Refuses if a CUDA process runs on either node.
3. Preflight (`PREFLIGHT=strict`): ssh to the worker; the image on both nodes with equal content keys (creation time
   + layer digests); `config.json` / tokenizer in both model folders; both Engram shards on each node; a prepared
   folder (warning only); the RoCE ports ACTIVE; nothing on `PORT` / `MASTER_PORT`; the `/cache/roce-failed` marker.
4. Drops the page cache on both nodes and waits until MemFree >= `MEM_GATE_GIB` (104) on both (`MEM_GATE_TIMEOUT`).
   On GB10 the page cache is GPU memory: a load next to a full page cache can start with fewer request slots.
5. Starts rank 1 on the worker, then rank 0 here (`python -m tensorfold serve /model --tp 2 ...`), and waits for
   `/v1/models` (`READY_TIMEOUT`).
6. Checks rank 0's `serving: N slot(s)` line against `PARALLEL` (one retry after a cache drop if short).
7. Drops the caches again, runs `scripts/canary.py` (a greedy chat, a thinking reply with separated reasoning, a forced
   tool call, a JSON-schema reply, `/tokenize`), drops the caches once more. `CANARY=strict` stops both ranks on a
   failed canary and saves their last logs in `STATE_DIR`.

`scripts/serve.sh args 0|1` prints the exact `docker run` arguments without starting anything.

## Knobs

Every `TF_DSV41_*`, `GLM53_TF_*`, `MALLOC_*` and `MIMALLOC_*` line of the config reaches both ranks. A non-empty caller export wins
over the file: `CONTEXT=196608 scripts/serve.sh restart`.

| knob | production | off / fallback |
| --- | --- | --- |
| `TF_DSV41_PREFILL` | `replay` (~2x prefill) | `full`: the exact prefill, ~kit speed |
| `TF_DSV41_PREFILL_KERNELS` | `fast` | `exact` |
| `TF_DSV41_FAST_EXPERTS` | `gm` | `grouped`; `tc` is 3-5x slower |
| `TF_DSV41_EXPERT_TOPP` / `_MIN_K` / `_RENORM` | 0.85 / 3 / kept (lossy, +5%) | delete the three lines: the unpruned model |
| `TF_DSV41_MHC_FN` | `bf16` | unset: fp32 mixing weights (prose -2.4%) |
| `TF_DSV41_L2PF` / `_MB` | 1 / 12 | 0 |
| `TF_DSV41_GRAPH_MODE` | `rows` | `mix`: C2 / C4 slower |
| `TF_DSV41_ROUTER` | `gemv` | `fused` / `split` |
| `GLM53_TF_COMM_BACKEND` | `roce` | `nccl` (-11% / -17%). A RoCE failure at run time writes `/cache/roce-failed` in the cache volume and the next start of both ranks uses NCCL; delete it to retry RoCE |
| `CONTEXT` | 300000 (4 x 300K) | 196608 keeps the worker further from its memory floor |
| `TF_DSV41_THINKING` / `_DEFAULT_EFFORT` | 1 / `high` (75) | requests override |
| `TF_DSV41_GRAMMAR` | 1 | 0: `response_format` and strict tools ignored |
| `MIMALLOC_ALLOW_THP` | 0: no transparent huge pages in torch's CPU allocator (an embedded mimalloc), read at process start | unset: mimalloc's default (on); long prefills then stall in synchronous memory compaction (G12) |

Knobs left at their defaults that you may meet in the code: `TF_DSV41_DEPTH` / `TF_DSV41_DRAFT_DEPTH` (draft depth:
the default policy reads the boot calibration and caps at 5; `static` + 3 is the kit's), `TF_DSV41_DRAFT_HEAD` (full),
`TF_DSV41_VERIFY_BUDGET` (unset; lossy, do not use), `TF_DSV41_TREE_PC` (0), `TF_DSV41_DSPARK_DELTA` (unset).

Host memory and stalls (G12; defaults in brackets):

- `TF_DSV41_STALL_S` (300; 0 = off): a round that runs longer prints a stall report on both ranks (rank, round, phase,
  the round's plan, the forward's waits, every thread's stack), again every 4x that. `TF_DSV41_STALL_EXIT_S` (0 =
  never) exits the process after that long in one round.
- `TF_DSV41_ENGRAM_WAIT_S` (300; 0 = none): deadline on an Engram read and on the Engram gate's worker.
- `TF_DSV41_NUMPY_HUGEPAGE` (0): NumPy's huge-page request on its large buffers, off on the ranks.
- `TF_DSV41_HOST_BUFFERS` (6): reused NumPy buffers for a prompt segment's Engram rows.
- `TF_DSV41_HASH_MEMO_ROWS` (16384): rows the Engram hash memo holds. `TF_DSV41_HOST_TRIM_ROWS` (32768; 0 = off):
  `malloc_trim` every that many prompt rows (it reaches glibc only; harmless).
- `TF_DSV41_MEMPROBE=N` (0 = off): one memory line a rank every N rounds (RssAnon / RssShmem, MemAvailable, torch's
  allocator); `TF_DSV41_MEMPROBE_TENSORS=K` and `_SMAPS=1` add live CPU tensors and anonymous RSS by mapping (slow,
  diagnostics only).

## Memory

- **Before a load:** MemFree >= 104 GiB on both nodes.
- **While serving:** `TF_DSV41_FLOOR_GIB=5` is the MemAvailable target the load-time budget plans for;
  `TF_DSV41_FLOOR_HARD_GIB=4` is the level below which nothing new is admitted.
- **Measured:** decode benchmarks >= 7.3 GiB on the worker. In the 4 x 300K stress the worker's minimum is 3.0-3.7
  GiB, reached in the first ~20K rows of the 299K prefill. That floor comes from the boot budget (4 x 300K KV pool)
  and the first segments' device-side reservations, not from host growth: since G12 each rank's anonymous RSS
  grows ~0.7 GiB over the prompt (was 4-5.6). Open; [RESULTS.md](RESULTS.md) section 5. `CONTEXT=196608` lowers
  it.
- Keep other workloads (desktop sessions, other containers, a second model) off both nodes.

## Watchdog and start at boot

- `scripts/serve.sh watch --once` (the `dsv41-tf-watchdog.timer` tick, every minute): a tick is bad when a rank
  exited or `/health` fails. `WATCH_FAILS` (3) bad ticks in a row restart both ranks in the background
  (`WATCH_HEAL=1`), at most once every `WATCH_MIN_HEAL` s (1800). It stands down while a start runs, while rank 0 is
  younger than `WATCH_GRACE` (1800 s), after a deliberate `serve.sh stop` during the same boot, and while
  `WATCH_LEASE` (an optional file you touch while benchmarking or maintaining the pair) is fresh. `WATCH_ALERT=<cmd>`
  is called with a message on every alert.
- `scripts/boot-start.sh` (the `dsv41-boot-start.service` oneshot): after a reboot, starts the service if it was
  serving (or crashed) before, once docker, the worker and the RoCE ports are up; `BOOT_DRY_RUN=1` shows the
  decision. The watchdog is the fallback.
- Before benchmarks with `scripts/serve.sh run`, stop the server and the watchdog timer
  (`systemctl --user stop dsv41-tf-watchdog.timer`), or set `WATCH_LEASE` and touch the file.

## Logs and state

- `docker logs dsv41-tf-r0` / `scripts/serve.sh logs 1`; the `[boot]` lines time each start phase.
- `HEAD_STATE` / `WORKER_STATE`: the request log (`requests.jsonl`) and the session NVMe tier (`sessions/`).
- `STATE_DIR` (default `~/.local/state/dsv41-tf`): the start lock, watchdog counters, stop marker, heal and
  boot-start logs, canary-failure logs.
