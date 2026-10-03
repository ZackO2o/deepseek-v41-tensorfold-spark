"""The V4.1 batcher on a fake forward: batched == alone (4 slots, drafts on and off), resumed == fresh (RAM, NVMe,
across a restart, both CED modes), pool spills, two ranks agreeing, cancellation, the plan codec."""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest
from fake_forward import FakeForward
from tensorfold.engine.exact_sampling import Sampling

from engine.serving import sessdisk
from engine.serving.batch import Batcher, Job
from engine.serving.plan import Admit, Plan, WindowSpec, piece_end
from engine.serving.pool import Pool
from engine.serving.sessions import Store, Tag

VOCAB = 4096


def make(topo, *, n=4, pool_tokens=16384, capacity=4096, store=False, ram=1 << 30, disk=None, mode="full",
         depth=0, rows=256, eos_bias=0.0, link=None, rank=0):
    pool = Pool(topo, pool_tokens)
    st = None
    if store:
        tier = None
        if disk is not None:
            tier = sessdisk.DiskTier(disk, rank, {"test": "fake", "mode": mode}, budget_gib=1, min_tokens=16)
            tier.reconcile()
        st = Store(pool, ram_bytes=ram, disk=tier)
    fwd = FakeForward(topo, pool, n, VOCAB, eos_bias=eos_bias)
    fwd.mode = mode
    b = Batcher(fwd, pool, n_slots=n, capacity=capacity, store=st, mode=mode, tag=Tag(ced=mode).code(),
                prefill_rows=rows, depth=depth, session_min=16, start=False, link=link, rank=rank)
    return b


def drain(job: Job) -> tuple[list[int], dict]:
    toks = []
    while True:
        item = job.out.get_nowait()
        if item is None:
            return toks, job.stats
        if isinstance(item, BaseException):
            raise item
        toks += item


def run(b: Batcher, jobs: list[Job]) -> list[list[int]]:
    for j in jobs:
        b.submit(j)
    for _ in range(100000):
        if not b.step():
            break
    return [drain(j)[0] for j in jobs]


def prompts(seed=0, lens=(37, 300, 700, 1200)):
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(2, VOCAB, size=n)] for n in lens]


SAMPLINGS = [None, Sampling(7, 0.8, 20, 0.9), Sampling(11, 1.0, 0, 0.95), None]


def jobs_for(ps, max_tokens=40, sampling=SAMPLINGS):
    return [Job(list(p), max_tokens, s) for p, s in zip(ps, sampling)]


@pytest.mark.parametrize("depth", [0, 3])
def test_batched_equals_alone(topo, depth):
    ps = prompts()
    alone = [run(make(topo, n=1, depth=0), [j])[0] for j in jobs_for(ps)]
    b = make(topo, n=4, depth=depth, rows=200)
    batched = run(b, jobs_for(ps))
    assert batched == alone
    assert all(len(t) == 40 for t in batched)
    if depth:
        assert b.counts["drafts_kept"] > 0                   # the oracle drafts were used, the reply did not change
    assert b.pool.alloc.n_free == b.pool.npages           # every page back
    assert b.pool.null_clean()


def test_eos_and_max_tokens(topo):
    b = make(topo, n=2, eos_bias=4.0)
    out = run(b, jobs_for(prompts(lens=(50, 60)), max_tokens=200, sampling=[None, Sampling(3, 1.0, 20, 0.95)]))
    for t in out:
        assert t[-1] == 1 and 1 not in t[:-1]                # ends at the first EOS


@pytest.mark.parametrize("mode", ["full", "replay"])
def test_resumed_equals_fresh_ram(topo, mode):
    a = prompts(1, (900,))[0]
    b = make(topo, store=True, mode=mode, depth=2)
    r1 = run(b, [Job(list(a), 30, Sampling(5, 0.7, 20, 0.95))])[0]
    turn2 = a + r1 + prompts(2, (150,))[0]
    j2 = Job(list(turn2), 30, Sampling(9, 0.7, 20, 0.95))
    got = run(b, [j2])[0]
    assert j2.stats["tier"] == "ram" and j2.stats["cached"] == len(a) + len(r1) - 1     # the turn-end entry
    fresh = run(make(topo, mode=mode), [Job(list(turn2), 30, Sampling(9, 0.7, 20, 0.95))])[0]
    assert got == fresh
    j3 = Job(list(a), 30, Sampling(5, 0.7, 20, 0.95))                                    # an identical resend
    assert run(b, [j3])[0] == r1
    assert j3.stats["cached"] == (len(a) - 1) // 16 * 16                                # the replay point (0540)
    assert b.store.stats["hits_ram"] == 2
    # every page held by the store's entries only; the slots are empty
    assert all(not s.pages for s in b.ex.slots) and b.pool.entry_pages() > 0


@pytest.mark.parametrize("mode", ["full", "replay"])
def test_resumed_equals_fresh_nvme_restart(topo, tmp_path, mode):
    a = prompts(3, (700,))[0]
    b = make(topo, store=True, disk=tmp_path, mode=mode)
    r1 = run(b, [Job(list(a), 25, None)])[0]
    for key in list(b.store.entries):
        b.store.park(key)
    assert b.store.stats["parks"] == 2 and b.pool.alloc.n_free == b.pool.npages
    turn2 = a + r1 + prompts(4, (90,))[0]
    # a restart: new pool, store, forward, batcher; the NVMe tier reconciled from the directory
    b2 = make(topo, store=True, disk=tmp_path, mode=mode)
    assert len(b2.store.disk.index) == 2
    j = Job(list(turn2), 25, Sampling(13, 0.9, 40, 0.9))
    got = run(b2, [j])[0]
    assert j.stats["tier"] == "disk" and j.stats["cached"] == len(a) + len(r1) - 1
    fresh = run(make(topo, mode=mode), [Job(list(turn2), 25, Sampling(13, 0.9, 40, 0.9))])[0]
    assert got == fresh


def test_damaged_nvme_entry_prefills(topo, tmp_path):
    a = prompts(5, (400,))[0]
    b = make(topo, store=True, disk=tmp_path)
    r1 = run(b, [Job(list(a), 10, None)])[0]
    for key in list(b.store.entries):
        b.store.park(key)
    for p in tmp_path.rglob("*.tfs"):                       # flip a byte in every entry's last segment
        data = bytearray(p.read_bytes())
        data[-5] ^= 0xFF
        p.write_bytes(bytes(data))
    b2 = make(topo, store=True, disk=tmp_path)
    turn2 = a + r1 + [5, 6, 7]
    j = Job(list(turn2), 10, None)
    got = run(b2, [j])[0]
    assert j.stats["tier"] == "none" and j.stats["cached"] == 0
    assert got == run(make(topo), [Job(list(turn2), 10, None)])[0]
    assert b2.store.disk.stats["bad"] >= 1


def test_pool_spills_park_sessions(topo, tmp_path):
    # a pool of 16 pages: four 1,100-token conversations do not all stay mapped; the coldest are parked
    b = make(topo, n=2, pool_tokens=16 * 256, store=True, disk=tmp_path)
    ps = prompts(6, (1100, 1100, 1100, 1100))
    first = run(b, jobs_for(ps, 20, [None] * 4))
    assert b.store.stats["parks"] > 0
    again = [Job(p + r + [9, 9], 20, None) for p, r in zip(ps, first)]
    out = run(b, again)
    fresh = run(make(topo, n=1, pool_tokens=16 * 256), [Job(p + r + [9, 9], 20, None) for p, r in zip(ps, first)])
    assert out == fresh
    assert {j.stats["tier"] for j in again} >= {"disk"}


class PairLink:
    def __init__(self):
        self.q = queue.Queue()

    def send(self, plan):
        self.q.put(list(plan))

    def recv(self):
        return self.q.get(timeout=30)


def test_two_ranks_agree(topo, tmp_path):
    link = PairLink()
    r0 = make(topo, store=True, disk=tmp_path / "r0", depth=3, link=link, rank=0)
    r1 = make(topo, store=True, disk=tmp_path / "r1", depth=3, link=link, rank=1)
    t = threading.Thread(target=r1.follow, daemon=True)
    t.start()
    ps = prompts(7)
    out = run(r0, jobs_for(ps, 30))
    turn2 = [Job(p + o + [3, 4, 5], 20, s) for p, o, s in zip(ps, out, SAMPLINGS)]
    run(r0, turn2)
    for key in list(r0.store.entries)[:2]:                 # a park, as a spill plan would carry it
        r0.store.park(key)
        r1.store.park(key)
    r0.send_stop()
    t.join(timeout=30)
    assert not t.is_alive()
    assert r0.fwd.calls == r1.fwd.calls
    assert r0.fwd.digest() == r1.fwd.digest()
    assert sorted(r0.store.entries) == sorted(r1.store.entries)
    assert r0.store.disk.keys() == r1.store.disk.keys()


def test_cancellation(topo):
    b = Batcher(*_threaded(topo), n_slots=2, capacity=4096, start=True, session_min=16, prefill_rows=64)
    try:
        seen = []

        def stop_after_5(new):
            seen.extend(new)
            return len(seen) >= 5

        stats = b.generate(prompts(8, (200,))[0], 500, None, stop_after_5)
        assert stats["finish"] == "cancelled" and 5 <= len(seen) < 500

        gone = threading.Event()

        def never(new):
            return False

        never.cancelled = gone.is_set                      # the client leaves during a long prefill
        threading.Timer(0.2, gone.set).start()
        stats = b.generate(prompts(9, (3000,))[0], 50, None, never)
        assert stats["finish"] == "cancelled" and stats["completion"] < 50
        for _ in range(100):
            if not b.busy():
                break
            threading.Event().wait(0.05)
        assert b.pool.alloc.n_free == b.pool.npages and all(s is None for s in b.seqs)
        out = []
        b.generate(prompts(10, (64,))[0], 8, None, lambda new: out.extend(new) or False)
        assert len(out) == 8                               # the engine serves on after cancellations
    finally:
        b.stop()


def _threaded(topo):
    pool = Pool(topo, 16384)
    return FakeForward(topo, pool, 2, VOCAB), pool


def test_cancel_while_queued(topo):
    b = make(topo, n=1)
    j1, j2 = Job(prompts(11, (100,))[0], 30, None), Job(prompts(12, (100,))[0], 30, None)
    b.submit(j1)
    b.submit(j2)
    b.step()
    b.cancel(j2)
    while b.step():
        pass
    assert len(drain(j1)[0]) == 30
    toks, st = drain(j2)
    assert toks == [] and st["finish"] == "cancelled"


def test_plan_codec_and_pieces():
    p = Plan(round=3, mode="replay", count=28, commits=[(0, 2, 77)], finishes=[(1, True, False)], spills=["ab"],
             admits=[Admit(2, [5, 6, 7], 10, "ram", "k", 2, 3, [1, 2, 3], 9)], pieces=[(2, 2, 2)], saves=[2],
             finals=[2], windows=[WindowSpec(0, 40, 77, 3), WindowSpec(2, 2, 7, 0, [8, 9])])
    q = Plan.decode(p.encode())
    assert q == p
    with pytest.raises(ValueError):
        Plan.decode(p.encode() + [1])
    assert piece_end(0, 1000, 2048, 16) == 1000
    assert piece_end(0, 5000, 2048, 16) == 2048
    assert piece_end(10, 5000, 100, 16) == 96
    assert piece_end(10, 5000, 3, 16) == 16                   # at least to the next grid point


def test_stack_build(topo, tmp_path, monkeypatch):
    from engine.serving import stack

    monkeypatch.setenv("TF_DSV41_POOL_TOKENS", "8192")
    monkeypatch.setenv("TF_DSV41_SESSION_DISK", str(tmp_path))
    monkeypatch.setenv("TF_DSV41_GRAMMAR", "0")
    monkeypatch.setenv("TF_DSV41_FLOOR_GIB", "0.5")             # the workstation, not a Spark
    monkeypatch.setenv("TF_DSV41_FLOOR_HARD_GIB", "0.25")
    fwd = FakeForward(topo, Pool(topo, 256), 2, VOCAB)        # its pool is replaced by the stack's below
    b, host = stack.build(fwd, topo, rank=0, link=None, model_dir=None, eos=(1,), slots=2, context=2048,
                          device="cpu", image="test", quiet=True)
    try:
        fwd.pool = b.pool
        assert b.store is not None and b.store.disk is not None and host.grammars is None
        out = []
        b.generate(prompts(13, (300,))[0], 12, None, lambda new: out.extend(new) or False)
        for _ in range(200):                    # the turn-end snapshot runs in the round after the reply ends
            if not b.busy():
                break
            threading.Event().wait(0.02)
        assert len(out) == 12 and b.store.stats["saves"] == 2
    finally:
        b.stop()
