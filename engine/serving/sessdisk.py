# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The NVMe tier of the V4.1 session store (GLM 0250's design on V4.1 entries): park a session's pool pages and
bounded state to local NVMe, resume it after the pool needed the pages or after a restart.

One file an entry: ``<root>/<compat hash[:16]>/rank<r>/<key>.tfs`` =

    "DSV41SS1" | u64 header bytes | header JSON | zero pad to 4 KiB | segments, each 4 KiB aligned

segments: ``ids`` (int32), one per pool family (``pool/<family>``: the entry's pages' rows, uint8), one per bounded
array (``b/<name>``). The header names each segment's dtype, shape, offset, length and SHA-256; a read checks every
one (a damaged entry is deleted and the request prefills instead).

- Writes go to ``<key>.tmp`` then ``os.replace``: a crash leaves no half entry (``reconcile`` deletes ``.tmp``).
- O_DIRECT for data (aligned staging buffers, 8 MiB chunks), so parking never fills the page cache; where the
  filesystem refuses O_DIRECT (tmpfs in tests), buffered with the written range dropped from the page cache
  (GLM ``memsafe.drop_file_cache``).
- The compat ident (GLM ``sessdisk.compat_ident``'s fields for this family: image / code digest, knobs, layout, tag
  set, page) names the directory: a different build never reads another's entries.
- ``reconcile`` (load / restart): index the directory's valid entries, delete the rest. Two ranks: the engine
  intersects both ranks' ``keys()`` through its communicator and calls ``retain`` (so a plan never names an entry
  one rank lacks).
- Budget (``TF_DSV41_SESSION_DISK_GIB``): least recently used entries are deleted past it.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorfold.families.glm5_next.spark import memsafe

from .protocol import Bounded
from .sessions import chain, is_prefix

MAGIC = b"DSV41SS1"
ALIGN = 4096
CHUNK = 8 << 20
FORMAT = 1


def compat_hash(ident: dict) -> str:
    return hashlib.sha256(json.dumps(dict(ident, format=FORMAT), sort_keys=True).encode()).hexdigest()


def compat_ident(*, image: str, knobs: dict, layout: dict, extra: dict | None = None) -> dict:
    """What decides whether an entry's bytes mean the same thing to this build (GLM ``sessdisk.compat_ident``)."""

    out = {"format": FORMAT, "image": image, "knobs": {k: knobs[k] for k in sorted(knobs)}, "layout": layout}
    out.update(extra or {})
    return out


def knobs_from_env(prefix: str = "TF_DSV41_", skip: Sequence[str] = ()) -> dict:
    """The build's knobs that change bytes (every ``prefix`` variable but the operational ones in ``skip``)."""

    return {k: v for k, v in os.environ.items() if k.startswith(prefix) and k not in skip}


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


@dataclass
class DiskEntry:
    key: str
    tag: int
    ids: list[int]
    chain: list[bytes]
    kind: str
    size: int
    used: int

    @property
    def pos(self) -> int:
        return len(self.ids)


class _File:
    """A file opened O_DIRECT when the filesystem allows it."""

    def __init__(self, path: Path, write: bool, direct: bool) -> None:
        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if write else os.O_RDONLY
        self.direct = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT, 0o644)
                self.direct = True
                return
            except OSError:
                pass
        self.fd = os.open(path, flags, 0o644)

    def close(self) -> None:
        os.close(self.fd)


class DiskTier:
    def __init__(self, root: str | Path, rank: int, ident: dict, *, budget_gib: float = 64.0, min_tokens: int = 1024,
                 direct: bool = True, quiet: bool = True) -> None:
        self.dir = Path(root) / compat_hash(ident)[:16] / f"rank{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "compat.json").write_text(json.dumps(ident, sort_keys=True))
        self.budget = int(budget_gib * (1 << 30))
        self.min_tokens = int(min_tokens)
        self.direct = direct
        self.quiet = quiet
        self.index: dict[str, DiskEntry] = {}
        self.clock = 0
        self.lock = threading.Lock()
        self.stats = {"writes": 0, "reads": 0, "bytes_written": 0, "bytes_read": 0, "bad": 0, "evicted": 0}

    # policy
    def accepts(self, pos: int) -> bool:
        return pos >= self.min_tokens

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.tfs"

    # lookup
    def find(self, prompt: Sequence[int], tag: int, p_chain: list[bytes] | None = None) -> tuple[int, str] | None:
        p_chain = chain(prompt) if p_chain is None else p_chain
        best = None
        for e in self.index.values():
            if e.tag == tag and is_prefix(e.ids, e.chain, prompt, p_chain) and (best is None or e.pos > best[0]):
                best = (e.pos, e.key)
        return best

    def length(self, key: str) -> int:
        return self.index[key].pos

    def keys(self) -> list[str]:
        return sorted(self.index)

    def retain(self, keys) -> None:
        """Keep only ``keys`` (the entries every rank holds), deleting the others."""

        for k in [k for k in self.index if k not in set(keys)]:
            self.delete(k)

    # write
    def write(self, key: str, tag: int, ids: Sequence[int], kind: str, pages: dict[str, np.ndarray],
              bounded: Bounded, page: int) -> int:
        segs: list[tuple[str, np.ndarray]] = [("ids", np.asarray(ids, dtype=np.int32))]
        segs += [(f"pool/{name}", np.ascontiguousarray(a)) for name, a in sorted(pages.items())]
        segs += [(f"b/{name}", np.ascontiguousarray(a)) for name, a in sorted(bounded.arrays.items())]
        table, off = [], 0
        for name, a in segs:
            raw = a.view(np.uint8).reshape(-1)
            table.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape), "offset": off,
                          "nbytes": int(raw.size), "sha256": hashlib.sha256(raw).hexdigest()})
            off = _up(off + raw.size)
        head = {"key": key, "tag": int(tag), "kind": kind, "page": int(page), "pos": int(bounded.pos),
                "meta": bounded.meta, "segments": table}
        hj = json.dumps(head).encode()
        hdr = MAGIC + struct.pack("<Q", len(hj)) + hj
        base = _up(len(hdr))
        total = base + off
        tmp = self.dir / f"{key}.tmp"
        f = _File(tmp, True, self.direct)
        try:
            buf = mmap.mmap(-1, CHUNK)
            try:
                self._put(f, buf, 0, hdr)
                for (name, a), t in zip(segs, table):
                    self._put(f, buf, base + t["offset"], a.view(np.uint8).reshape(-1))
            finally:
                buf.close()
            if f.direct:
                os.ftruncate(f.fd, total)
            else:
                memsafe.drop_file_cache(f.fd)
        finally:
            f.close()
        os.replace(tmp, self.path(key))
        with self.lock:
            self.clock += 1
            self.index[key] = DiskEntry(key, int(tag), [int(t) for t in ids], chain(ids), kind, total, self.clock)
            self.stats["writes"] += 1
            self.stats["bytes_written"] += total
        self._trim()
        return total

    def _put(self, f: _File, buf: mmap.mmap, off: int, data) -> None:
        """``data`` (bytes / uint8 array) at ``off`` (4 KiB aligned), through the aligned staging buffer."""

        mv = memoryview(data).cast("B") if not isinstance(data, np.ndarray) else memoryview(data)
        n = len(mv)
        done = 0
        while done < n:
            k = min(CHUNK, n - done)
            buf.seek(0)
            buf.write(mv[done:done + k])
            span = _up(k) if f.direct else k
            if span > k:
                buf.write(b"\0" * (span - k))
            view = memoryview(buf)[:span]
            try:
                wrote = os.pwritev(f.fd, [view], off + done)
            finally:
                view.release()
            if wrote != span:
                raise OSError(f"short write ({wrote} of {span})")
            done += k

    # read
    def _header(self, path: Path) -> tuple[dict, int]:
        with open(path, "rb") as fh:
            first = fh.read(16)
            if first[:8] != MAGIC:
                raise ValueError("not a V4.1 session entry")
            (n,) = struct.unpack("<Q", first[8:16])
            head = json.loads(fh.read(n))
        return head, _up(16 + n)

    def read(self, key: str) -> tuple[list[int], dict[str, np.ndarray], Bounded]:
        """(ids, pool pages by family, bounded state) of an entry, every segment checked; ValueError when damaged
        (the entry is deleted)."""

        path = self.path(key)
        try:
            head, base = self._header(path)
            size = os.path.getsize(path)
            f = _File(path, False, self.direct)
            try:
                span = _up(size)
                m = mmap.mmap(-1, span)
                try:
                    view = memoryview(m)
                    got = 0
                    while got < size:
                        k = min(CHUNK * 8, span - got)
                        sub = view[got:got + k]
                        r = os.preadv(f.fd, [sub], got)
                        sub.release()
                        if r <= 0:
                            break
                        got += r
                    view.release()
                    if got < size:
                        raise ValueError(f"short read ({got} of {size})")
                    raw = np.frombuffer(m, dtype=np.uint8, count=size).copy()
                finally:
                    m.close()
                if not f.direct:
                    memsafe.drop_file_cache(f.fd, sync=False)
            finally:
                f.close()
            out: dict[str, np.ndarray] = {}
            for t in head["segments"]:
                a = raw[base + t["offset"]: base + t["offset"] + t["nbytes"]]
                if hashlib.sha256(a).hexdigest() != t["sha256"]:
                    raise ValueError(f"segment {t['name']}: checksum mismatch")
                out[t["name"]] = a.view(np.dtype(t["dtype"])).reshape(t["shape"])
        except (OSError, ValueError, KeyError) as exc:
            self.stats["bad"] += 1
            self.delete(key)
            raise ValueError(f"session entry {key}: {exc}") from None
        ids = [int(t) for t in out.pop("ids").tolist()]
        pages = {k[5:]: v for k, v in out.items() if k.startswith("pool/")}
        arrays = {k[2:]: v for k, v in out.items() if k.startswith("b/")}
        with self.lock:
            self.clock += 1
            if key in self.index:
                self.index[key].used = self.clock
            self.stats["reads"] += 1
            self.stats["bytes_read"] += size
        return ids, pages, Bounded(int(head["pos"]), arrays, dict(head.get("meta") or {}))

    # upkeep
    def delete(self, key: str) -> None:
        with self.lock:
            self.index.pop(key, None)
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            pass

    def _trim(self) -> None:
        while sum(e.size for e in self.index.values()) > self.budget and len(self.index) > 1:
            victim = min(self.index.values(), key=lambda e: e.used)
            self.delete(victim.key)
            self.stats["evicted"] += 1

    def reconcile(self) -> int:
        """Index the directory's entries (oldest modified first in LRU order), deleting temporary and unreadable
        files. Returns the count indexed."""

        self.index.clear()
        found = []
        for p in self.dir.iterdir():
            if p.suffix == ".tmp":
                p.unlink()
                continue
            if p.suffix != ".tfs":
                continue
            try:
                head, base = self._header(p)
                ids_t = next(t for t in head["segments"] if t["name"] == "ids")
                with open(p, "rb") as fh:
                    fh.seek(base + ids_t["offset"])
                    raw = fh.read(ids_t["nbytes"])
                if hashlib.sha256(raw).hexdigest() != ids_t["sha256"] or head["key"] != p.stem:
                    raise ValueError("ids checksum / key")
                end = base + max(t["offset"] + t["nbytes"] for t in head["segments"])
                if os.path.getsize(p) < end:
                    raise ValueError("truncated")
                ids = np.frombuffer(raw, dtype=np.int32).tolist()
                found.append((p.stat().st_mtime_ns, head, ids, os.path.getsize(p)))
            except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
                self.stats["bad"] += 1
                p.unlink()
        for _, head, ids, size in sorted(found, key=lambda x: x[0]):
            self.clock += 1
            self.index[head["key"]] = DiskEntry(head["key"], int(head["tag"]), ids, chain(ids), head["kind"], size,
                                                self.clock)
        self._trim()
        return len(self.index)

    def describe(self) -> str:
        used = sum(e.size for e in self.index.values())
        return f"NVMe tier {self.dir}: {len(self.index)} entries, {used / 2**30:.2f} of {self.budget / 2**30:.0f} GiB"


def from_env(rank: int, ident: dict) -> DiskTier | None:
    """TF_DSV41_SESSION_DISK=<dir> (unset: no NVMe tier), TF_DSV41_SESSION_DISK_GIB (64), TF_DSV41_SESSION_DISK_MIN
    (1024 tokens: shorter entries are dropped, not parked)."""

    root = os.environ.get("TF_DSV41_SESSION_DISK", "").strip()
    if not root:
        return None
    gib = float(os.environ.get("TF_DSV41_SESSION_DISK_GIB", "64") or 64)
    low = int(os.environ.get("TF_DSV41_SESSION_DISK_MIN", "1024") or 1024)
    tier = DiskTier(root, rank, ident, budget_gib=gib, min_tokens=low)
    tier.reconcile()
    return tier


__all__ = ["DiskEntry", "DiskTier", "compat_hash", "compat_ident", "from_env", "knobs_from_env"]
