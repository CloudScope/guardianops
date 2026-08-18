"""Hash-chained JSONL audit ledger with group commit.

"Immutable" in front of an auditor means two things: the storage refuses
overwrites (S3 Object Lock, WORM), and the records prove their own ordering.
This module does the second half. Each record commits to its predecessor, so
removing or editing any line invalidates every hash after it, and ``verify``
finds the exact line where the chain broke.

Durability is separated from ordering. Records are appended in order and become
visible to any reader immediately; ``fsync`` only decides what survives a power
cut. Syncing every record cost 25x the policy decision it was recording, so the
default syncs the records that carry a governance decision -- a block, a hold,
a run boundary -- and batches the rest. Ordering, and therefore chain integrity,
is unaffected by that choice.

On macOS plain ``fsync`` does not actually flush the drive's write cache;
``F_FULLFSYNC`` does. Critical records use it where available, so paying the
cost buys the guarantee it looks like it is buying.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64

# Sync policies.
ALWAYS = "always"        # fsync every record; slowest, strongest
CRITICAL = "critical"    # fsync governance decisions, batch the rest (default)
INTERVAL = "interval"    # batch everything, bounded by time and count

# Events that are always worth a sync: losing one loses the record of a decision.
_CRITICAL_EVENTS = frozenset({"run.start", "run.end", "agent.end", "session.initialize"})

_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", None)


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _chain(prev_hash: str, record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(prev_hash.encode("ascii") + _canonical(body)).hexdigest()


class AuditLog:
    def __init__(
        self,
        path: str,
        run_id: str,
        *,
        sync_mode: str = CRITICAL,
        batch_size: int = 64,
        batch_seconds: float = 2.0,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.sync_mode = sync_mode
        self.batch_size = batch_size
        self.batch_seconds = batch_seconds
        self.seq = 0
        self.prev_hash = GENESIS
        self._unsynced = 0
        self._last_sync = time.monotonic()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._resume()
        # One handle for the process, rather than an open/close per record.
        self._fh = self.path.open("a", encoding="utf-8")
        atexit.register(self.close)

    def _resume(self) -> None:
        """Continue the existing chain so history spans runs, not just processes."""
        if not self.path.exists():
            return
        last = None
        with self.path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if last is None:
            return
        try:
            record = json.loads(last)
        except json.JSONDecodeError:
            # A torn tail from an earlier crash. Starting a fresh chain here
            # would hide it; leave the break for verify() to report.
            return
        self.prev_hash = record.get("hash", GENESIS)

    def _is_critical(self, event: str, fields: dict[str, Any]) -> bool:
        if event in _CRITICAL_EVENTS:
            return True
        if fields.get("held") or fields.get("shadowed"):
            return True
        return fields.get("outcome") in ("block", "escalate") or fields.get(
            "decision"
        ) in ("block", "escalate")

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        entry: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "seq": self.seq,
            "event": event,
            "prev_hash": self.prev_hash,
            **fields,
        }
        entry["hash"] = _chain(self.prev_hash, entry)
        self.prev_hash = entry["hash"]

        self._fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._fh.flush()  # ordering is guaranteed here; durability below
        self._unsynced += 1

        if self.sync_mode == ALWAYS:
            self._sync(full=True)
        elif self.sync_mode == CRITICAL and self._is_critical(event, fields):
            self._sync(full=True)
        elif (
            self._unsynced >= self.batch_size
            or time.monotonic() - self._last_sync >= self.batch_seconds
        ):
            self._sync(full=False)
        return entry

    def _sync(self, *, full: bool) -> None:
        try:
            fd = self._fh.fileno()
            if full and _FULLFSYNC is not None:
                # macOS: fsync alone leaves the data in the drive's write cache.
                fcntl.fcntl(fd, _FULLFSYNC)
            else:
                os.fsync(fd)
        except (OSError, ValueError):
            return
        self._unsynced = 0
        self._last_sync = time.monotonic()

    def flush(self) -> None:
        """Force everything written so far to durable storage."""
        if not self._fh.closed:
            self._fh.flush()
            self._sync(full=True)

    def close(self) -> None:
        if getattr(self, "_fh", None) and not self._fh.closed:
            self.flush()
            self._fh.close()


def read_all(path: str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: malformed JSON ({exc})") from exc


def verify(path: str) -> tuple[bool, int, str | None]:
    """Walk the chain. Returns (ok, records_checked, first_error).

    A malformed final line is reported as a truncated tail rather than tampering:
    losing the last record to a crash is an availability event, while an altered
    record in the middle is a security one, and an operator needs to be able to
    tell them apart.
    """
    lines = [
        (n, line)
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    prev = GENESIS
    count = 0
    for index, (line_no, line) in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                return False, count, (
                    f"line {line_no}: truncated final record — the writer was "
                    f"interrupted. The {count} records before it verify."
                )
            return False, count, f"line {line_no}: malformed JSON mid-ledger"

        count += 1
        if record.get("prev_hash") != prev:
            return False, count, (
                f"record seq={record.get('seq')} expected prev_hash {prev[:12]}... "
                f"but carries {str(record.get('prev_hash'))[:12]}..."
            )
        if record.get("hash") != _chain(prev, record):
            return False, count, (
                f"record seq={record.get('seq')} hash mismatch: contents were "
                f"altered after they were written"
            )
        prev = record["hash"]
    return True, count, None
