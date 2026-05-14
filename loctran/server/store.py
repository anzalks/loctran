"""Minimal SQLite-backed job store.

Jobs are kept in an in-process dict for fast reads; every write goes through
to the database so state survives a restart.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path.home() / ".loctran" / "jobs.db"

# In-process cache: job_id -> job dict
_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
"""


def init_db() -> None:
    """Initialise the database and warm the in-memory cache."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _connect()
    con.execute(_DDL)
    con.commit()
    # Warm cache from persisted rows
    for row in con.execute("SELECT job_id, data FROM jobs"):
        try:
            _cache[row[0]] = json.loads(row[1])
        except json.JSONDecodeError:
            pass
    con.close()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def upsert_job(job: dict) -> None:
    """Write job to cache and database."""
    job_id: str = job["id"]
    _cache[job_id] = job
    con = _connect()
    con.execute(
        "INSERT OR REPLACE INTO jobs (job_id, data, updated_at) VALUES (?, ?, ?)",
        (job_id, json.dumps(job), time.time()),
    )
    con.commit()
    con.close()


def get_job(job_id: str) -> dict | None:
    """Return a job dict or None if not found."""
    if job_id in _cache:
        return _cache[job_id]
    con = _connect()
    row = con.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    con.close()
    if row:
        job = json.loads(row[0])
        _cache[job_id] = job
        return job
    return None


def list_active_jobs() -> list[dict]:
    """Return all jobs that are not completed or failed."""
    terminal = {"completed", "failed"}
    return [j for j in _cache.values() if j.get("status") not in terminal]


def cleanup_old_jobs(retention_seconds: int = 3600) -> int:
    """Delete completed or failed job rows older than `retention_seconds`.

    This removes persisted job records from SQLite and clears matching cache
    entries. It does not delete user files from disk.
    """
    cutoff = time.time() - retention_seconds
    to_delete = [
        jid
        for jid, j in list(_cache.items())
        if j.get("status") in {"completed", "failed"}
        and j.get("created_at", 0) < cutoff
    ]
    if to_delete:
        con = _connect()
        con.executemany(
            "DELETE FROM jobs WHERE job_id = ?", [(jid,) for jid in to_delete]
        )
        con.commit()
        con.close()
        for jid in to_delete:
            _cache.pop(jid, None)
    return len(to_delete)
