"""
SQLite database: tracks every uploaded training image and every retraining run.
(Image files live on disk under data/uploads/; the DB stores the records.)
"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "app.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT,
    label       TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha1        TEXT UNIQUE NOT NULL,
    created_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS replica_stats (
    replica          TEXT PRIMARY KEY,
    role             TEXT,
    started_at       TEXT,
    predictions      INTEGER NOT NULL DEFAULT 0,
    errors           INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL    NOT NULL DEFAULT 0,
    updated_ts       REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type TEXT,
    status       TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    n_new        INTEGER,
    n_replay     INTEGER,
    epochs       INTEGER,
    old_metrics  TEXT,
    new_metrics  TEXT,
    promoted     INTEGER,
    message      TEXT
);
"""
_RUN_COLUMNS = {"status", "finished_at", "n_new", "n_replay", "epochs",
                "old_metrics", "new_metrics", "promoted", "message"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(reset_stale: bool = True) -> None:
    """Create tables. reset_stale=True marks runs still 'running' as interrupted (the process
    that owned them must have died). Inference replicas pass False so that scaling the
    replicas up never interrupts a retraining run owned by the control service."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        if reset_stale:
            conn.execute(
                "UPDATE retrain_runs SET status='interrupted', finished_at=? WHERE status='running'",
                (now(),),
            )


# ---------------------------------------------------------------- uploads
def has_hash(sha1: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM uploads WHERE sha1=?", (sha1,)).fetchone() is not None


def add_upload(filename: str, label: str, path: str, sha1: str) -> bool:
    """Insert a record. Returns False if the same image was already uploaded."""
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO uploads (filename, label, path, sha1, created_at) VALUES (?,?,?,?,?)",
                (filename, label, path, sha1, now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def count_pending() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads WHERE used=0").fetchone()[0]


def pending_by_class() -> dict:
    with connect() as conn:
        rows = conn.execute("SELECT label, COUNT(*) n FROM uploads WHERE used=0 GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}


def uploads_by_class() -> dict:
    with connect() as conn:
        rows = conn.execute("SELECT label, COUNT(*) n FROM uploads GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}


def max_upload_id() -> int:
    with connect() as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM uploads").fetchone()[0]


def mark_uploads_used(up_to_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE uploads SET used=1 WHERE used=0 AND id<=?", (up_to_id,))


# ---------------------------------------------------------------- retraining runs
def start_run(trigger_type: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs (trigger_type, status, started_at) VALUES (?, 'running', ?)",
            (trigger_type, now()),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, **fields) -> None:
    fields["status"] = status
    fields["finished_at"] = now()
    cols, vals = [], []
    for k, v in fields.items():
        if k not in _RUN_COLUMNS:
            raise ValueError(f"unknown column {k}")
        cols.append(f"{k}=?")
        vals.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
    with connect() as conn:
        conn.execute(f"UPDATE retrain_runs SET {', '.join(cols)} WHERE id=?", (*vals, run_id))


def list_runs(limit: int = 20) -> list:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("old_metrics", "new_metrics"):
            d[k] = json.loads(d[k]) if d[k] else None
        d["promoted"] = None if d["promoted"] is None else bool(d["promoted"])
        out.append(d)
    return out


# ---------------------------------------------------------------- serving statistics (shared by all replicas)
def upsert_replica_stats(replica: str, role: str, started_at: str,
                         predictions: int, errors: int, total_latency_ms: float) -> None:
    """Each serving process reports its own running totals every few seconds."""
    with connect() as conn:
        conn.execute(
            """INSERT INTO replica_stats (replica, role, started_at, predictions, errors, total_latency_ms, updated_ts)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(replica) DO UPDATE SET predictions=excluded.predictions, errors=excluded.errors,
                   total_latency_ms=excluded.total_latency_ms, updated_ts=excluded.updated_ts""",
            (replica, role, started_at, predictions, errors, total_latency_ms, time.time()),
        )


def get_stats_totals(online_window: float = 15.0) -> dict:
    """Totals across every replica (cumulative) + how many replicas reported recently."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(predictions),0) p, COALESCE(SUM(errors),0) e, COALESCE(SUM(total_latency_ms),0) t "
            "FROM replica_stats").fetchone()
        online = conn.execute("SELECT COUNT(*) FROM replica_stats WHERE updated_ts > ?",
                              (time.time() - online_window,)).fetchone()[0]
    return {"predictions": int(row["p"]), "errors": int(row["e"]),
            "total_latency_ms": float(row["t"]), "replicas_online": int(online)}