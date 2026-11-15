"""SQLite case tracker."""
import json
import sqlite3
from datetime import datetime

STATUSES = ["Open", "Under review", "Escalated", "Closed - fraud", "Closed - false positive"]


def _conn(path):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT UNIQUE, status TEXT, risk TEXT,
        typology TEXT, txn_ids TEXT, explanation TEXT, notes TEXT DEFAULT '', created_at TEXT, updated_at TEXT)""")
    return c


def open_case(path, ev):
    now = datetime.now().isoformat(timespec="seconds")
    with _conn(path) as c:
        c.execute("INSERT OR IGNORE INTO cases(customer_id,status,risk,typology,txn_ids,explanation,created_at,updated_at) "
                  "VALUES(?,?,?,?,?,?,?,?)",
                  (ev["customer_id"], "Open", ev["risk"], ev["typology"], json.dumps(ev["txn_ids"]), ev["explanation"], now, now))


def list_cases(path):
    with _conn(path) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM cases ORDER BY id DESC")]


def update_case(path, case_id, status=None, notes=None):
    assert status is None or status in STATUSES
    with _conn(path) as c:
        if status:
            c.execute("UPDATE cases SET status=?, updated_at=? WHERE id=?", (status, datetime.now().isoformat(timespec="seconds"), case_id))
        if notes is not None:
            c.execute("UPDATE cases SET notes=? WHERE id=?", (notes, case_id))
