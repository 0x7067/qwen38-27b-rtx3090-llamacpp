#!/usr/bin/env python3
"""Multi-turn cumulative file-editing session against llama-server /completion.
Simulates agentic editing: each turn asks for a full-file re-emission with one
small change, growing the conversation. This is the workload ngram-mod targets.
Usage: driver.py <port> <arm-name>"""
import json, sys, time, urllib.request

port, arm = sys.argv[1], sys.argv[2]

BASE_FILE = '''import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("inventory")

@dataclass
class Item:
    sku: str
    name: str
    quantity: int
    unit_price_cents: int
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents

class InventoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "sku TEXT PRIMARY KEY, name TEXT NOT NULL, quantity INTEGER NOT NULL,"
            "unit_price_cents INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        self.conn.commit()

    def upsert(self, item: Item) -> None:
        self.conn.execute(
            "INSERT INTO items (sku, name, quantity, unit_price_cents, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(sku) DO UPDATE SET"
            " name=excluded.name, quantity=excluded.quantity,"
            " unit_price_cents=excluded.unit_price_cents, updated_at=excluded.updated_at",
            (item.sku, item.name, item.quantity, item.unit_price_cents, item.updated_at),
        )
        self.conn.commit()

    def get(self, sku: str) -> Optional[Item]:
        row = self.conn.execute(
            "SELECT sku, name, quantity, unit_price_cents, updated_at FROM items WHERE sku=?",
            (sku,),
        ).fetchone()
        return Item(*row) if row else None

    def all_items(self) -> Iterable[Item]:
        for row in self.conn.execute(
            "SELECT sku, name, quantity, unit_price_cents, updated_at FROM items ORDER BY sku"
        ):
            yield Item(*row)

    def total_value_cents(self) -> int:
        return sum(i.total_cents() for i in self.all_items())

def export_json(store: InventoryStore, out_path: Path) -> int:
    items = [i.__dict__ for i in store.all_items()]
    out_path.write_text(json.dumps(items, indent=2))
    return len(items)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tiny inventory CLI")
    parser.add_argument("--db", type=Path, default=Path("inventory.db"))
    parser.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    store = InventoryStore(args.db)
    if args.export:
        n = export_json(store, args.export)
        log.info("exported %d items", n)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

EDITS = [
    "Rename the class InventoryStore to SqliteInventoryStore everywhere.",
    "Add a method `delete(self, sku: str) -> bool` to the store class that removes a row and returns whether it existed.",
    "Add a `--verbose` flag to the argument parser that sets logging level to DEBUG.",
    "Add a docstring to every public method (one line each).",
    "Add a `low_stock(self, threshold: int)` method returning items with quantity below threshold.",
    "Change export_json to also write a `total_value_cents` field at the top level (wrap items in an object).",
    "Add type hint `-> None` to __init__ and make db_path accept str or Path via `os.fspath`.",
    "Add a `restock(self, sku: str, amount: int)` method that increases quantity and updates updated_at.",
]

convo = ("You are a precise code editor. Here is a Python file:\n\n```python\n"
         + BASE_FILE + "```\n\n")

results = []
for turn, edit in enumerate(EDITS, 1):
    convo += (f"Task {turn}: {edit}\nOutput the COMPLETE updated file in a single "
              "```python code block. No commentary before or after.\n\nAnswer:\n")
    payload = {"prompt": convo, "n_predict": 1400, "temperature": 0, "cache_prompt": True}
    t0 = time.time()
    req = urllib.request.Request(f"http://localhost:{port}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.time() - t0
    t = resp["timings"]
    row = dict(turn=turn, tps=round(t["predicted_per_second"], 2),
               n=t["predicted_n"], draft_n=t.get("draft_n"),
               draft_acc=t.get("draft_n_accepted"),
               prompt_n=t["prompt_n"], wall=round(wall, 1))
    results.append(row)
    print(f"{arm} turn {turn}: {row}", flush=True)
    convo += resp["content"] + "\n\n"

tps = [r["tps"] for r in results]
print(f"{arm} SUMMARY: first3={sum(tps[:3])/3:.1f} last3={sum(tps[-3:])/3:.1f} "
      f"all={sum(tps)/len(tps):.1f} final_ctx={results[-1]['prompt_n']+results[-1]['n']}")
