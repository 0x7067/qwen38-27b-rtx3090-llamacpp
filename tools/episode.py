#!/usr/bin/env python3
"""North-star metric: wall-clock seconds to complete a fixed 12-turn agentic
coding episode against llama-server's chat endpoint (real template, so
reasoning tokens count). Deterministic: self-generated ~20k-token repo
context, temp 0, fixed turn script, per-turn correctness spot checks.

Usage: episode.py <port> <label> [reasoning_effort] [max_tokens_multiplier]
reasoning_effort: medium (default) | xhigh | high | low — sent per request,
overriding the server's template default. max_tokens_multiplier scales every
turn's cap (use >=4 for xhigh so long thinking is measured, not truncated).
Prints per-turn rows and an EPISODE summary (total wall, prefill ms, decode
ms, generated tokens, reasoning share, checks passed).
"""
import json, sys, time, urllib.request

port, label = sys.argv[1], sys.argv[2]
effort = sys.argv[3] if len(sys.argv) > 3 else "medium"
mult = int(sys.argv[4]) if len(sys.argv) > 4 else 1

# ---- deterministic synthetic repo (~20k tokens) ----
BASE = '''import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

@dataclass
class {Entity}:
    {key}: str
    name: str
    quantity: int
    unit_price_cents: int
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents

class {Entity}Store:
    """SQLite-backed store for {entity} rows."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS {table} ("
            "{key} TEXT PRIMARY KEY, name TEXT NOT NULL, quantity INTEGER NOT NULL,"
            "unit_price_cents INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        self.conn.commit()

    def upsert(self, row: {Entity}) -> None:
        self.conn.execute(
            "INSERT INTO {table} ({key}, name, quantity, unit_price_cents, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT({key}) DO UPDATE SET"
            " name=excluded.name, quantity=excluded.quantity,"
            " unit_price_cents=excluded.unit_price_cents, updated_at=excluded.updated_at",
            (row.{key}, row.name, row.quantity, row.unit_price_cents, row.updated_at),
        )
        self.conn.commit()

    def get(self, {key}: str) -> Optional[{Entity}]:
        r = self.conn.execute(
            "SELECT {key}, name, quantity, unit_price_cents, updated_at FROM {table} WHERE {key}=?",
            ({key},),
        ).fetchone()
        return {Entity}(*r) if r else None

    def all_rows(self) -> Iterable[{Entity}]:
        for r in self.conn.execute(
            "SELECT {key}, name, quantity, unit_price_cents, updated_at FROM {table} ORDER BY {key}"
        ):
            yield {Entity}(*r)

    def total_value_cents(self) -> int:
        return sum(x.total_cents() for x in self.all_rows())

def export_{table}_json(store: {Entity}Store, out_path: Path) -> int:
    rows = [x.__dict__ for x in store.all_rows()]
    out_path.write_text(json.dumps(rows, indent=2))
    return len(rows)
'''

ENTITIES = ["Item", "Product", "Asset", "Ticket", "Order", "Invoice", "Shipment",
            "Customer", "Vendor", "Contract", "License", "Device", "Sensor",
            "Reading", "Alert", "Task", "Note", "Tag", "Batch", "Refund"]

def repo():
    files = []
    for i, e in enumerate(ENTITIES):
        body = (BASE.replace("{Entity}", e).replace("{entity}", e.lower())
                    .replace("{table}", e.lower() + "s").replace("{key}", e.lower() + "_id"))
        files.append(f"### File: modules/{i:02d}_{e.lower()}.py\n```python\n{body}```\n")
    return "\n".join(files)

SYSTEM = ("You are a precise coding assistant working on the repository below. "
          "When asked to edit a file, output the COMPLETE updated file in one "
          "```python block with no commentary.\n\nREPOSITORY:\n\n" + repo())

TURNS = [
    ("Summarize the architecture of this codebase in one paragraph, then list the three most duplicated patterns.", 700, ["Store"]),
    ("Edit modules/02_asset.py: add a method `delete(self, asset_id: str) -> bool` to AssetStore that removes a row and returns whether it existed.", 1500, ["def delete", "class AssetStore"]),
    ("Edit modules/06_shipment.py: add a `low_stock(self, threshold: int)` method to ShipmentStore returning rows with quantity below threshold.", 1500, ["def low_stock", "class ShipmentStore"]),
    ("Which methods in this repository execute an INSERT statement? List file and method name only.", 500, ["upsert"]),
    ("Edit modules/02_asset.py again: also add a docstring to every public method (keep the delete method you added).", 1500, ["def delete", '"""']),
    ("Write a NEW file modules/20_report.py that imports ItemStore and OrderStore and produces a combined JSON report using the same style as the existing export functions.", 1200, ["ItemStore", "OrderStore"]),
    ("Edit modules/11_device.py: add a `restock(self, device_id: str, amount: int)` method that increases quantity and refreshes updated_at.", 1500, ["def restock", "class DeviceStore"]),
    ("If every entity gained a `currency` field, which methods and SQL statements would need to change? Answer as a short list.", 600, ["upsert"]),
    ("Edit modules/20_report.py (the file you wrote): add error handling so a missing table produces an empty section instead of raising.", 1200, ["except"]),
    ("Write pytest unit tests for AssetStore covering upsert, get, delete, and low_stock... wait, low_stock is on ShipmentStore. Test only what exists on AssetStore.", 1200, ["def test_"]),
    ("Edit modules/06_shipment.py again: rename low_stock to below_threshold everywhere in the file.", 1500, ["def below_threshold"]),
    ("Summarize every change made in this session as a bullet list.", 600, ["delete"]),
]

messages = [{"role": "system", "content": SYSTEM}]
tot_wall = tot_prompt_ms = tot_pred_ms = tot_tokens = tot_reason = 0.0
checks_passed = 0
checks_total = 0

for i, (task, npred, checks) in enumerate(TURNS, 1):
    messages.append({"role": "user", "content": task})
    payload = {"messages": messages, "max_tokens": npred * mult, "temperature": 0,
               "reasoning_effort": effort,
               "cache_prompt": True, "timings_per_token": False}
    t0 = time.time()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    wall = time.time() - t0
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    t = resp.get("timings", {})
    ok = sum(1 for c in checks if c in content)
    checks_passed += ok; checks_total += len(checks)
    tot_wall += wall
    tot_prompt_ms += t.get("prompt_ms", 0); tot_pred_ms += t.get("predicted_ms", 0)
    tot_tokens += t.get("predicted_n", 0); tot_reason += len(reasoning)
    print(f"{label} t{i:02d} wall={wall:6.1f}s prefill={t.get('prompt_ms',0)/1000:6.1f}s "
          f"decode={t.get('predicted_ms',0)/1000:6.1f}s gen={t.get('predicted_n',0):4} "
          f"tps={t.get('predicted_per_second',0):6.1f} reason_chars={len(reasoning):5} "
          f"checks={ok}/{len(checks)}", flush=True)
    messages.append({"role": "assistant", "content": content})

print(f"{label} EPISODE: wall={tot_wall:.1f}s prefill={tot_prompt_ms/1000:.1f}s "
      f"decode={tot_pred_ms/1000:.1f}s gen_tokens={int(tot_tokens)} "
      f"reason_chars={int(tot_reason)} checks={checks_passed}/{checks_total}")
