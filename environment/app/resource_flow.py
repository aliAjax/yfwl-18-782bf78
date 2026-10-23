#!/usr/bin/env python3
"""Legacy production planner with an incomplete resource model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REFUSED = "OPTIMIZATION-CASE-REFUSED: "


def compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--journal")
    parser.add_argument("--report")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        return 0
    if not args.scenario or not args.journal or not args.report:
        print(REFUSED + "scenario journal and report are required", file=sys.stderr)
        return 3
    try:
        scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
        product_ids = [item["id"] for item in scenario["products"]]
        resource_ids = [item["id"] for item in scenario["resources"]]
        allocations = {item: 0 for item in product_ids}
        yields = {resource: {item: 0.0 for item in product_ids} for resource in resource_ids}
        usage = {resource: 0.0 for resource in resource_ids}
        base = {
            "allocations": allocations,
            "case_id": scenario["case_id"],
            "objective": 0.0,
            "resource_usage": usage,
            "schema_version": 1,
            "solver": {"nodes": 0, "strategy": "branch_and_bound"},
            "yield_matrix": yields,
        }
        checksum = hashlib.sha256(compact(base)).hexdigest()
        entry = {
            "case_id": scenario["case_id"],
            "input_sha256": hashlib.sha256(compact(scenario)).hexdigest(),
            "report_sha256": hashlib.sha256(compact({**base, "checksum": checksum})).hexdigest(),
            "status": "completed",
        }
        report = {**base, "checksum": checksum, "ledger_entry": entry}
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(compact(report))
        journal_path = Path(args.journal)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(REFUSED + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
