#!/usr/bin/env python3
"""Recover a hidden resource-yield matrix and solve a bounded integer plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REFUSED = "OPTIMIZATION-CASE-REFUSED: "
TOP_KEYS = {"case_id", "products", "resources", "schema_version", "settlements", "unit"}
PRODUCT_KEYS = {"id", "max_batches", "revenue"}
RESOURCE_KEYS = {"capacity", "id"}
SETTLEMENT_KEYS = {"allocation", "observation", "usage"}
EPSILON = 1e-9


class Refusal(ValueError):
    """The optimization case is not admissible."""


def number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"cannot read scenario: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=without_duplicates)
    except json.JSONDecodeError as exc:
        raise Refusal(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal("scenario root must be an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Refusal(f"{label} keys differ; missing={missing} extra={extra}")


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    exact_keys(raw, TOP_KEYS, "scenario")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise Refusal("schema_version must be integer 1")
    if raw["unit"] != "batch":
        raise Refusal("unit must be batch")
    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not case_id.strip() or len(case_id.split("-")) < 3:
        raise Refusal("case_id must be a three-segment business identifier")

    products = raw["products"]
    if not isinstance(products, list) or not 4 <= len(products) <= 9:
        raise Refusal("products must contain between four and nine entries")
    product_ids: list[str] = []
    normalized_products: list[dict[str, Any]] = []
    for index, item in enumerate(products):
        if not isinstance(item, dict):
            raise Refusal(f"products[{index}] must be an object")
        exact_keys(item, PRODUCT_KEYS, f"products[{index}]")
        product_id = item["id"]
        if not isinstance(product_id, str) or not product_id.strip():
            raise Refusal(f"products[{index}].id must be non-empty")
        if product_id in product_ids:
            raise Refusal("product ids must be unique")
        maximum = item["max_batches"]
        if type(maximum) is not int or not 1 <= maximum <= 4:
            raise Refusal("max_batches must be an integer from 1 through 4")
        if not number(item["revenue"]) or float(item["revenue"]) <= 0.0:
            raise Refusal("revenue must be positive and finite")
        product_ids.append(product_id)
        normalized_products.append(
            {
                "id": product_id,
                "max_batches": maximum,
                "revenue": float(item["revenue"]),
            }
        )

    resources = raw["resources"]
    if not isinstance(resources, list) or len(resources) != 3:
        raise Refusal("resources must contain exactly three entries")
    resource_ids: list[str] = []
    normalized_resources: list[dict[str, Any]] = []
    for index, item in enumerate(resources):
        if not isinstance(item, dict):
            raise Refusal(f"resources[{index}] must be an object")
        exact_keys(item, RESOURCE_KEYS, f"resources[{index}]")
        resource_id = item["id"]
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise Refusal(f"resources[{index}].id must be non-empty")
        if resource_id in resource_ids:
            raise Refusal("resource ids must be unique")
        capacity = item["capacity"]
        if type(capacity) is not int or not 1 <= capacity <= 80:
            raise Refusal("capacity must be an integer from 1 through 80")
        resource_ids.append(resource_id)
        normalized_resources.append({"capacity": capacity, "id": resource_id})

    settlements = raw["settlements"]
    if not isinstance(settlements, list) or not 6 <= len(settlements) <= 12:
        raise Refusal("settlements must contain between six and twelve entries")
    observations: set[str] = set()
    normalized_settlements: list[dict[str, Any]] = []
    for index, item in enumerate(settlements):
        if not isinstance(item, dict):
            raise Refusal(f"settlements[{index}] must be an object")
        exact_keys(item, SETTLEMENT_KEYS, f"settlements[{index}]")
        observation = item["observation"]
        if not isinstance(observation, str) or not observation.strip():
            raise Refusal("observation must be a non-empty string")
        if observation in observations:
            raise Refusal("observation ids must be unique")
        observations.add(observation)
        allocation = item["allocation"]
        if not isinstance(allocation, dict) or set(allocation) != set(product_ids):
            raise Refusal("allocation keys must exactly cover products")
        clean_allocation: dict[str, int] = {}
        for product_id in product_ids:
            value = allocation[product_id]
            if type(value) is not int or not 0 <= value <= 4:
                raise Refusal("allocation values must be integers from 0 through 4")
            clean_allocation[product_id] = value
        usage = item["usage"]
        if not isinstance(usage, dict) or set(usage) != set(resource_ids):
            raise Refusal("usage keys must exactly cover resources")
        clean_usage: dict[str, float] = {}
        for resource_id in resource_ids:
            value = usage[resource_id]
            if not number(value) or float(value) < 0.0:
                raise Refusal("usage values must be non-negative finite numbers")
            clean_usage[resource_id] = float(value)
        normalized_settlements.append(
            {
                "allocation": clean_allocation,
                "observation": observation,
                "usage": clean_usage,
            }
        )

    return {
        "case_id": case_id,
        "products": normalized_products,
        "resources": normalized_resources,
        "schema_version": 1,
        "settlements": normalized_settlements,
        "unit": "batch",
    }


def solve_linear(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    work = [list(map(float, matrix[row])) + [float(target[row])] for row in range(size)]
    scale = max(1.0, max(abs(value) for row in matrix for value in row))
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= EPSILON * scale:
            raise Refusal("settlement matrix is rank deficient")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for cell in range(column, size + 1):
            work[column][cell] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if abs(factor) <= EPSILON:
                continue
            for cell in range(column, size + 1):
                work[row][cell] -= factor * work[column][cell]
    return [work[row][size] for row in range(size)]


def recover_yields(case: dict[str, Any]) -> dict[str, dict[str, float]]:
    product_ids = [item["id"] for item in case["products"]]
    resource_ids = [item["id"] for item in case["resources"]]
    width = len(product_ids)
    design = [[float(item["allocation"][name]) for name in product_ids] for item in case["settlements"]]
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    right = {name: [0.0 for _ in range(width)] for name in resource_ids}
    for row, settlement in zip(design, case["settlements"]):
        for left in range(width):
            for other in range(width):
                normal[left][other] += row[left] * row[other]
            for resource_id in resource_ids:
                right[resource_id][left] += row[left] * settlement["usage"][resource_id]
    coefficients: dict[str, list[float]] = {}
    for resource_id in resource_ids:
        values = solve_linear(normal, right[resource_id])
        cleaned = []
        for value in values:
            if abs(value - round(value)) <= 1e-7:
                value = float(round(value))
            if value < -1e-7:
                raise Refusal("recovered resource consumption is negative")
            cleaned.append(max(0.0, value))
        coefficients[resource_id] = cleaned
    if any(max(values) <= EPSILON for values in coefficients.values()):
        raise Refusal("a resource has no observable consumption")
    return {
        resource_id: {
            product_id: coefficients[resource_id][index]
            for index, product_id in enumerate(product_ids)
        }
        for resource_id in resource_ids
    }


def usage_of(
    vector: list[int],
    case: dict[str, Any],
    yields: dict[str, dict[str, float]],
) -> dict[str, float]:
    product_ids = [item["id"] for item in case["products"]]
    return {
        resource["id"]: math.fsum(
            vector[index] * yields[resource["id"]][product_id]
            for index, product_id in enumerate(product_ids)
        )
        for resource in case["resources"]
    }


def optimize(
    case: dict[str, Any],
    yields: dict[str, dict[str, float]],
) -> tuple[list[int], float, dict[str, float], int]:
    products = case["products"]
    resources = case["resources"]
    maximums = [item["max_batches"] for item in products]
    revenues = [item["revenue"] for item in products]
    resource_ids = [item["id"] for item in resources]
    capacities = [float(item["capacity"]) for item in resources]
    suffix_revenue = [0.0 for _ in range(len(products) + 1)]
    for index in range(len(products) - 1, -1, -1):
        suffix_revenue[index] = suffix_revenue[index + 1] + maximums[index] * revenues[index]

    best_vector: list[int] | None = None
    best_objective = -1.0
    best_usage: dict[str, float] = {}
    nodes = 0

    def visit(position: int, vector: list[int], used: list[float], objective: float) -> None:
        nonlocal best_vector, best_objective, best_usage, nodes
        nodes += 1
        if objective + suffix_revenue[position] < best_objective - EPSILON:
            return
        if position == len(products):
            candidate = list(vector)
            if objective > best_objective + EPSILON or (
                abs(objective - best_objective) <= EPSILON
                and (best_vector is None or candidate < best_vector)
            ):
                best_vector = candidate
                best_objective = objective
                best_usage = {
                    resource_ids[index]: used[index] for index in range(len(resource_ids))
                }
            return
        product_id = products[position]["id"]
        for units in range(maximums[position], -1, -1):
            next_used = [
                used[index] + units * yields[resource_ids[index]][product_id]
                for index in range(len(resource_ids))
            ]
            if any(next_used[index] > capacities[index] + 1e-8 for index in range(len(resource_ids))):
                continue
            vector.append(units)
            visit(
                position + 1,
                vector,
                next_used,
                objective + units * revenues[position],
            )
            vector.pop()

    visit(0, [], [0.0 for _ in resource_ids], 0.0)
    if best_vector is None:
        raise Refusal("no integer plan fits the recovered capacities")
    return best_vector, best_objective, best_usage, nodes


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_report(case: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    yields = recover_yields(case)
    vector, objective, usage, nodes = optimize(case, yields)
    product_ids = [item["id"] for item in case["products"]]
    base = {
        "allocations": {name: vector[index] for index, name in enumerate(product_ids)},
        "case_id": case["case_id"],
        "objective": objective,
        "resource_usage": usage,
        "schema_version": 1,
        "solver": {"nodes": nodes, "strategy": "branch_and_bound"},
        "yield_matrix": yields,
    }
    checksum = hashlib.sha256(compact_bytes(base)).hexdigest()
    core = {**base, "checksum": checksum}
    report_digest = hashlib.sha256(compact_bytes(core)).hexdigest()
    entry = {
        "case_id": case["case_id"],
        "input_sha256": hashlib.sha256(compact_bytes(case)).hexdigest(),
        "report_sha256": report_digest,
        "status": "completed",
    }
    result = {**core, "ledger_entry": entry}
    return result, compact_bytes(result)


def read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            item = json.loads(line, object_pairs_hook=without_duplicates)
        except (json.JSONDecodeError, Refusal) as exc:
            raise Refusal(f"journal line {number} is invalid: {exc}") from exc
        if not isinstance(item, dict):
            raise Refusal("journal entries must be objects")
        result.append(item)
    return result


def stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def publish(report_path: Path, journal_path: Path, result: dict[str, Any], payload: bytes) -> None:
    lock_path = journal_path.with_name(journal_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        records = read_journal(journal_path)
        existing = next(
            (
                item
                for item in records
                if item.get("case_id") == result["ledger_entry"]["case_id"]
                and item.get("input_sha256") == result["ledger_entry"]["input_sha256"]
                and item.get("report_sha256") == result["ledger_entry"]["report_sha256"]
                and item.get("status") == "completed"
            ),
            None,
        )
        report_stage = stage(report_path, payload)
        journal_stage: Path | None = None
        try:
            if existing is None:
                combined = b"".join(compact_bytes(item) for item in [*records, result["ledger_entry"]])
                journal_stage = stage(journal_path, combined)
                os.replace(journal_stage, journal_path)
                journal_stage = None
            os.replace(report_stage, report_path)
            report_stage = None
        finally:
            if report_stage is not None:
                report_stage.unlink(missing_ok=True)
            if journal_stage is not None:
                journal_stage.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--journal")
    parser.add_argument("--report")
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inspect:
        return 0
    if not args.scenario or not args.journal or not args.report:
        print(REFUSED + "scenario journal and report are required", file=sys.stderr)
        return 3
    try:
        case = validate(load_json(Path(args.scenario)))
        result, payload = build_report(case)
        publish(Path(args.report), Path(args.journal), result, payload)
        return 0
    except (Refusal, OSError, ValueError, TypeError, KeyError) as exc:
        print(REFUSED + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
