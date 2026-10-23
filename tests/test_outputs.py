"""Behavior tests for the inverse resource-planning task."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ENTRY = os.environ.get("OPTIMIZATION_ENTRY", "/app/resource_flow.py")
REFUSED = "OPTIMIZATION-CASE-REFUSED: "
PRODUCTS = [
    {"id": "V-HEX", "max_batches": 4, "revenue": 33.0},
    {"id": "V-ORB", "max_batches": 3, "revenue": 29.0},
    {"id": "V-CAL", "max_batches": 4, "revenue": 23.0},
    {"id": "V-LUX", "max_batches": 2, "revenue": 19.0},
]
RESOURCES = [
    {"id": "steam", "capacity": 42},
    {"id": "coolant", "capacity": 36},
    {"id": "catalyst", "capacity": 30},
]
YIELDS = {
    "steam": {"V-HEX": 2.0, "V-ORB": 1.0, "V-CAL": 3.0, "V-LUX": 1.0},
    "coolant": {"V-HEX": 1.0, "V-ORB": 3.0, "V-CAL": 1.0, "V-LUX": 2.0},
    "catalyst": {"V-HEX": 1.0, "V-ORB": 1.0, "V-CAL": 2.0, "V-LUX": 3.0},
}
ALLOCATIONS = [
    [4, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 2],
    [2, 1, 3, 1],
    [1, 2, 2, 1],
    [3, 1, 1, 2],
]


def usage_for(vector: list[int]) -> dict[str, float]:
    ids = [item["id"] for item in PRODUCTS]
    return {
        resource["id"]: sum(vector[index] * YIELDS[resource["id"]][product_id] for index, product_id in enumerate(ids))
        for resource in RESOURCES
    }


def build_payload(case_id: str = "OPT-E-CHECK-18A") -> dict:
    ids = [item["id"] for item in PRODUCTS]
    settlements = []
    for index, vector in enumerate(ALLOCATIONS):
        settlements.append(
            {
                "allocation": {product_id: vector[position] for position, product_id in enumerate(ids)},
                "observation": f"settle-{index + 1:02d}",
                "usage": usage_for(vector),
            }
        )
    return {
        "case_id": case_id,
        "products": copy.deepcopy(PRODUCTS),
        "resources": copy.deepcopy(RESOURCES),
        "schema_version": 1,
        "settlements": settlements,
        "unit": "batch",
    }


def write_case(tmp_path: Path, payload: dict, name: str = "scenario") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def invoke(
    tmp_path: Path,
    payload: dict | None = None,
    *,
    raw: str | None = None,
    journal_name: str = "journal.jsonl",
    report_name: str = "plan.json",
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    scenario = tmp_path / "input" / "scenario.json"
    scenario.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        scenario.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    else:
        scenario.write_text(raw, encoding="utf-8")
    journal = tmp_path / "state" / journal_name
    report = tmp_path / "out" / report_name
    process = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--scenario",
            str(scenario),
            "--journal",
            str(journal),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    return process, scenario, journal, report


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def expected_plan(payload: dict) -> tuple[dict[str, int], float, dict[str, float]]:
    products = payload["products"]
    resources = payload["resources"]
    best_vector: tuple[int, ...] | None = None
    best_objective = -1.0

    def visit(position: int, vector: list[int]) -> None:
        nonlocal best_vector, best_objective
        if position == len(products):
            candidate = tuple(vector)
            objective = sum(candidate[index] * products[index]["revenue"] for index in range(len(products)))
            usage = usage_for(list(candidate))
            if any(usage[item["id"]] > item["capacity"] + 1e-9 for item in resources):
                return
            if objective > best_objective + 1e-9 or (
                abs(objective - best_objective) <= 1e-9
                and (best_vector is None or candidate < best_vector)
            ):
                best_vector = candidate
                best_objective = objective
            return
        for units in range(products[position]["max_batches"] + 1):
            visit(position + 1, [*vector, units])

    visit(0, [])
    assert best_vector is not None
    allocations = {
        products[index]["id"]: best_vector[index] for index in range(len(products))
    }
    return allocations, best_objective, usage_for(list(best_vector))


def test_existing_inspect_mode_returns_zero() -> None:
    """题面：「`--inspect` 只做台账查询，进程退出状态应为 0」"""
    process = subprocess.run(
        [sys.executable, ENTRY, "--inspect"],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert process.returncode == 0

def test_success_is_silent_and_creates_report(tmp_path: Path) -> None:
    """题面：「完成后标准错误和标准输出均无字符」"""
    process, _, _, report = invoke(tmp_path, build_payload())
    assert process.returncode == 0
    assert process.stdout == ""
    assert process.stderr == ""
    assert report.is_file()

def test_missing_argument_returns_three(tmp_path: Path) -> None:
    """题面：「异常收口时退出码为 3」"""
    report_path = tmp_path / "out" / "case_02.json"
    process = subprocess.run(
        [sys.executable, ENTRY, "--scenario", str(tmp_path / "scenario.json")],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_malformed_json_returns_three(tmp_path: Path) -> None:
    """题面：「顶层六项：`case_id`、`products`、`resources`、`schema_version`、`settlements`、`unit`」"""
    process, _, _, report_path = invoke(tmp_path, raw="{")
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stdout == ""

def test_duplicate_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「顶层六项：`case_id`、`products`、`resources`、`schema_version`、`settlements`、`unit`」"""
    raw = (
        '{"case_id":"A-B-C","case_id":"D-E-F","products":[],"resources":[],'
        '"schema_version":1,"settlements":[],"unit":"batch"}'
    )
    process, _, _, report_path = invoke(tmp_path, raw=raw)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_top_level_array_returns_three(tmp_path: Path) -> None:
    """题面：「顶层六项：`case_id`、`products`、`resources`、`schema_version`、`settlements`、`unit`」"""
    process, _, _, report_path = invoke(tmp_path, raw="[]")
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_missing_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「顶层六项：`case_id`、`products`、`resources`、`schema_version`、`settlements`、`unit`」"""
    payload = build_payload()
    del payload["unit"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_extra_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「顶层六项：`case_id`、`products`、`resources`、`schema_version`、`settlements`、`unit`」"""
    payload = build_payload()
    payload["draft_note"] = "unused"
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_schema_version_bool_returns_three(tmp_path: Path) -> None:
    """题面：「版本号写整数 1」"""
    payload = build_payload()
    payload["schema_version"] = True
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_unit_value_returns_three(tmp_path: Path) -> None:
    """题面：「其中版本号写整数 1，单位写 `"batch"`」"""
    payload = build_payload()
    payload["unit"] = "kg"
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_case_id_shape_returns_three(tmp_path: Path) -> None:
    """题面：「案例编号由三段非空文字组成」"""
    process, _, _, report_path = invoke(tmp_path, build_payload(case_id="ONE-PART"))
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_too_few_products_returns_three(tmp_path: Path) -> None:
    """题面：「制造项数量 4 到 9」"""
    payload = build_payload()
    payload["products"] = payload["products"][:3]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_duplicate_product_returns_three(tmp_path: Path) -> None:
    """题面：「标识不许重名」"""
    payload = build_payload()
    payload["products"][1]["id"] = payload["products"][0]["id"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_revenue_bool_returns_three(tmp_path: Path) -> None:
    """题面：「单批收益须大于 0 且有限」"""
    payload = build_payload()
    payload["products"][0]["revenue"] = True
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_max_batches_zero_returns_three(tmp_path: Path) -> None:
    """题面：「上限批数落在 1..4」"""
    payload = build_payload()
    payload["products"][0]["max_batches"] = 0
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_max_batches_upper_boundary_accepted(tmp_path: Path) -> None:
    """题面：「上限批数落在 1..4」"""
    payload = build_payload()
    payload["products"][0]["max_batches"] = 4
    process, _, _, report = invoke(tmp_path, payload)
    assert process.returncode == 0
    assert report.is_file()

def test_resource_count_returns_three(tmp_path: Path) -> None:
    """题面：「公用资源有三份」"""
    payload = build_payload()
    payload["resources"] = payload["resources"][:2]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_capacity_zero_returns_three(tmp_path: Path) -> None:
    """题面：「容量字段落在 1..80」"""
    payload = build_payload()
    payload["resources"][0]["capacity"] = 0
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_too_few_settlements_returns_three(tmp_path: Path) -> None:
    """题面：「历史回执保留 6 到 12 份」"""
    payload = build_payload()
    payload["settlements"] = payload["settlements"][:5]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_duplicate_observation_returns_three(tmp_path: Path) -> None:
    """题面：「回执号不许重名」"""
    payload = build_payload()
    payload["settlements"][1]["observation"] = payload["settlements"][0]["observation"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_allocation_missing_product_returns_three(tmp_path: Path) -> None:
    """题面：「批次向量的键与产品对齐」"""
    payload = build_payload()
    del payload["settlements"][2]["allocation"]["V-CAL"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_allocation_bool_returns_three(tmp_path: Path) -> None:
    """题面：「批次向量的键与产品对齐」"""
    payload = build_payload()
    payload["settlements"][2]["allocation"]["V-CAL"] = True
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_usage_extra_resource_returns_three(tmp_path: Path) -> None:
    """题面：「消耗向量的键与资源对齐」"""
    payload = build_payload()
    payload["settlements"][0]["usage"]["waste"] = 1.0
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_yield_matrix_recovered(tmp_path: Path) -> None:
    """题面：「参考实现应对全部观测做联合最小二乘反演，得到三行产品消耗矩阵」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    matrix = read_report(report_path)["yield_matrix"]
    assert process.returncode == 0
    assert matrix["steam"]["V-HEX"] == pytest.approx(2.0, abs=1e-7)
    assert matrix["coolant"]["V-ORB"] == pytest.approx(3.0, abs=1e-7)
    assert matrix["catalyst"]["V-LUX"] == pytest.approx(3.0, abs=1e-7)

def test_allocations_match_global_optimum(tmp_path: Path) -> None:
    """题面：「选择目标是在三行资源容量都不超限时最大化产品收益总和」"""
    payload = build_payload()
    expected, _, _ = expected_plan(payload)
    process, _, _, report_path = invoke(tmp_path, payload)
    report = read_report(report_path)
    assert process.returncode == 0
    assert report["allocations"] == expected

def test_objective_matches_selected_batches(tmp_path: Path) -> None:
    """题面：「`objective` 汇总选中收益」"""
    payload = build_payload()
    process, _, _, report_path = invoke(tmp_path, payload)
    report = read_report(report_path)
    lookup = {item["id"]: item["revenue"] for item in payload["products"]}
    expected = sum(report["allocations"][name] * lookup[name] for name in lookup)
    assert process.returncode == 0
    assert report["objective"] == pytest.approx(expected, abs=1e-9)

def test_resource_usage_matches_yield_matrix(tmp_path: Path) -> None:
    """题面：「`resource_usage` 按反演矩阵复算」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    report = read_report(report_path)
    recomputed = {
        resource_id: sum(
            report["allocations"][product_id] * loading
            for product_id, loading in report["yield_matrix"][resource_id].items()
        )
        for resource_id in report["yield_matrix"]
    }
    assert process.returncode == 0
    assert report["resource_usage"] == pytest.approx(recomputed, abs=1e-8)

def test_report_exact_keys(tmp_path: Path) -> None:
    """题面：「方案 JSON 使用 UTF-8 编码，根层按区域组织」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    report = read_report(report_path)
    assert process.returncode == 0
    assert set(report) == {
        "allocations",
        "case_id",
        "checksum",
        "ledger_entry",
        "objective",
        "resource_usage",
        "schema_version",
        "solver",
        "yield_matrix",
    }

def test_solver_declares_branch_and_bound(tmp_path: Path) -> None:
    """题面：「| 求解 | `solver` 只有 `nodes` 与 `strategy`，后者写 `"branch_and_bound"` |」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    solver = read_report(report_path)["solver"]
    assert process.returncode == 0
    assert set(solver) == {"nodes", "strategy"}
    assert solver["strategy"] == "branch_and_bound"
    assert solver["nodes"] > 0

def test_compact_json_and_trailing_newline(tmp_path: Path) -> None:
    """题面：「输出摘要采用字典序键、无空格布局，并在末尾放一个换行」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    raw = report_path.read_bytes()
    assert process.returncode == 0
    assert raw == compact(json.loads(raw))

def test_checksum_matches_result(tmp_path: Path) -> None:
    """题面：「`checksum` 只覆盖 `ledger_entry` 以外的业务字段，并按紧凑 JSON 取摘要」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    report = read_report(report_path)
    base = {key: value for key, value in report.items() if key not in {"checksum", "ledger_entry"}}
    assert process.returncode == 0
    assert report["checksum"] == hashlib.sha256(compact(base)).hexdigest()

def test_journal_first_run_appends_one(tmp_path: Path) -> None:
    """题面：「重复导入相同情景且报告字节未变时，历史结算流水保持原样」"""
    process, _, journal, _ = invoke(tmp_path, build_payload())
    assert process.returncode == 0
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1

def test_journal_repeat_is_idempotent(tmp_path: Path) -> None:
    """题面：「重复导入相同情景且报告字节未变时，历史结算流水保持原样」"""
    first, scenario, journal, report = invoke(tmp_path, build_payload())
    before = report.read_bytes()
    second = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--scenario",
            str(scenario),
            "--journal",
            str(journal),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert first.returncode == 0
    assert second.returncode == 0
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1
    assert report.read_bytes() == before

def test_failure_preserves_existing_outputs(tmp_path: Path) -> None:
    """题面：「不修改旧报告和流水，也不留下临时文件」"""
    success, scenario, journal, report = invoke(tmp_path, build_payload())
    journal_before = journal.read_bytes()
    report_before = report.read_bytes()
    scenario.write_text("{broken", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--scenario",
            str(scenario),
            "--journal",
            str(journal),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert success.returncode == 0
    assert failed.returncode == 3
    assert journal.read_bytes() == journal_before
    assert report.read_bytes() == report_before

def test_normal_case_finishes_within_three_seconds(tmp_path: Path) -> None:
    """题面：「单次处理耗时上限为 3 秒」"""
    scenario = write_case(tmp_path, build_payload())
    journal_path = tmp_path / "state" / "timed.jsonl"
    report_path = tmp_path / "out" / "timed.json"
    start = time.monotonic()
    process = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--scenario",
            str(scenario),
            "--journal",
            str(journal_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    elapsed = time.monotonic() - start
    assert process.returncode == 0
    assert report_path.is_file()
    assert elapsed < 3.0
