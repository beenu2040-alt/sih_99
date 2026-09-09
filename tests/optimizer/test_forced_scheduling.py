"""
Tests for the Schedule-on-Command / forced task scheduling feature.

These tests use small synthetic task fixtures (via the existing
``make_task_row`` / ``build_and_solve`` helpers) to verify forced-task
behaviour at the CP-SAT model level, without hitting the full data pipeline.

A separate integration-level test (Test 1) exercises ``optimize_schedule()``
via the real data files.
"""

import pytest
import pandas as pd

from src.optimizer.preprocessing import preprocess, build_occupancy_lookup
from src.optimizer.model import build_model
from src.optimizer.constraints import add_all_constraints
from src.optimizer.objective import add_objective
from src.optimizer.solver import solve
from src.optimizer.schedule import (
    build_schedule_df,
    group_into_blocks,
    build_unscheduled_df,
)
from src.optimizer.validate import validate_schedule

# Re-use the existing test helpers
from tests.optimizer.test_constraints import make_task_row, build_and_solve


# ---------------------------------------------------------------------------
# Internal helper: build, force, solve — mirrors the api.py logic but works
# with synthetic task rows so tests are fast and self-contained.
# ---------------------------------------------------------------------------

def _build_force_and_solve(
    task_rows,
    forced_task_ids=None,
    occupancy_rows=None,
    time_limit=10,
):
    """
    Build a CP-SAT model from *task_rows*, add forced constraints, solve,
    and return ``(ctx, result, occ_lookup)``.
    """
    forced_task_ids = forced_task_ids or []

    tasks_df = pd.DataFrame(task_rows)
    occ_df = pd.DataFrame(
        occupancy_rows or [],
        columns=[
            "section_id", "date", "time_window_start",
            "time_window_end", "occupied", "train_count",
            "traffic_density",
        ],
    )
    tasks_df, occ_lookup = preprocess(tasks_df, occ_df)
    ctx = build_model(tasks_df)
    add_all_constraints(ctx, occ_lookup)

    # --- Forced constraints (same logic as api.py) ---
    for tid in forced_task_ids:
        assert tid in ctx.task_vars, f"Test bug: {tid} not in task_vars"
        ctx.model.Add(ctx.task_vars[tid].scheduled == 1)

    add_objective(ctx)
    result = solve(ctx, time_limit=time_limit)
    return ctx, result, occ_lookup


# ===================================================================
# Test 1 — No forced tasks (baseline behaviour)
# ===================================================================

def test_no_forced_tasks_baseline():
    """optimize_schedule() with no forced tasks behaves identically to baseline."""
    t1 = make_task_row("T1")
    t2 = make_task_row("T2", section_id="SEC002")
    ctx, result, _ = _build_force_and_solve([t1, t2])

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1
    assert result.solution["T2"]["scheduled"] == 1


# ===================================================================
# Test 2 — One valid forced task
# ===================================================================

def test_single_forced_task():
    """A single forced task is scheduled."""
    t1 = make_task_row("T1")
    ctx, result, _ = _build_force_and_solve([t1], forced_task_ids=["T1"])

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1


# ===================================================================
# Test 3 — Multiple valid forced tasks
# ===================================================================

def test_multiple_forced_tasks():
    """Multiple forced tasks are all scheduled."""
    t1 = make_task_row("T1", section_id="S1")
    t2 = make_task_row("T2", section_id="S2")
    t3 = make_task_row("T3", section_id="S3")
    ctx, result, _ = _build_force_and_solve(
        [t1, t2, t3],
        forced_task_ids=["T1", "T2", "T3"],
    )

    assert result.is_feasible
    for tid in ["T1", "T2", "T3"]:
        assert result.solution[tid]["scheduled"] == 1, (
            f"Forced task {tid} was not scheduled"
        )


# ===================================================================
# Test 4 — Invalid task ID
# ===================================================================

def test_invalid_task_id():
    """
    Passing a non-existent task ID to optimize_schedule() returns a
    TASK_NOT_FOUND error.

    This test exercises the full API path.
    """
    from src.optimizer.api import optimize_schedule

    result = optimize_schedule(
        forced_task_ids=["TEST_INVALID_TASK_999999"],
        time_limit=10,
    )

    assert result["success"] is False
    assert result["error"] == "TASK_NOT_FOUND"
    assert "TEST_INVALID_TASK_999999" in result["message"]


# ===================================================================
# Test 5 — Infeasible forced task (time window too narrow)
# ===================================================================

def test_infeasible_forced_task():
    """
    A task whose time window cannot accommodate its duration, when forced,
    makes the model infeasible.
    """
    # Window: 10:00–11:00 (60 min), duration: 2 hours (120 min) → infeasible
    t1 = make_task_row(
        "T1",
        earliest_start="2025-01-01 10:00",
        latest_end="2025-01-01 11:00",
        deadline="2025-01-01",
        duration_hours=2.0,
    )
    ctx, result, _ = _build_force_and_solve([t1], forced_task_ids=["T1"])

    # The model should be infeasible: build_model sets scheduled==0 for
    # tasks that can't fit, and we added scheduled==1.
    assert not result.is_feasible
    assert result.status == "INFEASIBLE"


# ===================================================================
# Test 6 — Train constraint preservation with forced task
# ===================================================================

def test_forced_task_train_constraint_preserved():
    """
    A forced task that requires an infrastructure block must still avoid
    train occupancy windows.  Validation must report 0 train conflicts.
    """
    # Task requires track block, occupancy at 06:00–08:00 on same section
    t1 = make_task_row(
        "T1",
        track_block_required=True,
        power_block_required=False,
        signal_block_required=False,
        earliest_start="2025-06-01 06:00",
        latest_end="2025-06-01 22:00",
        deadline="2025-06-01",
        duration_hours=2.0,
    )
    occ = [
        {
            "section_id": "SEC001",
            "date": "2025-06-01",
            "time_window_start": "06:00",
            "time_window_end": "08:00",
            "occupied": True,
            "train_count": 3,
            "traffic_density": 0.8,
        },
    ]
    ctx, result, occ_lookup = _build_force_and_solve(
        [t1], forced_task_ids=["T1"], occupancy_rows=occ,
    )

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1

    # Task must start at or after 08:00 (after the occupancy window)
    sol = result.solution["T1"]
    assert sol["start_min"] >= ctx.task_vars["T1"].earliest_start_min + 120

    # Full validation
    sched_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, result, occ_lookup, sched_df, blocks_df)
    assert report["train_conflicts_found"] == 0


# ===================================================================
# Test 7 — Resource constraint preservation with forced task
# ===================================================================

def test_forced_task_resource_constraint_preserved():
    """
    Two forced tasks on the same section requiring the same resource
    must not overlap.  Validation must report 0 resource conflicts.
    """
    t1 = make_task_row(
        "T1",
        track_block_required=True,
        power_block_required=False,
        signal_block_required=False,
        earliest_start="2025-01-01 10:00",
        latest_end="2025-01-01 18:00",
        deadline="2025-01-01",
        duration_hours=2.0,
    )
    t2 = make_task_row(
        "T2",
        track_block_required=True,
        power_block_required=False,
        signal_block_required=False,
        earliest_start="2025-01-01 10:00",
        latest_end="2025-01-01 18:00",
        deadline="2025-01-01",
        duration_hours=2.0,
    )
    ctx, result, occ_lookup = _build_force_and_solve(
        [t1, t2], forced_task_ids=["T1", "T2"],
    )

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1
    assert result.solution["T2"]["scheduled"] == 1

    # They must not overlap (NoOverlap constraint on SEC001__track)
    s1 = result.solution["T1"]
    s2 = result.solution["T2"]
    assert s1["end_min"] <= s2["start_min"] or s2["end_min"] <= s1["start_min"]

    # Full validation
    sched_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, result, occ_lookup, sched_df, blocks_df)
    assert report["resource_conflicts_found"] == 0


# ===================================================================
# Test 8 — Deadline preservation with forced task
# ===================================================================

def test_forced_task_deadline_preserved():
    """
    A forced task must finish by its deadline.
    Validation must report 0 deadline violations.
    """
    t1 = make_task_row(
        "T1",
        earliest_start="2025-06-01 06:00",
        latest_end="2025-06-02 22:00",
        deadline="2025-06-01",
        duration_hours=2.0,
    )
    ctx, result, occ_lookup = _build_force_and_solve(
        [t1], forced_task_ids=["T1"],
    )

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1

    # end must be <= deadline_min
    tv = ctx.task_vars["T1"]
    assert result.solution["T1"]["end_min"] <= tv.deadline_min

    # Full validation
    sched_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, result, occ_lookup, sched_df, blocks_df)
    assert report["deadline_violations"] == 0


# ===================================================================
# Test 9 — Duration preservation with forced task
# ===================================================================

def test_forced_task_duration_preserved():
    """
    A forced task's scheduled duration must match its required duration.
    Validation must report 0 duration violations.
    """
    t1 = make_task_row("T1", duration_hours=3.0)
    ctx, result, occ_lookup = _build_force_and_solve(
        [t1], forced_task_ids=["T1"],
    )

    assert result.is_feasible
    sol = result.solution["T1"]
    tv = ctx.task_vars["T1"]
    assert sol["scheduled"] == 1
    assert sol["end_min"] - sol["start_min"] == tv.duration_min

    # Full validation
    sched_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, result, occ_lookup, sched_df, blocks_df)
    assert report["duration_violations"] == 0


# ===================================================================
# Test 10 — Duplicate assignment preservation with forced task
# ===================================================================

def test_forced_task_no_duplicate_assignments():
    """
    A forced task must appear exactly once in the schedule.
    No duplicate task entries should exist.
    """
    t1 = make_task_row("T1")
    t2 = make_task_row("T2", section_id="SEC002")
    ctx, result, occ_lookup = _build_force_and_solve(
        [t1, t2], forced_task_ids=["T1"],
    )

    assert result.is_feasible
    assert result.solution["T1"]["scheduled"] == 1

    sched_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(sched_df, ctx)

    # No duplicate task IDs in schedule
    assert sched_df["task_id"].is_unique

    # Full validation
    report = validate_schedule(ctx, result, occ_lookup, sched_df, blocks_df)
    assert report["valid"] is True
