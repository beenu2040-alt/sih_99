import pytest
from src.optimizer.validate import validate_schedule
from src.optimizer.schedule import build_schedule_df, group_into_blocks
from tests.optimizer.test_constraints import make_task_row, build_and_solve


def test_valid_schedule():
    """Test 10: A correctly solved schedule passes independent validation."""
    t1 = make_task_row("T1")
    ctx, res = build_and_solve([t1])
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is True


def test_impossible_task_unscheduled():
    """Test 8: A task whose window can't fit its duration is unscheduled."""
    t1 = make_task_row(
        "T1",
        earliest_start="2025-01-01 10:00",
        latest_end="2025-01-01 11:00",
        deadline="2025-01-01",
        duration_hours=2.0,
    )
    ctx, res = build_and_solve([t1])
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is True
    assert len(sched_df) == 0
    assert res.solution["T1"]["scheduled"] == 0


def test_validator_catches_time_window_violation():
    """Fabricate a solution where start < earliest_start."""
    t1 = make_task_row("T1")
    ctx, res = build_and_solve([t1])

    # Tamper with the solution to create a violation
    tv = ctx.task_vars["T1"]
    res.solution["T1"]["start_min"] = tv.earliest_start_min - 100
    res.solution["T1"]["end_min"] = tv.earliest_start_min - 100 + tv.duration_min

    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is False
    assert any("start" in e for e in report["errors"])


def test_validator_catches_duration_mismatch():
    """Fabricate a solution where duration doesn't match."""
    t1 = make_task_row("T1")
    ctx, res = build_and_solve([t1])

    tv = ctx.task_vars["T1"]
    # Keep start valid but make end wrong
    res.solution["T1"]["end_min"] = res.solution["T1"]["start_min"] + tv.duration_min + 30

    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is False
    assert report["duration_violations"] > 0


def test_validator_catches_task_coverage_gap():
    """Remove a task from the solution dict to simulate a silent drop."""
    t1 = make_task_row("T1")
    t2 = make_task_row("T2")
    ctx, res = build_and_solve([t1, t2])

    # Remove T2 from solution
    del res.solution["T2"]

    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is False
    assert any("dropped" in e for e in report["errors"])
