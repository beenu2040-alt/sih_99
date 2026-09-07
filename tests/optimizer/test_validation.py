import pytest
import pandas as pd
from src.optimizer.validate import validate_schedule
from src.optimizer.schedule import build_schedule_df, group_into_blocks
from tests.optimizer.test_constraints import make_task_row, build_and_solve
from src.optimizer.preprocessing import build_occupancy_lookup


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


# ---------------------------------------------------------------------------
# Added Regression Tests
# ---------------------------------------------------------------------------

def test_test1_no_infrastructure_no_conflict():
    """TEST 1: Task requires no infrastructure blocks and overlaps a train."""
    t1 = make_task_row("T1", power_block_required=False, signal_block_required=False, track_block_required=False)
    occ = [{"section_id": "SEC001", "date": "2025-06-01", "time_window_start": "06:00", "time_window_end": "08:00", "occupied": True, "train_count": 1, "traffic_density": 0.5}]
    ctx, res = build_and_solve([t1], occ)
    
    tv = ctx.task_vars["T1"]
    # Force overlap
    res.solution["T1"]["start_min"] = tv.earliest_start_min
    res.solution["T1"]["end_min"] = tv.earliest_start_min + tv.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    occ_lookup = build_occupancy_lookup(pd.DataFrame(occ))
    
    report = validate_schedule(ctx, res, occ_lookup, sched_df, blocks_df)
    assert report["train_conflicts_found"] == 0

def test_test2_track_block_conflict():
    """TEST 2: Task requires track block and overlaps an occupied train interval."""
    t1 = make_task_row("T1", track_block_required=True, power_block_required=False, signal_block_required=False)
    occ = [{"section_id": "SEC001", "date": "2025-06-01", "time_window_start": "06:00", "time_window_end": "08:00", "occupied": True, "train_count": 1, "traffic_density": 0.5}]
    ctx, res = build_and_solve([t1], occ)
    
    tv = ctx.task_vars["T1"]
    # Force overlap
    res.solution["T1"]["start_min"] = tv.earliest_start_min
    res.solution["T1"]["end_min"] = tv.earliest_start_min + tv.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    occ_lookup = build_occupancy_lookup(pd.DataFrame(occ))
    
    report = validate_schedule(ctx, res, occ_lookup, sched_df, blocks_df)
    assert report["valid"] is False
    assert report["train_conflicts_found"] == 1

def test_test3_signal_block_conflict():
    """TEST 3: Task requires signal block and overlaps an occupied train interval."""
    t1 = make_task_row("T1", track_block_required=False, power_block_required=False, signal_block_required=True)
    occ = [{"section_id": "SEC001", "date": "2025-06-01", "time_window_start": "06:00", "time_window_end": "08:00", "occupied": True, "train_count": 1, "traffic_density": 0.5}]
    ctx, res = build_and_solve([t1], occ)
    
    tv = ctx.task_vars["T1"]
    # Force overlap
    res.solution["T1"]["start_min"] = tv.earliest_start_min
    res.solution["T1"]["end_min"] = tv.earliest_start_min + tv.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    occ_lookup = build_occupancy_lookup(pd.DataFrame(occ))
    
    report = validate_schedule(ctx, res, occ_lookup, sched_df, blocks_df)
    assert report["valid"] is False
    assert report["train_conflicts_found"] == 1

def test_test4_power_block_conflict():
    """TEST 4: Task requires power block and overlaps an occupied train interval."""
    t1 = make_task_row("T1", track_block_required=False, power_block_required=True, signal_block_required=False)
    occ = [{"section_id": "SEC001", "date": "2025-06-01", "time_window_start": "06:00", "time_window_end": "08:00", "occupied": True, "train_count": 1, "traffic_density": 0.5}]
    ctx, res = build_and_solve([t1], occ)
    
    tv = ctx.task_vars["T1"]
    # Force overlap
    res.solution["T1"]["start_min"] = tv.earliest_start_min
    res.solution["T1"]["end_min"] = tv.earliest_start_min + tv.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    occ_lookup = build_occupancy_lookup(pd.DataFrame(occ))
    
    report = validate_schedule(ctx, res, occ_lookup, sched_df, blocks_df)
    assert report["valid"] is False
    assert report["train_conflicts_found"] == 1

def test_test5_different_resources_no_conflict():
    """TEST 5: Two blocks overlap in the same section but use different resources."""
    t1 = make_task_row("T1", track_block_required=True, power_block_required=False, signal_block_required=False, can_combine=False)
    t2 = make_task_row("T2", track_block_required=False, power_block_required=False, signal_block_required=True, can_combine=False)
    ctx, res = build_and_solve([t1, t2])
    
    tv1, tv2 = ctx.task_vars["T1"], ctx.task_vars["T2"]
    res.solution["T1"]["start_min"] = tv1.earliest_start_min
    res.solution["T1"]["end_min"] = tv1.earliest_start_min + tv1.duration_min
    res.solution["T2"]["start_min"] = tv2.earliest_start_min
    res.solution["T2"]["end_min"] = tv2.earliest_start_min + tv2.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["resource_conflicts_found"] == 0

def test_test6_same_resource_conflict():
    """TEST 6: Two blocks overlap in the same section and use the same resource."""
    t1 = make_task_row("T1", track_block_required=True, power_block_required=False, signal_block_required=False, can_combine=False)
    t2 = make_task_row("T2", track_block_required=True, power_block_required=False, signal_block_required=False, can_combine=False)
    ctx, res = build_and_solve([t1, t2])
    
    tv1, tv2 = ctx.task_vars["T1"], ctx.task_vars["T2"]
    res.solution["T1"]["start_min"] = tv1.earliest_start_min
    res.solution["T1"]["end_min"] = tv1.earliest_start_min + tv1.duration_min
    res.solution["T2"]["start_min"] = tv2.earliest_start_min
    res.solution["T2"]["end_min"] = tv2.earliest_start_min + tv2.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["valid"] is False
    assert report["resource_conflicts_found"] >= 1

def test_test7_one_resource_one_none_no_conflict():
    """TEST 7: One block uses a resource and another requires no resource."""
    t1 = make_task_row("T1", track_block_required=True, power_block_required=False, signal_block_required=False, can_combine=False)
    t2 = make_task_row("T2", track_block_required=False, power_block_required=False, signal_block_required=False, can_combine=False)
    ctx, res = build_and_solve([t1, t2])
    
    tv1, tv2 = ctx.task_vars["T1"], ctx.task_vars["T2"]
    res.solution["T1"]["start_min"] = tv1.earliest_start_min
    res.solution["T1"]["end_min"] = tv1.earliest_start_min + tv1.duration_min
    res.solution["T2"]["start_min"] = tv2.earliest_start_min
    res.solution["T2"]["end_min"] = tv2.earliest_start_min + tv2.duration_min
    
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    
    report = validate_schedule(ctx, res, {}, sched_df, blocks_df)
    assert report["resource_conflicts_found"] == 0
