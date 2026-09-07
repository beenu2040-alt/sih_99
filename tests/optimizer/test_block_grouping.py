import pytest
from src.optimizer.schedule import group_into_blocks, build_schedule_df, assign_block_ids
from tests.optimizer.test_constraints import make_task_row, build_and_solve

def test_compatible_tasks_one_block():
    t1 = make_task_row("T1", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", track_block_required=False, can_combine=True)
    t2 = make_task_row("T2", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", track_block_required=False, can_combine=True)
    ctx, res = build_and_solve([t1, t2])
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    assert len(blocks_df) == 1
    assert "T1" in blocks_df.iloc[0]["task_ids"] and "T2" in blocks_df.iloc[0]["task_ids"]

def test_incompatible_tasks():
    t1 = make_task_row("T1", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", track_block_required=False, can_combine=False)
    t2 = make_task_row("T2", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", track_block_required=False, can_combine=False)
    ctx, res = build_and_solve([t1, t2])
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    assert len(blocks_df) == 2

def test_block_id_assigned():
    t1 = make_task_row("T1", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", can_combine=True)
    ctx, res = build_and_solve([t1])
    sched_df = build_schedule_df(ctx, res)
    blocks_df = group_into_blocks(sched_df, ctx)
    sched_df = assign_block_ids(sched_df, blocks_df)
    assert "block_id" in sched_df.columns
    assert sched_df.iloc[0]["block_id"] == blocks_df.iloc[0]["block_id"]
