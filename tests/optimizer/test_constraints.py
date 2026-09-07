import pytest
import pandas as pd
from src.optimizer.preprocessing import preprocess
from src.optimizer.model import build_model
from src.optimizer.constraints import add_all_constraints
from src.optimizer.objective import add_objective
from src.optimizer.solver import solve

def make_task_row(task_id, section_id='SEC001', department='Engineering',
                  priority_score=80.0, duration_hours=2.0,
                  earliest_start='2025-06-01 06:00',
                  latest_end='2025-06-01 22:00',
                  deadline='2025-06-01',
                  train_conflict_cost=100.0, traffic_density=0.3,
                  power_block_required=False, signal_block_required=False,
                  track_block_required=True, can_combine=True,
                  predicted_priority_score=80.0,
                  predicted_priority_class='CRITICAL'):
    return {
        'task_id': task_id, 'section_id': section_id, 'department': department,
        'priority_score': priority_score, 'duration_hours': duration_hours,
        'earliest_start': earliest_start, 'latest_end': latest_end,
        'deadline': deadline, 'train_conflict_cost': train_conflict_cost,
        'traffic_density': traffic_density,
        'power_block_required': power_block_required,
        'signal_block_required': signal_block_required,
        'track_block_required': track_block_required,
        'can_combine': can_combine,
        'predicted_priority_score': predicted_priority_score,
        'predicted_priority_class': predicted_priority_class,
    }

def build_and_solve(task_rows, occupancy_rows=None, time_limit=10):
    tasks_df = pd.DataFrame(task_rows)
    occ_df = pd.DataFrame(occupancy_rows or [], columns=['section_id','date','time_window_start','time_window_end','occupied','train_count','traffic_density'])
    tasks_df, occ_lookup = preprocess(tasks_df, occ_df)
    ctx = build_model(tasks_df)
    add_all_constraints(ctx, occ_lookup)
    add_objective(ctx)
    result = solve(ctx, time_limit=time_limit)
    return ctx, result

def test_basic_scheduling():
    ctx, res = build_and_solve([make_task_row("T1")])
    assert res.is_feasible
    assert res.solution["T1"]["scheduled"] == 1

def test_deadline_constraint():
    """Task must finish before its deadline."""
    ctx, res = build_and_solve([make_task_row(
        "T1",
        earliest_start="2025-06-01 06:00",
        latest_end="2025-06-02 22:00",
        deadline="2025-06-01",
        duration_hours=2.0,
    )])
    sol = res.solution["T1"]
    tv = ctx.task_vars["T1"]
    assert sol["scheduled"] == 1
    assert sol["end_min"] <= tv.deadline_min

def test_train_conflict():
    task = make_task_row("T1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", deadline="2025-01-01", duration_hours=2.0)
    occ = [{"section_id": "SEC001", "date": "2025-01-01", "time_window_start": "10:00", "time_window_end": "12:00", "occupied": True, "train_count": 1, "traffic_density": 0.5}]
    ctx, res = build_and_solve([task], occ)
    sol = res.solution["T1"]
    assert sol["start_min"] >= ctx.task_vars["T1"].earliest_start_min + 120

def test_resource_conflict():
    task1 = make_task_row("T1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", deadline="2025-01-01", duration_hours=2.0, track_block_required=True)
    task2 = make_task_row("T2", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 14:00", deadline="2025-01-01", duration_hours=2.0, track_block_required=True)
    ctx, res = build_and_solve([task1, task2])
    s1, s2 = res.solution["T1"], res.solution["T2"]
    assert s1["end_min"] <= s2["start_min"] or s2["end_min"] <= s1["start_min"]

def test_multi_section():
    task1 = make_task_row("T1", section_id="S1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01", duration_hours=2.0)
    task2 = make_task_row("T2", section_id="S2", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01", duration_hours=2.0)
    ctx, res = build_and_solve([task1, task2])
    s1, s2 = res.solution["T1"], res.solution["T2"]
    assert s1["scheduled"] == 1 and s2["scheduled"] == 1
    assert s1["start_min"] == s2["start_min"]
