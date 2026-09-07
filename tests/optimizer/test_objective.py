import pytest
from tests.optimizer.test_constraints import make_task_row, build_and_solve

def test_high_priority_wins():
    task1 = make_task_row("T1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01", predicted_priority_score=90)
    task2 = make_task_row("T2", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01", predicted_priority_score=50)
    
    ctx, res = build_and_solve([task1, task2])
    assert res.solution["T1"]["scheduled"] == 1
    assert res.solution["T2"]["scheduled"] == 0

def test_urgency_impact():
    t1 = make_task_row("T1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01 12:00", predicted_priority_score=50)
    t2 = make_task_row("T2", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 12:00", deadline="2025-01-01 14:00", predicted_priority_score=50)
    ctx, res = build_and_solve([t1, t2])
    assert res.solution["T1"]["scheduled"] == 1
    assert res.solution["T2"]["scheduled"] == 0
