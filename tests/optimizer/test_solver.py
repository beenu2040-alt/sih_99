import pytest
from tests.optimizer.test_constraints import make_task_row, build_and_solve

def test_solver_feasible():
    ctx, res = build_and_solve([make_task_row("T1")])
    assert res.status in ("OPTIMAL", "FEASIBLE")

def test_solver_infeasible_window():
    task = make_task_row("T1", earliest_start="2025-01-01 10:00", latest_end="2025-01-01 11:00", duration_hours=2.0)
    ctx, res = build_and_solve([task])
    assert res.solution["T1"]["scheduled"] == 0

def test_solution_keys():
    ctx, res = build_and_solve([make_task_row("T1")])
    sol = res.solution["T1"]
    assert "scheduled" in sol
    assert "start_min" in sol
    assert "end_min" in sol
