"""
CP-SAT solver wrapper for the maintenance block optimizer.

Configures the solver, runs it, and returns a structured result.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from ortools.sat.python import cp_model

from . import config as cfg
from .model import ModelContext

logger = logging.getLogger(__name__)

# Map CP-SAT status codes to human-readable strings
_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


@dataclass
class SolverResult:
    """Container for solver output."""

    status: str
    status_code: int
    objective_value: int
    best_bound: int
    wall_time_seconds: float
    num_conflicts: int
    num_branches: int

    # Extracted solution values  {task_id: {"scheduled": 0|1, "start": int, "end": int}}
    solution: Dict[str, Dict] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")


def solve(
    ctx: ModelContext,
    time_limit: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> SolverResult:
    """
    Run the CP-SAT solver and extract the solution.

    Parameters
    ----------
    ctx : ModelContext
        Fully built model with constraints and objective.
    time_limit : int, optional
        Override ``SOLVER_TIME_LIMIT_SECONDS``.
    num_workers : int, optional
        Override ``NUM_WORKERS``.
    seed : int, optional
        Override ``RANDOM_SEED``.

    Returns
    -------
    SolverResult
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit or cfg.SOLVER_TIME_LIMIT_SECONDS
    solver.parameters.num_workers = num_workers or cfg.NUM_WORKERS
    solver.parameters.random_seed = seed if seed is not None else cfg.RANDOM_SEED

    logger.info(
        "Starting CP-SAT solver (time_limit=%ds, workers=%d, seed=%d) …",
        solver.parameters.max_time_in_seconds,
        solver.parameters.num_workers,
        solver.parameters.random_seed,
    )

    wall_start = time.perf_counter()
    status_code = solver.solve(ctx.model)
    wall_time = time.perf_counter() - wall_start

    status_name = _STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")

    result = SolverResult(
        status=status_name,
        status_code=status_code,
        objective_value=int(solver.objective_value) if status_code in (
            cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
        best_bound=int(solver.best_objective_bound) if status_code in (
            cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
        wall_time_seconds=round(wall_time, 3),
        num_conflicts=solver.num_conflicts,
        num_branches=solver.num_branches,
    )

    logger.info(
        "Solver finished: status=%s  objective=%d  bound=%d  "
        "time=%.2fs  conflicts=%d  branches=%d",
        result.status,
        result.objective_value,
        result.best_bound,
        result.wall_time_seconds,
        result.num_conflicts,
        result.num_branches,
    )

    # ---- Extract solution values ----
    if result.is_feasible:
        for tid, tv in ctx.task_vars.items():
            scheduled = solver.value(tv.scheduled)
            start = solver.value(tv.start) if scheduled else None
            end = solver.value(tv.end) if scheduled else None

            result.solution[tid] = {
                "scheduled": scheduled,
                "start_min": start,
                "end_min": end,
            }

        n_scheduled = sum(
            1 for v in result.solution.values() if v["scheduled"]
        )
        logger.info(
            "Solution extracted: %d scheduled, %d unscheduled",
            n_scheduled,
            len(result.solution) - n_scheduled,
        )
    else:
        logger.warning(
            "No feasible solution found (status=%s). "
            "All tasks will be marked unscheduled.",
            result.status,
        )
        for tid in ctx.task_vars:
            result.solution[tid] = {
                "scheduled": 0,
                "start_min": None,
                "end_min": None,
            }

    return result
