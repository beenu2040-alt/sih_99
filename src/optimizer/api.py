"""
Programmatic API for the OR-Tools CP-SAT Maintenance Block Optimizer.

Provides ``optimize_schedule()`` — the primary entry point for forced-task
scheduling (Schedule-on-Command).  This module wraps the existing pipeline
without modifying any existing optimizer module.

Usage::

    from src.optimizer.api import optimize_schedule

    # Normal optimization (identical to baseline)
    result = optimize_schedule()

    # Force specific tasks to be scheduled
    result = optimize_schedule(forced_task_ids=["TRC1015"])

    # Force multiple tasks
    result = optimize_schedule(forced_task_ids=["TRC1015", "TRC1021"])
"""

import logging
import time
from typing import Dict, List, Optional

from .data_loader import load_all
from .preprocessing import preprocess
from .model import build_model, ModelContext
from .constraints import add_all_constraints
from .objective import add_objective
from .solver import solve, SolverResult
from .schedule import (
    build_schedule_df,
    group_into_blocks,
    assign_block_ids,
    build_unscheduled_df,
    build_summary,
)
from .validate import validate_schedule

logger = logging.getLogger(__name__)


def optimize_schedule(
    forced_task_ids: Optional[List[str]] = None,
    *,
    time_limit: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Run the full optimization pipeline with optional forced-task scheduling.

    Parameters
    ----------
    forced_task_ids : list of str, optional
        Task IDs that **must** be scheduled.  The optimizer will add a
        hard constraint ``scheduled[task_id] == 1`` for each ID while
        preserving every existing constraint (train occupancy, resource
        no-overlap, time windows, durations, deadlines).

        If ``None`` or empty, the optimizer behaves identically to the
        existing baseline.

    time_limit : int, optional
        Solver time limit in seconds.  Falls back to config default.
    num_workers : int, optional
        Number of parallel solver workers.  Falls back to config default.
    seed : int, optional
        Random seed for solver reproducibility.

    Returns
    -------
    dict
        A result dictionary with at minimum:

        - ``success`` (bool)
        - ``error`` (str or None) — error code if ``success`` is False
        - ``message`` (str) — human-readable description
        - ``forced_task_ids`` (list) — echo of the requested forced tasks
        - ``schedule_df`` — scheduled tasks DataFrame (on success)
        - ``blocks_df`` — maintenance blocks DataFrame (on success)
        - ``unscheduled_df`` — unscheduled tasks DataFrame (on success)
        - ``summary`` — optimization summary dict (on success)
        - ``validation`` — validation report dict (on success)
        - ``solver_result`` — raw ``SolverResult`` (on success)
    """
    # Normalize forced_task_ids
    if forced_task_ids is None:
        forced_task_ids = []
    forced_task_ids = list(forced_task_ids)  # defensive copy

    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("API: Loading data …")
    tasks_df, occupancy_df = load_all()

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    logger.info("API: Preprocessing …")
    tasks_df, occ_lookup = preprocess(tasks_df, occupancy_df)

    # ------------------------------------------------------------------
    # 3. Build CP-SAT model
    # ------------------------------------------------------------------
    logger.info("API: Building CP-SAT model …")
    ctx = build_model(tasks_df)

    # ------------------------------------------------------------------
    # 4. Validate forced task IDs (before adding any constraints)
    # ------------------------------------------------------------------
    if forced_task_ids:
        for tid in forced_task_ids:
            if tid not in ctx.task_vars:
                logger.error("Forced task ID '%s' not found in task set.", tid)
                return {
                    "success": False,
                    "error": "TASK_NOT_FOUND",
                    "message": f"Task ID '{tid}' was not found.",
                    "task_id": tid,
                    "forced_task_ids": forced_task_ids,
                }

    # ------------------------------------------------------------------
    # 5. Add constraints (unchanged — all existing constraints preserved)
    # ------------------------------------------------------------------
    logger.info("API: Adding constraints …")
    add_all_constraints(ctx, occ_lookup)

    # ------------------------------------------------------------------
    # 6. Add forced-task constraints
    # ------------------------------------------------------------------
    if forced_task_ids:
        logger.info(
            "API: Adding forced scheduling constraints for %d task(s): %s",
            len(forced_task_ids),
            forced_task_ids,
        )
        for tid in forced_task_ids:
            tv = ctx.task_vars[tid]

            # Check if the model builder already marked this task as
            # infeasible (time window too narrow for duration).  In that
            # case, forcing scheduled==1 will create a guaranteed
            # contradiction — report it early.
            if not tv.is_feasible:
                logger.warning(
                    "Forced task '%s' has an infeasible time window "
                    "(window=%d min, duration=%d min).  The model will "
                    "be infeasible.",
                    tid,
                    tv.latest_end_min - tv.earliest_start_min,
                    tv.duration_min,
                )

            # Add the mandatory scheduling constraint.
            # This works even for infeasible tasks — CP-SAT will correctly
            # report INFEASIBLE because build_model already added
            # scheduled==0 for those tasks, and this adds scheduled==1.
            ctx.model.Add(tv.scheduled == 1)

    # ------------------------------------------------------------------
    # 7. Add objective (unchanged)
    # ------------------------------------------------------------------
    logger.info("API: Setting objective …")
    add_objective(ctx)

    # ------------------------------------------------------------------
    # 8. Solve
    # ------------------------------------------------------------------
    logger.info("API: Solving …")
    result = solve(
        ctx,
        time_limit=time_limit,
        num_workers=num_workers,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 9. Handle infeasible result
    # ------------------------------------------------------------------
    if not result.is_feasible:
        logger.warning(
            "API: Solver returned %s — no feasible schedule found.",
            result.status,
        )
        return {
            "success": False,
            "error": "INFEASIBLE",
            "message": (
                "No feasible maintenance window exists under current "
                "constraints."
            ),
            "forced_task_ids": forced_task_ids,
            "solver_status": result.status,
        }

    # ------------------------------------------------------------------
    # 10. Generate schedule outputs (unchanged pipeline)
    # ------------------------------------------------------------------
    logger.info("API: Generating schedule …")
    schedule_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(schedule_df, ctx)
    schedule_df = assign_block_ids(schedule_df, blocks_df)
    unscheduled_df = build_unscheduled_df(ctx, result)

    # ------------------------------------------------------------------
    # 11. Verify forced tasks are actually scheduled
    # ------------------------------------------------------------------
    if forced_task_ids:
        for tid in forced_task_ids:
            sol = result.solution.get(tid, {})
            if not sol.get("scheduled"):
                # This should never happen if the solver reports FEASIBLE
                # and we added scheduled==1, but defend against it.
                logger.error(
                    "BUG: Forced task '%s' is not scheduled despite "
                    "FEASIBLE solver status.",
                    tid,
                )
                return {
                    "success": False,
                    "error": "INTERNAL_ERROR",
                    "message": (
                        f"Forced task '{tid}' was not scheduled despite "
                        f"a feasible solver result. This is unexpected."
                    ),
                    "forced_task_ids": forced_task_ids,
                }

    # ------------------------------------------------------------------
    # 12. Validate (unchanged validator)
    # ------------------------------------------------------------------
    logger.info("API: Validating schedule …")
    validation = validate_schedule(
        ctx, result, occ_lookup, schedule_df, blocks_df
    )

    # ------------------------------------------------------------------
    # 13. Build summary
    # ------------------------------------------------------------------
    summary = build_summary(ctx, result, schedule_df, blocks_df, unscheduled_df)

    total_time = time.perf_counter() - total_start

    logger.info(
        "API: Optimization complete in %.2fs — %d scheduled, %d unscheduled, "
        "%d blocks, validation=%s",
        total_time,
        len(schedule_df),
        len(unscheduled_df),
        len(blocks_df),
        "PASSED" if validation["valid"] else "FAILED",
    )

    return {
        "success": True,
        "error": None,
        "message": "Optimization completed successfully.",
        "forced_task_ids": forced_task_ids,
        "schedule_df": schedule_df,
        "blocks_df": blocks_df,
        "unscheduled_df": unscheduled_df,
        "summary": summary,
        "validation": validation,
        "solver_result": result,
        "wall_time_seconds": round(total_time, 3),
    }
