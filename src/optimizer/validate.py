"""
Independent post-solution validator for the maintenance block optimizer.

This module does NOT trust the CP-SAT feasibility status alone.
It independently checks the generated schedule against all constraints.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import config as cfg
from .model import ModelContext, TaskVar
from .preprocessing import OccupancyLookup, minutes_to_datetime
from .solver import SolverResult

logger = logging.getLogger(__name__)


def validate_schedule(
    ctx: ModelContext,
    result: SolverResult,
    occupancy: OccupancyLookup,
    schedule_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
) -> dict:
    """
    Independently validate the generated schedule.

    Returns a dict with:
        valid: bool
        errors: list[str]
        warnings: list[str]
        checked_tasks: int
        checked_blocks: int
        train_conflicts_found: int
        resource_conflicts_found: int
        deadline_violations: int
        duration_violations: int
    """
    errors: List[str] = []
    warnings: List[str] = []
    train_conflicts = 0
    resource_conflicts = 0
    deadline_violations = 0
    duration_violations = 0

    # Build lookup of scheduled task intervals
    scheduled_tasks: Dict[str, Dict] = {}
    for tid, sol in result.solution.items():
        if sol["scheduled"]:
            scheduled_tasks[tid] = sol

    # ==================================================================
    # 1. Task coverage: every input task is either SCHEDULED or UNSCHEDULED
    # ==================================================================
    all_input_ids = set(ctx.task_vars.keys())
    all_solution_ids = set(result.solution.keys())
    missing = all_input_ids - all_solution_ids
    if missing:
        errors.append(
            f"{len(missing)} tasks silently dropped from solution: "
            f"{list(missing)[:5]}..."
        )

    # ==================================================================
    # 2. No duplicate scheduling
    # ==================================================================
    # (Each task should appear at most once in the schedule)
    if not schedule_df.empty:
        dup_tasks = schedule_df[schedule_df.duplicated(subset="task_id", keep=False)]
        if not dup_tasks.empty:
            errors.append(
                f"{len(dup_tasks)} duplicate task entries in schedule: "
                f"{dup_tasks['task_id'].unique().tolist()[:5]}"
            )

    # ==================================================================
    # 3. Time-window constraints
    # ==================================================================
    for tid, sol in scheduled_tasks.items():
        tv = ctx.task_vars[tid]
        start = sol["start_min"]
        end = sol["end_min"]

        if start < tv.earliest_start_min:
            errors.append(
                f"{tid}: start ({start}) < earliest_start ({tv.earliest_start_min})"
            )

        if end > tv.latest_end_min:
            errors.append(
                f"{tid}: end ({end}) > latest_end ({tv.latest_end_min})"
            )

        if start > end:
            errors.append(f"{tid}: start ({start}) > end ({end})")

    # ==================================================================
    # 4. Duration constraints
    # ==================================================================
    for tid, sol in scheduled_tasks.items():
        tv = ctx.task_vars[tid]
        actual_dur = sol["end_min"] - sol["start_min"]
        if actual_dur != tv.duration_min:
            duration_violations += 1
            errors.append(
                f"{tid}: duration mismatch — "
                f"scheduled={actual_dur}min vs expected={tv.duration_min}min"
            )

    # ==================================================================
    # 5. Deadline constraints
    # ==================================================================
    for tid, sol in scheduled_tasks.items():
        tv = ctx.task_vars[tid]
        if sol["end_min"] > tv.deadline_min:
            deadline_violations += 1
            errors.append(
                f"{tid}: end ({sol['end_min']}) exceeds deadline ({tv.deadline_min})"
            )

    # ==================================================================
    # 6. Train occupancy conflicts
    # ==================================================================
    for tid, sol in scheduled_tasks.items():
        tv = ctx.task_vars[tid]
        needs_block = (
            tv.track_block_required
            or tv.signal_block_required
            or tv.power_block_required
        )
        if not needs_block:
            continue

        occ_windows = occupancy.get(tv.section_id, [])
        start = sol["start_min"]
        end = sol["end_min"]

        for occ_s, occ_e in occ_windows:
            # Check overlap: task [start, end) ∩ occupancy [occ_s, occ_e)
            if start < occ_e and occ_s < end:
                train_conflicts += 1
                start_dt = minutes_to_datetime(start).strftime("%Y-%m-%d %H:%M")
                end_dt = minutes_to_datetime(end).strftime("%Y-%m-%d %H:%M")
                occ_s_dt = minutes_to_datetime(occ_s).strftime("%Y-%m-%d %H:%M")
                occ_e_dt = minutes_to_datetime(occ_e).strftime("%Y-%m-%d %H:%M")
                errors.append(
                    f"{tid}: overlaps train occupancy on {tv.section_id} "
                    f"(task: {start_dt}–{end_dt}, train: {occ_s_dt}–{occ_e_dt})"
                )

    # ==================================================================
    # 7. Resource conflicts (same section, same resource type)
    # ==================================================================
    for resource_key, task_ids in ctx.tasks_by_resource.items():
        # Get scheduled tasks in this resource group
        sched_in_group = [
            (tid, scheduled_tasks[tid])
            for tid in task_ids
            if tid in scheduled_tasks
        ]

        if len(sched_in_group) < 2:
            continue

        # Check pairwise for overlaps
        for i in range(len(sched_in_group)):
            for j in range(i + 1, len(sched_in_group)):
                tid_a, sol_a = sched_in_group[i]
                tid_b, sol_b = sched_in_group[j]

                # Overlap check
                if (sol_a["start_min"] < sol_b["end_min"] and
                        sol_b["start_min"] < sol_a["end_min"]):
                    resource_conflicts += 1
                    errors.append(
                        f"Resource conflict ({resource_key}): "
                        f"{tid_a} [{sol_a['start_min']}–{sol_a['end_min']}] "
                        f"overlaps {tid_b} [{sol_b['start_min']}–{sol_b['end_min']}]"
                    )

    # ==================================================================
    # 8. Block consistency
    # ==================================================================
    if not blocks_df.empty:
        for _, block_row in blocks_df.iterrows():
            bid = block_row["block_id"]
            block_tids = block_row["task_ids"].split(";")
            section = block_row["section_id"]

            for tid in block_tids:
                if tid not in ctx.task_vars:
                    errors.append(
                        f"Block {bid} references unknown task {tid}"
                    )
                    continue

                tv = ctx.task_vars[tid]
                if tv.section_id != section:
                    errors.append(
                        f"Block {bid}: task {tid} is on section {tv.section_id} "
                        f"but block is on {section}"
                    )

                if tid not in scheduled_tasks:
                    errors.append(
                        f"Block {bid}: task {tid} is listed but not scheduled"
                    )

    # ==================================================================
    # Summary
    # ==================================================================
    valid = len(errors) == 0

    report = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checked_tasks": len(all_input_ids),
        "checked_blocks": len(blocks_df) if not blocks_df.empty else 0,
        "train_conflicts_found": train_conflicts,
        "resource_conflicts_found": resource_conflicts,
        "deadline_violations": deadline_violations,
        "duration_violations": duration_violations,
    }

    if valid:
        logger.info(
            "Validation PASSED: %d tasks, %d blocks checked — no violations.",
            report["checked_tasks"],
            report["checked_blocks"],
        )
    else:
        logger.error(
            "Validation FAILED with %d errors. See report for details.",
            len(errors),
        )

    return report


def save_validation_report(
    report: dict,
    output_dir: str | None = None,
) -> str:
    """Save the validation report to JSON and return the file path."""
    out = output_dir or cfg.ARTIFACTS_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "validation_report.json")

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved to %s", path)
    return path
