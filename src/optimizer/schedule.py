"""
Schedule extraction and output generation.

Converts solver results into human-readable CSV and JSON artefacts:
  - optimal_schedule.csv
  - maintenance_blocks.csv
  - unscheduled_tasks.csv
  - optimization_summary.json
  - solver_log.txt
"""

import json
import logging
import os
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

import pandas as pd

from . import config as cfg
from .model import ModelContext, TaskVar
from .preprocessing import minutes_to_datetime
from .solver import SolverResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _priority_class(score: float) -> str:
    """Map a numeric priority score to a class label."""
    for cls, threshold in sorted(
        cfg.PRIORITY_CLASS_THRESHOLDS.items(),
        key=lambda x: -x[1],
    ):
        if score >= threshold:
            return cls
    return "LOW"


def _reason_unscheduled(tv: TaskVar) -> str:
    """Heuristic explanation for why a task was not scheduled."""
    if not tv.is_feasible:
        return "Time window too narrow for task duration"
    return "Excluded by solver due to resource or train-occupancy conflicts"


# ---------------------------------------------------------------------------
# Schedule DataFrame
# ---------------------------------------------------------------------------

def build_schedule_df(
    ctx: ModelContext,
    result: SolverResult,
) -> pd.DataFrame:
    """
    Build the ``optimal_schedule.csv`` DataFrame from solver results.
    """
    rows = []
    for tid, sol in result.solution.items():
        tv = ctx.task_vars[tid]
        if not sol["scheduled"]:
            continue

        start_dt = minutes_to_datetime(sol["start_min"])
        end_dt = minutes_to_datetime(sol["end_min"])

        rows.append({
            "task_id": tid,
            "section_id": tv.section_id,
            "department": tv.department,
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
            "duration_hours": tv.duration_hours,
            "predicted_priority_score": round(tv.predicted_priority_score, 2),
            "predicted_priority_class": tv.predicted_priority_class,
            "train_conflict_cost": round(tv.train_conflict_cost, 1),
            "power_block_required": tv.power_block_required,
            "signal_block_required": tv.signal_block_required,
            "track_block_required": tv.track_block_required,
            "status": "SCHEDULED",
            # Internal minute values for block grouping
            "_start_min": sol["start_min"],
            "_end_min": sol["end_min"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No tasks were scheduled.")
    return df


# ---------------------------------------------------------------------------
# Block grouping
# ---------------------------------------------------------------------------

def _intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Check whether two closed intervals overlap or are adjacent."""
    return a_start < b_end and b_start < a_end


def group_into_blocks(
    schedule_df: pd.DataFrame,
    ctx: ModelContext,
) -> pd.DataFrame:
    """
    Group scheduled tasks into maintenance blocks.

    Two tasks are placed in the same block when:
      1. Same section_id
      2. At least one has ``can_combine = True``
      3. Their scheduled intervals overlap or are adjacent
      4. Compatible resource requirements (no conflicting exclusive resources
         that would violate NoOverlap — but since the solver already
         enforces NoOverlap, tasks sharing a resource group are sequential
         within the block window)

    The block window spans from the earliest start to the latest end
    of its constituent tasks.
    """
    if schedule_df.empty:
        return pd.DataFrame(columns=[
            "block_id", "section_id", "start_time", "end_time",
            "duration_hours", "task_count", "task_ids", "departments",
            "resource_requirements",
        ])

    blocks: List[Dict] = []
    block_counter = 0

    for section_id, section_tasks in schedule_df.groupby("section_id"):
        # Sort by start time
        section_tasks = section_tasks.sort_values("_start_min").reset_index(drop=True)

        # Greedy merge: walk through tasks and merge overlapping/adjacent ones
        assigned = [False] * len(section_tasks)

        for i in range(len(section_tasks)):
            if assigned[i]:
                continue

            row_i = section_tasks.iloc[i]
            tv_i = ctx.task_vars[row_i["task_id"]]

            block_task_ids = [row_i["task_id"]]
            block_start = row_i["_start_min"]
            block_end = row_i["_end_min"]
            block_depts = {row_i["department"]}
            block_resources = set()
            if tv_i.track_block_required:
                block_resources.add("track")
            if tv_i.signal_block_required:
                block_resources.add("signal")
            if tv_i.power_block_required:
                block_resources.add("power")

            assigned[i] = True

            # Try to merge subsequent tasks
            for j in range(i + 1, len(section_tasks)):
                if assigned[j]:
                    continue

                row_j = section_tasks.iloc[j]
                tv_j = ctx.task_vars[row_j["task_id"]]

                # Both must allow combination (or at least one)
                if not (tv_i.can_combine or tv_j.can_combine):
                    continue

                # Must overlap or be adjacent with current block window
                if not _intervals_overlap(
                    block_start, block_end,
                    row_j["_start_min"], row_j["_end_min"],
                ):
                    # Also allow adjacency (gap ≤ granularity)
                    gap = row_j["_start_min"] - block_end
                    if gap > cfg.TIME_GRANULARITY_MINUTES:
                        continue

                # Merge
                block_task_ids.append(row_j["task_id"])
                block_start = min(block_start, row_j["_start_min"])
                block_end = max(block_end, row_j["_end_min"])
                block_depts.add(row_j["department"])
                if tv_j.track_block_required:
                    block_resources.add("track")
                if tv_j.signal_block_required:
                    block_resources.add("signal")
                if tv_j.power_block_required:
                    block_resources.add("power")
                assigned[j] = True

            block_counter += 1
            block_id = f"BLK{block_counter:05d}"
            start_dt = minutes_to_datetime(block_start)
            end_dt = minutes_to_datetime(block_end)
            dur_hours = round((block_end - block_start) / 60.0, 2)

            blocks.append({
                "block_id": block_id,
                "section_id": section_id,
                "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
                "duration_hours": dur_hours,
                "task_count": len(block_task_ids),
                "task_ids": ";".join(block_task_ids),
                "departments": ";".join(sorted(block_depts)),
                "resource_requirements": ";".join(sorted(block_resources)) if block_resources else "none",
            })

    blocks_df = pd.DataFrame(blocks)
    logger.info(
        "Grouped %d scheduled tasks into %d maintenance blocks",
        len(schedule_df), len(blocks_df),
    )
    return blocks_df


def assign_block_ids(
    schedule_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``block_id`` column to the schedule DataFrame."""
    if schedule_df.empty or blocks_df.empty:
        schedule_df["block_id"] = None
        return schedule_df

    # Build task→block mapping
    task_to_block = {}
    for _, block_row in blocks_df.iterrows():
        bid = block_row["block_id"]
        for tid in block_row["task_ids"].split(";"):
            task_to_block[tid] = bid

    schedule_df = schedule_df.copy()
    schedule_df["block_id"] = schedule_df["task_id"].map(task_to_block)
    return schedule_df


# ---------------------------------------------------------------------------
# Unscheduled tasks
# ---------------------------------------------------------------------------

def build_unscheduled_df(
    ctx: ModelContext,
    result: SolverResult,
) -> pd.DataFrame:
    """Build the ``unscheduled_tasks.csv`` DataFrame."""
    rows = []
    for tid, sol in result.solution.items():
        if sol["scheduled"]:
            continue

        tv = ctx.task_vars[tid]
        rows.append({
            "task_id": tid,
            "section_id": tv.section_id,
            "department": tv.department,
            "predicted_priority_score": round(tv.predicted_priority_score, 2),
            "predicted_priority_class": tv.predicted_priority_class,
            "duration_hours": tv.duration_hours,
            "reason": _reason_unscheduled(tv),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(
    ctx: ModelContext,
    result: SolverResult,
    schedule_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    unscheduled_df: pd.DataFrame,
) -> dict:
    """Build the ``optimization_summary.json`` data."""
    total = len(ctx.task_vars)
    n_scheduled = len(schedule_df)
    n_unscheduled = len(unscheduled_df)
    n_blocks = len(blocks_df)

    avg_priority = (
        schedule_df["predicted_priority_score"].mean()
        if not schedule_df.empty else 0.0
    )

    high_scheduled = (
        len(schedule_df[schedule_df["predicted_priority_class"].isin(["HIGH", "CRITICAL"])])
        if not schedule_df.empty else 0
    )
    critical_scheduled = (
        len(schedule_df[schedule_df["predicted_priority_class"] == "CRITICAL"])
        if not schedule_df.empty else 0
    )
    total_conflict_cost = (
        schedule_df["train_conflict_cost"].sum()
        if not schedule_df.empty else 0.0
    )

    consolidation_ratio = round(n_scheduled / n_blocks, 2) if n_blocks > 0 else 0.0

    cross_dept_blocks = 0
    cross_dept_tasks = 0

    if not blocks_df.empty:
        for _, row in blocks_df.iterrows():
            depts = [d for d in row["departments"].split(";") if d.strip()]
            if len(depts) >= 2:
                cross_dept_blocks += 1
                cross_dept_tasks += row["task_count"]

    return {
        "solver_status": result.status,
        "objective_value": result.objective_value,
        "best_bound": result.best_bound,
        "solver_time_seconds": result.wall_time_seconds,
        "total_tasks": total,
        "scheduled_tasks": n_scheduled,
        "unscheduled_tasks": n_unscheduled,
        "total_blocks": n_blocks,
        "average_priority": round(avg_priority, 2),
        "high_priority_scheduled": high_scheduled,
        "critical_priority_scheduled": critical_scheduled,
        "total_train_conflict_cost": round(total_conflict_cost, 2),
        "block_consolidation_ratio": consolidation_ratio,
        "cross_department_coordinated_blocks": cross_dept_blocks,
        "cross_department_task_pairs_groups": cross_dept_blocks,
        "tasks_participating_in_cross_department_coordination": cross_dept_tasks,
    }


# ---------------------------------------------------------------------------
# Solver log
# ---------------------------------------------------------------------------

def build_solver_log(
    result: SolverResult,
    summary: dict,
    validation_result: dict | None = None,
) -> str:
    """Build the ``solver_log.txt`` content."""
    lines = [
        "=" * 60,
        "OR-Tools CP-SAT Maintenance Block Optimizer — Solver Log",
        "=" * 60,
        "",
        f"Solver status       : {result.status}",
        f"Objective value     : {result.objective_value}",
        f"Best bound          : {result.best_bound}",
        f"Solve time (s)      : {result.wall_time_seconds}",
        f"Conflicts           : {result.num_conflicts}",
        f"Branches            : {result.num_branches}",
        "",
        f"Total tasks         : {summary['total_tasks']}",
        f"Scheduled tasks     : {summary['scheduled_tasks']}",
        f"Unscheduled tasks   : {summary['unscheduled_tasks']}",
        f"Total blocks        : {summary['total_blocks']}",
        f"Consolidation ratio : {summary['block_consolidation_ratio']}",
        f"Average priority    : {summary['average_priority']}",
        f"Total conflict cost : {summary['total_train_conflict_cost']}",
        "",
    ]

    if validation_result is not None:
        lines.append("Validation result   : " + (
            "PASSED" if validation_result.get("valid") else "FAILED"
        ))
        if not validation_result.get("valid"):
            for err in validation_result.get("errors", []):
                lines.append(f"  ERROR: {err}")
        for w in validation_result.get("warnings", []):
            lines.append(f"  WARNING: {w}")
    else:
        lines.append("Validation result   : not yet run")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_outputs(
    schedule_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    unscheduled_df: pd.DataFrame,
    summary: dict,
    solver_log: str,
    output_dir: str | None = None,
) -> None:
    """Write all output files to the artefacts directory."""
    out = output_dir or cfg.ARTIFACTS_DIR
    os.makedirs(out, exist_ok=True)

    # Drop internal columns before saving
    save_sched = schedule_df.drop(
        columns=[c for c in ("_start_min", "_end_min") if c in schedule_df.columns],
        errors="ignore",
    )

    # Reorder columns so block_id is first if present
    if "block_id" in save_sched.columns:
        cols = ["block_id"] + [c for c in save_sched.columns if c != "block_id"]
        save_sched = save_sched[cols]

    save_sched.to_csv(os.path.join(out, "optimal_schedule.csv"), index=False)
    blocks_df.to_csv(os.path.join(out, "maintenance_blocks.csv"), index=False)
    unscheduled_df.to_csv(os.path.join(out, "unscheduled_tasks.csv"), index=False)

    with open(os.path.join(out, "optimization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out, "solver_log.txt"), "w") as f:
        f.write(solver_log)

    logger.info("All outputs saved to %s", out)
