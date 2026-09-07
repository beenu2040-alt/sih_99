"""
CP-SAT model builder for the maintenance block optimizer.

Creates decision variables (start, end, scheduled, interval) for every
maintenance task and returns a structured model context that downstream
modules use to add constraints and objectives.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from ortools.sat.python import cp_model

from . import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class TaskVar:
    """Decision variables for a single maintenance task."""

    task_id: str
    section_id: str
    department: str

    # Integer-minute bounds
    earliest_start_min: int
    latest_end_min: int
    deadline_min: int
    duration_min: int

    # Flags
    power_block_required: bool
    signal_block_required: bool
    track_block_required: bool
    can_combine: bool

    # Objective coefficients (integers)
    priority_int: int
    urgency_int: int
    conflict_cost_int: int

    # Raw values for reporting
    predicted_priority_score: float
    predicted_priority_class: str
    duration_hours: float
    train_conflict_cost: float

    # CP-SAT variables (assigned during model building)
    scheduled: cp_model.IntVar = field(default=None, repr=False)
    start: cp_model.IntVar = field(default=None, repr=False)
    end: cp_model.IntVar = field(default=None, repr=False)
    interval: cp_model.IntervalVar = field(default=None, repr=False)

    @property
    def is_feasible(self) -> bool:
        """Whether the task's time window can accommodate its duration."""
        return (self.latest_end_min - self.earliest_start_min) >= self.duration_min


@dataclass
class ModelContext:
    """
    Container for the CP-SAT model and all task variables.

    Passed between model.py → constraints.py → objective.py → solver.py.
    """

    model: cp_model.CpModel
    task_vars: Dict[str, TaskVar]

    # Group indices built during model construction for efficient
    # constraint application.
    tasks_by_section: Dict[str, List[str]] = field(default_factory=dict)
    tasks_by_resource: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def all_task_ids(self) -> List[str]:
        return list(self.task_vars.keys())

    def get_section_tasks(self, section_id: str) -> List[TaskVar]:
        """Return TaskVars for all tasks in *section_id*."""
        return [
            self.task_vars[tid]
            for tid in self.tasks_by_section.get(section_id, [])
        ]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(tasks_df: pd.DataFrame) -> ModelContext:
    """
    Create the CP-SAT model with decision variables for every task.

    Parameters
    ----------
    tasks_df : DataFrame
        Preprocessed task data (must include integer time columns and
        scaled coefficients).

    Returns
    -------
    ModelContext
        Contains the ``CpModel`` and all ``TaskVar`` objects.
    """
    model = cp_model.CpModel()
    task_vars: Dict[str, TaskVar] = {}
    tasks_by_section: Dict[str, List[str]] = {}
    tasks_by_resource: Dict[str, List[str]] = {}

    for _, row in tasks_df.iterrows():
        tid = str(row["task_id"])
        sid = str(row["section_id"])
        dept = str(row["department"])

        es = int(row["earliest_start_min"])
        le = int(row["latest_end_min"])
        dl = int(row["deadline_min"])
        dur = int(row["duration_min"])

        pwr = bool(row["power_block_required"])
        sig = bool(row["signal_block_required"])
        trk = bool(row["track_block_required"])
        comb = bool(row["can_combine"])

        pri = int(row["priority_int"])
        urg = int(row["urgency_int"])
        cfl = int(row["conflict_cost_int"])

        # ---- Decision variables ----

        # Whether this task is scheduled (1) or not (0)
        scheduled = model.new_bool_var(f"scheduled_{tid}")

        # Effective latest end is min(latest_end, deadline)
        effective_le = min(le, dl)

        # If the window is too tight, the task is optional but unlikely
        feasible = (effective_le - es) >= dur
        if not feasible:
            # Force unscheduled for infeasible tasks
            model.add(scheduled == 0)
            effective_le = max(effective_le, es + dur)  # avoid domain error

        # Start time
        start = model.new_int_var(es, effective_le - dur, f"start_{tid}")

        # End time
        end = model.new_int_var(es + dur, effective_le, f"end_{tid}")

        # Optional interval — only present when scheduled=1
        interval = model.new_optional_fixed_size_interval_var(
            start, dur, scheduled, f"interval_{tid}"
        )

        # Link end = start + dur
        model.add(end == start + dur)

        # ---- Build TaskVar ----

        tv = TaskVar(
            task_id=tid,
            section_id=sid,
            department=dept,
            earliest_start_min=es,
            latest_end_min=le,
            deadline_min=dl,
            duration_min=dur,
            power_block_required=pwr,
            signal_block_required=sig,
            track_block_required=trk,
            can_combine=comb,
            priority_int=pri,
            urgency_int=urg,
            conflict_cost_int=cfl,
            predicted_priority_score=float(row["predicted_priority_score"]),
            predicted_priority_class=str(row["predicted_priority_class"]),
            duration_hours=float(row["duration_hours"]),
            train_conflict_cost=float(row["train_conflict_cost"]),
            scheduled=scheduled,
            start=start,
            end=end,
            interval=interval,
        )

        task_vars[tid] = tv

        # ---- Group indices ----
        tasks_by_section.setdefault(sid, []).append(tid)

        # Resource groups: (section, resource_type) → task list
        for resource_flag, resource_name in [
            (pwr, "power"),
            (sig, "signal"),
            (trk, "track"),
        ]:
            if resource_flag:
                key = f"{sid}__{resource_name}"
                tasks_by_resource.setdefault(key, []).append(tid)

    ctx = ModelContext(
        model=model,
        task_vars=task_vars,
        tasks_by_section=tasks_by_section,
        tasks_by_resource=tasks_by_resource,
    )

    logger.info(
        "Built CP-SAT model: %d tasks, %d sections, %d resource groups",
        len(task_vars),
        len(tasks_by_section),
        len(tasks_by_resource),
    )

    return ctx
