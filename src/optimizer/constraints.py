"""
Constraint definitions for the CP-SAT maintenance block optimizer.

Each public function adds a family of constraints to the model.
All functions receive a ``ModelContext`` and modify its ``model`` in place.
"""

import logging
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from .model import ModelContext, TaskVar
from .preprocessing import OccupancyLookup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Time-window & deadline constraints
# ---------------------------------------------------------------------------

def add_time_window_constraints(ctx: ModelContext) -> None:
    """
    Enforce that each scheduled task stays within its
    ``[earliest_start, latest_end]`` window and respects its deadline.

    These are already partially encoded in the variable domains, but we
    add explicit constraints to interact correctly with the ``scheduled``
    flag.
    """
    model = ctx.model
    count = 0

    for tv in ctx.task_vars.values():
        # When scheduled, start >= earliest_start (domain already enforces)
        # When scheduled, end <= min(latest_end, deadline)
        effective_end = min(tv.latest_end_min, tv.deadline_min)
        model.add(tv.end <= effective_end).only_enforce_if(tv.scheduled)
        model.add(tv.start >= tv.earliest_start_min).only_enforce_if(tv.scheduled)
        count += 1

    logger.info("Added %d time-window / deadline constraints", count)


# ---------------------------------------------------------------------------
# 2. Train-occupancy constraints
# ---------------------------------------------------------------------------

def add_train_occupancy_constraints(
    ctx: ModelContext,
    occupancy: OccupancyLookup,
) -> None:
    """
    Prevent scheduled maintenance from overlapping train occupancy windows
    on the same section, when the task requires an infrastructure block
    (track, signal, or power).

    For each occupied interval ``[occ_s, occ_e)`` that overlaps the task's
    feasible window, we add:

        scheduled → (end <= occ_s) OR (start >= occ_e)

    This is linearised using an auxiliary BoolVar.
    """
    model = ctx.model
    n_constraints = 0

    for tv in ctx.task_vars.values():
        # Only constrain tasks that actually need an infrastructure block
        needs_block = (
            tv.track_block_required
            or tv.signal_block_required
            or tv.power_block_required
        )
        if not needs_block:
            continue

        occ_windows = occupancy.get(tv.section_id, [])
        if not occ_windows:
            continue

        for occ_s, occ_e in occ_windows:
            # Skip windows entirely outside the task's feasible range
            if occ_e <= tv.earliest_start_min or occ_s >= tv.latest_end_min:
                continue

            # Disjunction: task ends before occupancy OR starts after occupancy
            b = model.new_bool_var(
                f"occ_{tv.task_id}_{occ_s}_{occ_e}"
            )
            # b = 1 → task finishes before occupancy starts
            model.add(tv.end <= occ_s).only_enforce_if(tv.scheduled, b)
            # b = 0 → task starts after occupancy ends
            model.add(tv.start >= occ_e).only_enforce_if(tv.scheduled, b.negated())
            n_constraints += 1

    logger.info(
        "Added %d train-occupancy disjunction constraints", n_constraints
    )


# ---------------------------------------------------------------------------
# 3. Resource no-overlap constraints
# ---------------------------------------------------------------------------

def add_resource_constraints(ctx: ModelContext) -> None:
    """
    For each ``(section, resource_type)`` group, add a ``NoOverlap``
    constraint so that tasks requiring the same exclusive resource on
    the same section do not overlap.

    Resource types: ``track``, ``signal``, ``power``.
    """
    model = ctx.model
    n_groups = 0

    for resource_key, task_ids in ctx.tasks_by_resource.items():
        if len(task_ids) < 2:
            continue  # no conflict possible with a single task

        intervals = [ctx.task_vars[tid].interval for tid in task_ids]
        model.add_no_overlap(intervals)
        n_groups += 1

    logger.info(
        "Added NoOverlap constraints for %d resource groups", n_groups
    )


# ---------------------------------------------------------------------------
# 4. Section-level capacity (all block types combined)
# ---------------------------------------------------------------------------

def add_section_capacity_constraints(ctx: ModelContext) -> None:
    """
    Optional: limit how many simultaneous maintenance blocks can exist
    on a single section.  Currently unused (the resource-level NoOverlap
    constraints provide sufficient protection) but the hook exists for
    future section-wide capacity limits.
    """
    # Placeholder — can be activated with a section_max_concurrent config
    pass


# ---------------------------------------------------------------------------
# 5. Dependency constraints (framework)
# ---------------------------------------------------------------------------

def add_dependency_constraints(
    ctx: ModelContext,
    dependencies: List[Tuple[str, str]] | None = None,
) -> None:
    """
    Add precedence constraints of the form ``end(pred) <= start(succ)``.

    Parameters
    ----------
    dependencies : list of (predecessor_task_id, successor_task_id)
        If ``None`` or empty, no constraints are added.
        The existing dataset does not contain dependencies, but this
        framework supports them for future use.
    """
    if not dependencies:
        logger.info("No task dependencies provided — skipping.")
        return

    model = ctx.model
    count = 0

    for pred_id, succ_id in dependencies:
        pred = ctx.task_vars.get(pred_id)
        succ = ctx.task_vars.get(succ_id)
        if pred is None or succ is None:
            logger.warning(
                "Dependency %s → %s references unknown task — skipped",
                pred_id, succ_id,
            )
            continue

        # Both must be scheduled for the dependency to apply
        model.add(pred.end <= succ.start).only_enforce_if(
            pred.scheduled, succ.scheduled
        )
        count += 1

    logger.info("Added %d dependency (precedence) constraints", count)


# ---------------------------------------------------------------------------
# 6. Unscheduled Limit Constraints
# ---------------------------------------------------------------------------

def add_unscheduled_limit_constraint(
    ctx: ModelContext,
    min_unscheduled: int = 600,
) -> None:
    """
    Enforce that at least min_unscheduled tasks remain unscheduled.
    """
    model = ctx.model
    total_tasks = len(ctx.task_vars)
    max_scheduled = max(0, total_tasks - min_unscheduled)

    scheduled_vars = [tv.scheduled for tv in ctx.task_vars.values()]
    model.add(sum(scheduled_vars) <= max_scheduled)

    logger.info("Added constraint: maximum scheduled tasks <= %d (min unscheduled >= %d)", max_scheduled, min_unscheduled)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_all_constraints(
    ctx: ModelContext,
    occupancy: OccupancyLookup,
    dependencies: List[Tuple[str, str]] | None = None,
) -> None:
    """Add every constraint family to the model."""
    add_time_window_constraints(ctx)
    add_train_occupancy_constraints(ctx, occupancy)
    add_resource_constraints(ctx)
    add_section_capacity_constraints(ctx)
    add_dependency_constraints(ctx, dependencies)
    add_unscheduled_limit_constraint(ctx)
