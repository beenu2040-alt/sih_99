"""
Objective function for the CP-SAT maintenance block optimizer.

All coefficients are integers.  Floating-point policy weights and
priority scores are scaled before being passed to CP-SAT.
"""

import logging
from typing import Dict, List

from ortools.sat.python import cp_model

from . import config as cfg
from .model import ModelContext, TaskVar

logger = logging.getLogger(__name__)


def _build_combination_bonus(ctx: ModelContext) -> List:
    """
    For each pair of compatible tasks on the same section whose time
    windows overlap, create a bonus variable that is 1 when both tasks
    are scheduled and their intervals overlap (i.e. they can share a
    maintenance block window).

    Returns a list of ``(bonus_var, weight)`` tuples.
    """
    model = ctx.model
    bonuses = []

    for section_id, task_ids in ctx.tasks_by_section.items():
        combinable = [
            ctx.task_vars[tid]
            for tid in task_ids
            if ctx.task_vars[tid].can_combine
        ]
        if len(combinable) < 2:
            continue

        # Only consider pairs that *could* overlap (time windows intersect)
        for i in range(len(combinable)):
            for j in range(i + 1, len(combinable)):
                a = combinable[i]
                b = combinable[j]

                # Quick feasibility check: windows must intersect
                if (a.latest_end_min <= b.earliest_start_min or
                        b.latest_end_min <= a.earliest_start_min):
                    continue

                # bonus = 1  iff  both scheduled AND intervals overlap
                # Overlap ↔ a.start < b.end AND b.start < a.end
                both_scheduled = model.new_bool_var(
                    f"both_{a.task_id}_{b.task_id}"
                )
                model.add_min_equality(
                    both_scheduled, [a.scheduled, b.scheduled]
                )

                overlap = model.new_bool_var(
                    f"overlap_{a.task_id}_{b.task_id}"
                )
                # overlap can be 1 only if both scheduled
                model.add(overlap <= both_scheduled)

                # If overlap=1 → a.start < b.end AND b.start < a.end
                model.add(a.start < b.end).only_enforce_if(overlap)
                model.add(b.start < a.end).only_enforce_if(overlap)

                # Average priority bonus
                avg_pri = (a.priority_int + b.priority_int) // 2
                weight = avg_pri * cfg.CONSOLIDATION_WEIGHT // 100
                
                # Cross-department coordination multiplier
                if a.department != b.department:
                    weight *= 2

                weight = max(weight, 1)

                bonuses.append((overlap, weight))

    return bonuses


def add_objective(ctx: ModelContext) -> None:
    """
    Build and set the composite optimisation objective.

    The objective is expressed as a single **maximisation** target.
    Minimisation components are negated.

    Components
    ----------
    1. **Priority** — reward scheduling high-priority tasks.
    2. **Urgency** — reward scheduling tasks with tight deadlines.
    3. **Train conflict penalty** — penalise scheduling in high-traffic slots.
    4. **Block consolidation bonus** — reward overlapping compatible tasks.
    5. **Unscheduled penalty** — penalise leaving important tasks unscheduled.
    """
    model = ctx.model
    objective_terms = []

    # ---- 1. Priority reward ----
    for tv in ctx.task_vars.values():
        # priority_int ∈ [0, 1000],  PRIORITY_WEIGHT default 100
        coeff = tv.priority_int * cfg.PRIORITY_WEIGHT // 100
        if coeff > 0:
            objective_terms.append(coeff * tv.scheduled)

    # ---- 2. Urgency reward ----
    for tv in ctx.task_vars.values():
        coeff = tv.urgency_int * cfg.DEADLINE_WEIGHT // 100
        if coeff > 0:
            objective_terms.append(coeff * tv.scheduled)

    # ---- 3. Train conflict penalty (negated) ----
    for tv in ctx.task_vars.values():
        coeff = tv.conflict_cost_int * cfg.TRAIN_CONFLICT_WEIGHT // 100
        if coeff > 0:
            objective_terms.append(-coeff * tv.scheduled)

    # ---- 4. Block consolidation bonus ----
    bonuses = _build_combination_bonus(ctx)
    for bonus_var, weight in bonuses:
        objective_terms.append(weight * bonus_var)

    logger.info("Added %d block-combination bonus variables", len(bonuses))

    # ---- 5. Unscheduled penalty (negated) ----
    for tv in ctx.task_vars.values():
        coeff = tv.priority_int * cfg.UNSCHEDULED_PENALTY // 100
        if coeff > 0:
            # Penalty for NOT scheduling: -coeff * (1 - scheduled)
            #   = -coeff + coeff * scheduled
            # The constant -coeff doesn't affect the optimum, so we add
            # only the variable part and note the constant offset.
            objective_terms.append(coeff * tv.scheduled)

    # ---- Set objective ----
    if objective_terms:
        model.maximize(sum(objective_terms))
        logger.info(
            "Objective set: %d linear terms + %d bonus terms",
            len(ctx.task_vars) * 4,  # rough count
            len(bonuses),
        )
    else:
        logger.warning("No objective terms — the model has no tasks?")
