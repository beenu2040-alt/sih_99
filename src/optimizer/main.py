"""
CLI entry point for the OR-Tools CP-SAT Maintenance Block Optimizer.

Usage::

    python -m src.optimizer.main [--time-limit 60] [--workers 8] [--seed 42]
"""

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimizer")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="OR-Tools CP-SAT Maintenance Block Optimizer for SIH PS 26027",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Solver time limit in seconds (default: from config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel solver workers (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for solver reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artefacts (default: artifacts/optimization)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete optimisation pipeline."""
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 64)
    print("  OR-Tools CP-SAT Maintenance Block Optimizer")
    print("  SIH PS 26027 — AI-Powered Automatic Block Planning")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    from .data_loader import load_all

    logger.info("Step 1/10: Loading data …")
    tasks_df, occupancy_df = load_all()
    print(f"  Loaded {len(tasks_df)} tasks, {len(occupancy_df)} occupancy records\n")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    from .preprocessing import preprocess

    logger.info("Step 2/10: Preprocessing …")
    tasks_df, occ_lookup = preprocess(tasks_df, occupancy_df)
    print(f"  After validation: {len(tasks_df)} tasks ready\n")

    # ------------------------------------------------------------------
    # 3. Build CP-SAT model
    # ------------------------------------------------------------------
    from .model import build_model

    logger.info("Step 3/10: Building CP-SAT model …")
    ctx = build_model(tasks_df)
    print(f"  Variables: {len(ctx.task_vars)} tasks, "
          f"{len(ctx.tasks_by_section)} sections, "
          f"{len(ctx.tasks_by_resource)} resource groups\n")

    # ------------------------------------------------------------------
    # 4. Add constraints
    # ------------------------------------------------------------------
    from .constraints import add_all_constraints

    logger.info("Step 4/10: Adding constraints …")
    add_all_constraints(ctx, occ_lookup)
    print("  Constraints added: time-window, train-occupancy, resource no-overlap\n")

    # ------------------------------------------------------------------
    # 5. Add objective
    # ------------------------------------------------------------------
    from .objective import add_objective

    logger.info("Step 5/10: Setting objective function …")
    add_objective(ctx)
    print("  Objective set: priority + urgency + consolidation - conflict - unscheduled\n")

    # ------------------------------------------------------------------
    # 6. Solve
    # ------------------------------------------------------------------
    from .solver import solve

    logger.info("Step 6/10: Solving …")
    result = solve(
        ctx,
        time_limit=args.time_limit,
        num_workers=args.workers,
        seed=args.seed,
    )
    print(f"  Solver status : {result.status}")
    print(f"  Objective     : {result.objective_value}")
    print(f"  Best bound    : {result.best_bound}")
    print(f"  Wall time     : {result.wall_time_seconds}s\n")

    # ------------------------------------------------------------------
    # 7. Generate schedule
    # ------------------------------------------------------------------
    from .schedule import (
        build_schedule_df,
        group_into_blocks,
        assign_block_ids,
        build_unscheduled_df,
        build_summary,
        build_solver_log,
        save_outputs,
    )

    logger.info("Step 7/10: Generating schedule …")
    schedule_df = build_schedule_df(ctx, result)
    blocks_df = group_into_blocks(schedule_df, ctx)
    schedule_df = assign_block_ids(schedule_df, blocks_df)
    unscheduled_df = build_unscheduled_df(ctx, result)

    n_sched = len(schedule_df)
    n_unsched = len(unscheduled_df)
    n_blocks = len(blocks_df)
    print(f"  Scheduled   : {n_sched} tasks")
    print(f"  Unscheduled : {n_unsched} tasks")
    print(f"  Blocks      : {n_blocks}\n")

    # ------------------------------------------------------------------
    # 8. Validate
    # ------------------------------------------------------------------
    from .validate import validate_schedule, save_validation_report

    logger.info("Step 8/10: Validating schedule …")
    validation = validate_schedule(ctx, result, occ_lookup, schedule_df, blocks_df)
    print(f"  Validation  : {'PASSED' if validation['valid'] else 'FAILED'}")
    if not validation["valid"]:
        print(f"  Errors      : {len(validation['errors'])}")
        for err in validation["errors"][:10]:
            print(f"    • {err}")
    print()

    # ------------------------------------------------------------------
    # 9. Build summary & log
    # ------------------------------------------------------------------
    logger.info("Step 9/10: Building summary …")
    summary = build_summary(ctx, result, schedule_df, blocks_df, unscheduled_df)
    solver_log = build_solver_log(result, summary, validation)

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    logger.info("Step 10/10: Saving outputs …")
    output_dir = args.output_dir
    save_outputs(schedule_df, blocks_df, unscheduled_df, summary, solver_log, output_dir)
    save_validation_report(validation, output_dir)

    total_time = time.perf_counter() - total_start

    print("=" * 64)
    print("  OPTIMISATION COMPLETE")
    print(f"  Total time: {total_time:.2f}s")
    print()
    print("  Output files:")
    out = output_dir or "artifacts/optimization"
    for fname in [
        "optimal_schedule.csv",
        "maintenance_blocks.csv",
        "unscheduled_tasks.csv",
        "optimization_summary.json",
        "solver_log.txt",
        "validation_report.json",
    ]:
        print(f"    {out}/{fname}")
    print("=" * 64)


if __name__ == "__main__":
    main()
