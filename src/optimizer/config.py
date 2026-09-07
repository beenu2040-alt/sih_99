"""
Centralized configuration for the OR-Tools CP-SAT Maintenance Block Optimizer.

All tuneable parameters are defined here with sensible defaults.
Values can be overridden via environment variables where practical.
"""

import os
import multiprocessing


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable, falling back to *default*."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_float(name: str, default: float) -> float:
    """Read a float from an environment variable, falling back to *default*."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# Solver parameters
# ---------------------------------------------------------------------------

SOLVER_TIME_LIMIT_SECONDS: int = _env_int("OPT_TIME_LIMIT", 60)
"""Maximum wall-clock time the CP-SAT solver is allowed to run."""

NUM_WORKERS: int = _env_int(
    "OPT_NUM_WORKERS", min(8, multiprocessing.cpu_count())
)
"""Number of parallel search workers for the solver."""

RANDOM_SEED: int = _env_int("OPT_RANDOM_SEED", 42)
"""Fixed random seed for reproducibility."""

# ---------------------------------------------------------------------------
# Time representation
# ---------------------------------------------------------------------------

TIME_GRANULARITY_MINUTES: int = _env_int("OPT_TIME_GRANULARITY", 15)
"""Granularity of the integer time grid (minutes)."""

GLOBAL_EPOCH: str = "2025-01-01 00:00"
"""Reference datetime from which all integer time offsets are measured."""

MAX_BLOCK_DURATION_MINUTES: int = _env_int("OPT_MAX_BLOCK_DUR", 720)
"""Maximum allowed duration for a single maintenance block (12 h)."""

MIN_BLOCK_DURATION_MINUTES: int = _env_int("OPT_MIN_BLOCK_DUR", 60)
"""Minimum allowed duration for a single maintenance block (1 h)."""

# ---------------------------------------------------------------------------
# Priority & objective scaling
# ---------------------------------------------------------------------------

PRIORITY_SCALE_FACTOR: int = _env_int("OPT_PRIORITY_SCALE", 10)
"""
Multiplier used to convert floating-point priority scores (0–100)
to integers suitable for CP-SAT coefficients.  95.2 → 952.
"""

PRIORITY_MIN: float = 0.0
"""Lower bound of the raw priority score range."""

PRIORITY_MAX: float = 100.0
"""Upper bound of the raw priority score range."""

# ---------------------------------------------------------------------------
# Objective weights
#
# These are *policy* parameters — tuneable knobs that reflect how much the
# railway operator values each objective component.  They are NOT learned
# from data.  Defaults have been chosen to produce reasonable schedules on
# the synthetic SIH dataset.
# ---------------------------------------------------------------------------

PRIORITY_WEIGHT: int = _env_int("OPT_W_PRIORITY", 100)
"""
Weight for scheduling high-priority tasks.
Higher → the solver works harder to schedule critical maintenance.
"""

DEADLINE_WEIGHT: int = _env_int("OPT_W_DEADLINE", 50)
"""
Weight for deadline urgency.
Higher → the solver strongly prefers tasks whose deadlines are imminent.
"""

TRAIN_CONFLICT_WEIGHT: int = _env_int("OPT_W_TRAIN_CONFLICT", 30)
"""
Penalty weight for train conflict cost.
Higher → the solver avoids scheduling maintenance during busy traffic periods.
"""

BLOCK_COUNT_WEIGHT: int = _env_int("OPT_W_BLOCK_COUNT", 200)
"""
Penalty per maintenance block opened.
Higher → the solver consolidates tasks into fewer blocks to reduce disruption.
"""

CONSOLIDATION_WEIGHT: int = _env_int("OPT_W_CONSOLIDATION", 40)
"""
Bonus for combining compatible tasks into the same block window.
Higher → the solver tries harder to group combinable tasks.
"""

UNSCHEDULED_PENALTY: int = _env_int("OPT_W_UNSCHEDULED", 80)
"""
Penalty for leaving a task unscheduled, scaled by its priority.
Higher → the solver avoids leaving important tasks unscheduled.
"""

SAFETY_WEIGHT: int = _env_int("OPT_W_SAFETY", 20)
"""
Bonus weight for tasks with high safety impact.
Higher → the solver prioritises safety-critical maintenance.
"""

# ---------------------------------------------------------------------------
# File paths (relative to project root)
# ---------------------------------------------------------------------------

DATA_DIR: str = os.path.join("data", "processed")
ARTIFACTS_DIR: str = os.path.join("artifacts", "optimization")

OPTIMIZATION_INPUT_FILE: str = os.path.join(DATA_DIR, "optimization_input.csv")
MAINTENANCE_PREDICTIONS_FILE: str = os.path.join(DATA_DIR, "maintenance_predictions.csv")
TRAIN_OCCUPANCY_FILE: str = os.path.join(DATA_DIR, "train_occupancy.csv")

OUTPUT_SCHEDULE_FILE: str = os.path.join(ARTIFACTS_DIR, "optimal_schedule.csv")
OUTPUT_BLOCKS_FILE: str = os.path.join(ARTIFACTS_DIR, "maintenance_blocks.csv")
OUTPUT_UNSCHEDULED_FILE: str = os.path.join(ARTIFACTS_DIR, "unscheduled_tasks.csv")
OUTPUT_SUMMARY_FILE: str = os.path.join(ARTIFACTS_DIR, "optimization_summary.json")
OUTPUT_SOLVER_LOG_FILE: str = os.path.join(ARTIFACTS_DIR, "solver_log.txt")
OUTPUT_VALIDATION_FILE: str = os.path.join(ARTIFACTS_DIR, "validation_report.json")

# ---------------------------------------------------------------------------
# Column name mappings  (actual CSV column → canonical internal name)
# ---------------------------------------------------------------------------

# optimization_input.csv
OPT_INPUT_COLUMNS = {
    "task_id": "task_id",
    "section_id": "section_id",
    "department": "department",
    "priority_score": "priority_score",
    "duration_hours": "duration_hours",
    "earliest_start": "earliest_start",
    "latest_end": "latest_end",
    "deadline": "deadline",
    "train_conflict_cost": "train_conflict_cost",
    "traffic_density": "traffic_density",
    "power_block_required": "power_block_required",
    "signal_block_required": "signal_block_required",
    "track_block_required": "track_block_required",
    "can_combine": "can_combine",
}

# maintenance_predictions.csv
PRED_COLUMNS = {
    "task_id": "task_id",
    "predicted_priority_score": "predicted_priority_score",
    "predicted_priority_class": "predicted_priority_class",
}

# train_occupancy.csv
OCCUPANCY_COLUMNS = {
    "section_id": "section_id",
    "date": "date",
    "time_window_start": "time_window_start",
    "time_window_end": "time_window_end",
    "occupied": "occupied",
    "train_count": "train_count",
    "traffic_density": "traffic_density",
}

# ---------------------------------------------------------------------------
# Priority class thresholds (matching src/ml/predict.py score_to_class)
# ---------------------------------------------------------------------------

PRIORITY_CLASS_THRESHOLDS = {
    "CRITICAL": 80,
    "HIGH": 60,
    "MEDIUM": 40,
    "LOW": 0,
}
