"""
Preprocessing for the OR-Tools maintenance block optimizer.

Converts raw DataFrame columns into integer time representations
suitable for CP-SAT, validates inputs, and computes derived quantities
(urgency scores, scaled priority coefficients, occupancy lookup).
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised for unrecoverable data-quality issues."""


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

_EPOCH = datetime.strptime(cfg.GLOBAL_EPOCH, "%Y-%m-%d %H:%M")
_GRAN = cfg.TIME_GRANULARITY_MINUTES


def datetime_to_minutes(dt: datetime) -> int:
    """Convert a datetime to integer minutes since the global epoch."""
    delta = dt - _EPOCH
    total_minutes = delta.total_seconds() / 60.0
    return int(math.floor(total_minutes))


def minutes_to_datetime(minutes: int) -> datetime:
    """Convert integer minutes since epoch back to a datetime."""
    return _EPOCH + timedelta(minutes=int(minutes))


def snap_to_grid(minutes: int, direction: str = "down") -> int:
    """
    Snap *minutes* to the nearest grid boundary defined by
    ``TIME_GRANULARITY_MINUTES``.

    Parameters
    ----------
    direction : ``"down"`` | ``"up"``
    """
    if direction == "down":
        return (minutes // _GRAN) * _GRAN
    else:  # up
        return math.ceil(minutes / _GRAN) * _GRAN


def duration_hours_to_minutes(hours: float) -> int:
    """Convert a floating-point duration in hours to integer minutes, rounded up."""
    raw = hours * 60.0
    snapped = max(_GRAN, snap_to_grid(int(math.ceil(raw)), direction="up"))
    return snapped


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the merged task DataFrame.

    Returns the (possibly reduced) DataFrame with only valid rows.
    Invalid rows are logged as warnings.
    """
    issues: List[str] = []
    valid_mask = pd.Series(True, index=df.index)

    # --- task_id must exist and be unique ---
    if df["task_id"].isna().any():
        bad = df["task_id"].isna().sum()
        issues.append(f"{bad} rows have null task_id — dropped")
        valid_mask &= df["task_id"].notna()

    dups = df[valid_mask].duplicated(subset="task_id", keep="first")
    if dups.any():
        n = dups.sum()
        issues.append(f"{n} duplicate task_ids — keeping first occurrence")
        valid_mask.loc[dups[dups].index] = False

    # --- priority score ---
    ps = df["predicted_priority_score"]
    bad_ps = ps.isna() | (ps < 0)
    if bad_ps.any():
        issues.append(
            f"{bad_ps.sum()} tasks have invalid predicted_priority_score — clamped to 0"
        )
        df.loc[bad_ps, "predicted_priority_score"] = 0.0

    # Clamp to [0, 100] (predictions may slightly exceed training range)
    df["predicted_priority_score"] = df["predicted_priority_score"].clip(
        cfg.PRIORITY_MIN, cfg.PRIORITY_MAX
    )

    # --- duration ---
    dur = df["duration_hours"]
    bad_dur = dur.isna() | (dur <= 0)
    if bad_dur.any():
        issues.append(
            f"{bad_dur.sum()} tasks have invalid duration_hours — dropped"
        )
        valid_mask &= ~bad_dur

    # --- Parse datetime columns ---
    for col in ("earliest_start", "latest_end"):
        df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce")
        bad = df[col + "_dt"].isna()
        if bad.any():
            issues.append(f"{bad.sum()} tasks have unparseable {col} — dropped")
            valid_mask &= ~bad

    df["deadline_dt"] = pd.to_datetime(df["deadline"], errors="coerce")
    bad_dl = df["deadline_dt"].isna()
    if bad_dl.any():
        issues.append(
            f"{bad_dl.sum()} tasks have unparseable deadline — "
            "using latest_end as fallback"
        )
        df.loc[bad_dl, "deadline_dt"] = df.loc[bad_dl, "latest_end_dt"]

    # --- Time-window sanity ---
    bad_window = df["earliest_start_dt"] >= df["latest_end_dt"]
    if bad_window.any():
        issues.append(
            f"{bad_window.sum()} tasks have earliest_start >= latest_end — dropped"
        )
        valid_mask &= ~bad_window

    # Report
    for issue in issues:
        logger.warning("Validation: %s", issue)

    df_valid = df[valid_mask].copy()
    n_dropped = len(df) - len(df_valid)
    if n_dropped:
        logger.warning("Dropped %d invalid tasks (%d remain)", n_dropped, len(df_valid))
    else:
        logger.info("All %d tasks passed validation", len(df_valid))

    return df_valid


# ---------------------------------------------------------------------------
# Compute integer time columns
# ---------------------------------------------------------------------------

def add_integer_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add integer-minute columns to the task DataFrame:

    * ``earliest_start_min``
    * ``latest_end_min``
    * ``deadline_min``
    * ``duration_min``
    """
    df = df.copy()

    df["earliest_start_min"] = df["earliest_start_dt"].apply(
        lambda dt: snap_to_grid(datetime_to_minutes(dt), "up")
    )
    df["latest_end_min"] = df["latest_end_dt"].apply(
        lambda dt: snap_to_grid(datetime_to_minutes(dt), "down")
    )
    df["deadline_min"] = df["deadline_dt"].apply(
        lambda dt: snap_to_grid(
            datetime_to_minutes(dt + timedelta(hours=23, minutes=59)),
            "down",
        )
    )

    df["duration_min"] = df["duration_hours"].apply(duration_hours_to_minutes)

    # Drop tasks whose window is too small for their duration
    too_short = (df["latest_end_min"] - df["earliest_start_min"]) < df["duration_min"]
    if too_short.any():
        logger.warning(
            "%d tasks have time windows shorter than their duration — "
            "they will be marked optional (unlikely to schedule).",
            too_short.sum(),
        )

    return df


# ---------------------------------------------------------------------------
# Priority / urgency scaling
# ---------------------------------------------------------------------------

def compute_priority_int(df: pd.DataFrame) -> pd.DataFrame:
    """Scale ``predicted_priority_score`` to an integer coefficient for CP-SAT."""
    df = df.copy()
    df["priority_int"] = (
        df["predicted_priority_score"] * cfg.PRIORITY_SCALE_FACTOR
    ).round().astype(int)
    return df


def compute_urgency_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an integer urgency score based on how close the deadline is
    to the earliest start.

    The urgency is higher when the available slack is small.
    """
    df = df.copy()

    # Slack = deadline - earliest_start (in minutes)
    slack = (df["deadline_min"] - df["earliest_start_min"]).clip(lower=1)
    duration = df["duration_min"]

    # Ratio: how much of the window is consumed by the task itself
    # ratio near 1.0 → very tight → high urgency
    ratio = (duration / slack).clip(0.0, 1.0)

    # Scale to 0–1000 integer
    df["urgency_int"] = (ratio * 1000).round().astype(int)
    return df


def compute_conflict_cost_int(df: pd.DataFrame) -> pd.DataFrame:
    """Scale ``train_conflict_cost`` to integer."""
    df = df.copy()
    df["conflict_cost_int"] = (df["train_conflict_cost"]).round().astype(int)
    return df


# ---------------------------------------------------------------------------
# Occupancy lookup
# ---------------------------------------------------------------------------

OccupancyLookup = Dict[str, List[Tuple[int, int]]]


def build_occupancy_lookup(
    occ_df: pd.DataFrame,
    task_sections: set | None = None,
) -> OccupancyLookup:
    """
    Build a mapping from ``section_id`` to a sorted list of
    ``(start_min, end_min)`` occupied intervals.

    If *task_sections* is given, only sections relevant to the current
    task set are included (performance optimization).
    """
    lookup: OccupancyLookup = {}

    # Keep only occupied windows
    occupied = occ_df[occ_df["occupied"] == True].copy()  # noqa: E712

    if task_sections is not None:
        occupied = occupied[occupied["section_id"].isin(task_sections)]

    if occupied.empty:
        logger.info("No occupied train windows found for relevant sections.")
        return lookup

    for section_id, group in occupied.groupby("section_id"):
        intervals = []
        for _, row in group.iterrows():
            date_str = str(row["date"])
            start_str = str(row["time_window_start"])
            end_str = str(row["time_window_end"])

            try:
                start_dt = datetime.strptime(f"{date_str} {start_str}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date_str} {end_str}", "%Y-%m-%d %H:%M")
                # Handle midnight crossing (e.g., 22:00 → 00:00)
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)

                s_min = datetime_to_minutes(start_dt)
                e_min = datetime_to_minutes(end_dt)
                intervals.append((s_min, e_min))
            except (ValueError, TypeError):
                continue

        # Sort and store
        intervals.sort()
        if intervals:
            lookup[str(section_id)] = intervals

    logger.info(
        "Built occupancy lookup: %d sections, %d total occupied windows",
        len(lookup),
        sum(len(v) for v in lookup.values()),
    )
    return lookup


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    tasks_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, OccupancyLookup]:
    """
    Run the full preprocessing pipeline.

    Returns
    -------
    tasks : DataFrame
        Validated, time-converted, coefficient-scaled task data.
    occupancy_lookup : dict
        ``{section_id: [(start_min, end_min), ...]}``
    """
    # 1. Validate
    tasks = validate_tasks(tasks_df)

    # 2. Integer times
    tasks = add_integer_times(tasks)

    # 3. Coefficients
    tasks = compute_priority_int(tasks)
    tasks = compute_urgency_int(tasks)
    tasks = compute_conflict_cost_int(tasks)

    # 4. Occupancy
    sections = set(tasks["section_id"].unique())
    occ_lookup = build_occupancy_lookup(occupancy_df, task_sections=sections)

    logger.info(
        "Preprocessing complete: %d tasks ready for optimization", len(tasks)
    )
    return tasks, occ_lookup
