"""
Data loader for the OR-Tools maintenance block optimizer.

Loads CSV files, validates required columns, merges ML predictions
with optimization input, and returns clean DataFrames.
"""

import os
import logging
from typing import Tuple

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when a required data file is missing or malformed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_file(path: str) -> str:
    """Return *path* if it exists, otherwise raise ``DataLoadError``."""
    if not os.path.isfile(path):
        raise DataLoadError(
            f"Required data file not found: {path}\n"
            "Please ensure the ML pipeline has been run and the processed "
            "data directory is populated."
        )
    return path


def _validate_columns(df: pd.DataFrame, required: dict, filename: str) -> None:
    """Check that *df* contains every key listed in *required*."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataLoadError(
            f"{filename} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_optimization_input(filepath: str | None = None) -> pd.DataFrame:
    """
    Load ``optimization_input.csv`` and validate its schema.

    Returns a DataFrame with canonical column names.
    """
    filepath = filepath or cfg.OPTIMIZATION_INPUT_FILE
    _require_file(filepath)

    df = pd.read_csv(filepath)
    _validate_columns(df, cfg.OPT_INPUT_COLUMNS, filepath)

    # Rename to canonical names (currently identity but future-proofs mapping)
    df = df.rename(columns=cfg.OPT_INPUT_COLUMNS)

    logger.info("Loaded %d tasks from %s", len(df), filepath)
    return df


def load_predictions(filepath: str | None = None) -> pd.DataFrame:
    """
    Load ``maintenance_predictions.csv`` and validate its schema.
    """
    filepath = filepath or cfg.MAINTENANCE_PREDICTIONS_FILE
    _require_file(filepath)

    df = pd.read_csv(filepath)
    _validate_columns(df, cfg.PRED_COLUMNS, filepath)
    df = df.rename(columns=cfg.PRED_COLUMNS)

    logger.info("Loaded %d predictions from %s", len(df), filepath)
    return df


def load_train_occupancy(filepath: str | None = None) -> pd.DataFrame:
    """
    Load ``train_occupancy.csv`` and validate its schema.
    """
    filepath = filepath or cfg.TRAIN_OCCUPANCY_FILE
    _require_file(filepath)

    df = pd.read_csv(filepath)
    _validate_columns(df, cfg.OCCUPANCY_COLUMNS, filepath)
    df = df.rename(columns=cfg.OCCUPANCY_COLUMNS)

    logger.info("Loaded %d occupancy records from %s", len(df), filepath)
    return df


def merge_predictions(
    opt_input: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join *opt_input* with *predictions* on ``task_id``.

    The ``predicted_priority_score`` from the ML model replaces the
    training-label ``priority_score`` as the primary priority value.

    Raises ``DataLoadError`` if critical tasks disappear during the merge.
    """
    n_before = len(opt_input)
    merged = opt_input.merge(predictions, on="task_id", how="left")

    # Sanity: no rows should be lost
    if len(merged) != n_before:
        raise DataLoadError(
            f"Row count changed during merge: {n_before} → {len(merged)}. "
            "Check for duplicate task_ids in predictions."
        )

    # Report missing predictions
    n_missing = merged["predicted_priority_score"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d tasks have no ML prediction — falling back to original "
            "priority_score for those tasks.",
            n_missing,
        )
        merged["predicted_priority_score"] = merged[
            "predicted_priority_score"
        ].fillna(merged["priority_score"])
        merged["predicted_priority_class"] = merged[
            "predicted_priority_class"
        ].fillna("UNKNOWN")

    logger.info(
        "Merged predictions: %d tasks, %d with ML predictions",
        len(merged),
        len(merged) - n_missing,
    )
    return merged


def load_all(
    opt_path: str | None = None,
    pred_path: str | None = None,
    occ_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: load, validate, and merge all inputs.

    Returns
    -------
    tasks : DataFrame
        Merged optimisation input + ML predictions.
    occupancy : DataFrame
        Train occupancy data.
    """
    opt_input = load_optimization_input(opt_path)
    predictions = load_predictions(pred_path)
    occupancy = load_train_occupancy(occ_path)

    tasks = merge_predictions(opt_input, predictions)
    return tasks, occupancy
