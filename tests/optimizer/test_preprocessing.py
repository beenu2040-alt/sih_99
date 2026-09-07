import pandas as pd
import pytest
from datetime import datetime
from src.optimizer.preprocessing import (
    datetime_to_minutes, minutes_to_datetime, duration_hours_to_minutes,
    snap_to_grid, validate_tasks, compute_priority_int, compute_urgency_int,
    build_occupancy_lookup
)
import src.optimizer.config as cfg

def test_datetime_to_minutes():
    dt = datetime(2025, 1, 1, 1, 0)
    assert datetime_to_minutes(dt) == 60

def test_minutes_to_datetime():
    dt = minutes_to_datetime(60)
    assert dt == datetime(2025, 1, 1, 1, 0)

def test_duration_hours_to_minutes():
    cfg.TIME_GRANULARITY_MINUTES = 15
    assert duration_hours_to_minutes(2.0) == 120
    assert duration_hours_to_minutes(1.5) == 90

def test_snap_to_grid():
    cfg.TIME_GRANULARITY_MINUTES = 15
    assert snap_to_grid(17, "down") == 15
    assert snap_to_grid(17, "up") == 30

def test_validate_tasks():
    df = pd.DataFrame([
        {"task_id": "T1", "duration_hours": 2, "earliest_start": "2025-01-01 00:00", "latest_end": "2025-01-02 00:00", "deadline": "2025-01-02 00:00", "predicted_priority_score": 50},
        {"task_id": None, "duration_hours": 2, "earliest_start": "2025-01-01 00:00", "latest_end": "2025-01-02 00:00", "deadline": "2025-01-02 00:00", "predicted_priority_score": 50},
        {"task_id": "T3", "duration_hours": -1, "earliest_start": "2025-01-01 00:00", "latest_end": "2025-01-02 00:00", "deadline": "2025-01-02 00:00", "predicted_priority_score": 50},
        {"task_id": "T4", "duration_hours": 2, "earliest_start": "2025-01-02 00:00", "latest_end": "2025-01-01 00:00", "deadline": "2025-01-02 00:00", "predicted_priority_score": 50},
    ])
    valid = validate_tasks(df)
    assert len(valid) == 1
    assert valid.iloc[0]["task_id"] == "T1"

def test_compute_priority_int():
    df = pd.DataFrame([{"predicted_priority_score": 95.0}])
    cfg.PRIORITY_SCALE_FACTOR = 10
    res = compute_priority_int(df)
    assert res.iloc[0]["priority_int"] == 950

def test_compute_urgency_int():
    df = pd.DataFrame([{"duration_min": 120, "deadline_min": 1000, "earliest_start_min": 880}])
    res = compute_urgency_int(df)
    assert res.iloc[0]["urgency_int"] == 1000

def test_build_occupancy_lookup():
    occ_df = pd.DataFrame([
        {"section_id": "S1", "date": "2025-01-01", "time_window_start": "10:00", "time_window_end": "11:00", "occupied": True}
    ])
    lookup = build_occupancy_lookup(occ_df)
    assert "S1" in lookup
    assert len(lookup["S1"]) == 1
