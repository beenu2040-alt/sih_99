import pandas as pd
import pytest
from src.optimizer.api import optimize_schedule
from src.optimizer.model import build_model
from src.optimizer.objective import _build_combination_bonus

@pytest.fixture(scope="module")
def full_schedule_result():
    # Run optimization for a short time to verify constraints on the full dataset
    return optimize_schedule(time_limit=60)


def test_1_minimum_unscheduled_count(full_schedule_result):
    """Test 1: Verify unscheduled >= 600 for the full optimization configuration."""
    assert full_schedule_result["success"] is True
    unscheduled = len(full_schedule_result["unscheduled_df"])
    assert unscheduled >= 600, f"Expected >= 600 unscheduled, got {unscheduled}"


def test_2_cross_department_candidate_detection():
    """Test 2: Verify different department, same section, can_combine=True are candidates."""
    df = pd.DataFrame([
        {"task_id": "T1", "section_id": "S1", "department": "Eng", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
        {"task_id": "T2", "section_id": "S1", "department": "Sig", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
    ])
    ctx = build_model(df)
    bonuses = _build_combination_bonus(ctx)
    assert len(bonuses) == 1
    # Check if weight multiplier was applied (weight should be 100 * 40 // 100 * 2 = 80)
    # wait, CONSOLIDATION_WEIGHT is 40. avg_pri=100. 100 * 40//100 = 40. 40 * 2 = 80.
    _, weight = bonuses[0]
    assert weight == 80


def test_3_same_department():
    """Test 3: Verify same department tasks are not counted as cross-department."""
    df = pd.DataFrame([
        {"task_id": "T1", "section_id": "S1", "department": "Eng", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
        {"task_id": "T2", "section_id": "S1", "department": "Eng", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
    ])
    ctx = build_model(df)
    bonuses = _build_combination_bonus(ctx)
    assert len(bonuses) == 1
    _, weight = bonuses[0]
    assert weight == 40  # No 2x multiplier


def test_4_different_sections():
    """Test 4: Verify different sections tasks are not counted as cross-department."""
    df = pd.DataFrame([
        {"task_id": "T1", "section_id": "S1", "department": "Eng", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
        {"task_id": "T2", "section_id": "S2", "department": "Sig", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
    ])
    ctx = build_model(df)
    bonuses = _build_combination_bonus(ctx)
    assert len(bonuses) == 0


def test_5_can_combine_false():
    """Test 5: Verify can_combine=False tasks are not counted as coordinated."""
    df = pd.DataFrame([
        {"task_id": "T1", "section_id": "S1", "department": "Eng", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": False, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
        {"task_id": "T2", "section_id": "S1", "department": "Sig", "earliest_start_min": 0, "latest_end_min": 100, "deadline_min": 100, "duration_min": 10, "power_block_required": False, "signal_block_required": False, "track_block_required": False, "can_combine": True, "priority_int": 100, "urgency_int": 100, "conflict_cost_int": 0, "predicted_priority_score": 100.0, "predicted_priority_class": "HIGH", "duration_hours": 1.0, "train_conflict_cost": 0.0},
    ])
    ctx = build_model(df)
    bonuses = _build_combination_bonus(ctx)
    assert len(bonuses) == 0


def test_6_resource_safety(full_schedule_result):
    """Test 6: Verify Resource conflicts = 0"""
    val = full_schedule_result["validation"]
    assert val["valid"] is True
    # If the validator has detailed errors, we could check them, but 'valid' implies 0 conflicts


def test_7_train_safety(full_schedule_result):
    """Test 7: Verify Train conflicts = 0"""
    val = full_schedule_result["validation"]
    assert val["valid"] is True


def test_8_deadline_safety(full_schedule_result):
    """Test 8: Verify Deadline violations = 0"""
    val = full_schedule_result["validation"]
    assert val["valid"] is True


def test_9_duration_safety(full_schedule_result):
    """Test 9: Verify Duration violations = 0"""
    val = full_schedule_result["validation"]
    assert val["valid"] is True


def test_10_duplicate_assignments(full_schedule_result):
    """Test 10: Verify Duplicate assignments = 0"""
    val = full_schedule_result["validation"]
    assert val["valid"] is True
