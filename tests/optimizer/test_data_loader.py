import os
import pandas as pd
import pytest
from src.optimizer.data_loader import load_optimization_input, merge_predictions, DataLoadError

def test_load_valid_csv(tmp_path):
    f = tmp_path / "opt.csv"
    f.write_text("task_id,section_id,department,priority_score,duration_hours,earliest_start,latest_end,deadline,train_conflict_cost,traffic_density,power_block_required,signal_block_required,track_block_required,can_combine\nT1,S1,D1,80,2,2025-06-01,2025-06-02,2025-06-02,10,0.5,False,False,True,True\n")
    df = load_optimization_input(str(f))
    assert len(df) == 1
    assert df.iloc[0]["task_id"] == "T1"

def test_missing_file():
    with pytest.raises(DataLoadError):
        load_optimization_input("nonexistent.csv")

def test_missing_columns(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("task_id,section_id\nT1,S1\n")
    with pytest.raises(DataLoadError, match="missing required columns"):
        load_optimization_input(str(f))

def test_merge_predictions():
    opt = pd.DataFrame({"task_id": ["T1", "T2"], "priority_score": [50.0, 60.0]})
    pred = pd.DataFrame({"task_id": ["T1"], "predicted_priority_score": [95.0], "predicted_priority_class": ["CRITICAL"]})
    merged = merge_predictions(opt, pred)
    assert len(merged) == 2
    
    t1 = merged[merged["task_id"] == "T1"].iloc[0]
    assert t1["predicted_priority_score"] == 95.0
    
    t2 = merged[merged["task_id"] == "T2"].iloc[0]
    assert t2["predicted_priority_score"] == 60.0
