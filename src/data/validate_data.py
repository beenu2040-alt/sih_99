# =============================================================================
# SYNTHETIC DATA FOR SIH PS 26027 -- Data Validation
# Runs comprehensive checks on all 7 raw datasets and outputs a report.
# =============================================================================

import os
import sys
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'validation_report.txt')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self):
        self.lines = []
        self.all_pass = True

    def add(self, msg, passed=True):
        self.lines.append(msg)
        if not passed:
            self.all_pass = False

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"\nValidation report written to {path}")


def _check_no_duplicates(df, col, name, result):
    dups = df[col].duplicated().sum()
    if dups > 0:
        result.add(f"  FAIL: {dups} duplicate {col} values in {name}", False)
        return False
    return True


def _check_no_missing(df, name, result, exclude_cols=None):
    check_df = df.drop(columns=exclude_cols or [], errors='ignore')
    missing = check_df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        detail = "; ".join(f"{c}={v}" for c, v in cols_with_missing.items())
        result.add(f"  FAIL: Missing values in {name}: {detail}", False)
        return False
    return True


def _check_fk(child_df, child_col, parent_set, parent_desc, name, result):
    """Check that all values in child_col exist in parent_set."""
    child_vals = set(child_df[child_col].dropna().astype(str).str.strip().unique())
    child_vals.discard('')
    invalid = child_vals - parent_set
    if invalid:
        sample = list(invalid)[:5]
        result.add(f"  FAIL: {len(invalid)} invalid {child_col} in {name} "
                    f"(not in {parent_desc}): {sample}", False)
        return False
    return True


def _check_date_order(df, col_before, col_after, name, result):
    """Ensure col_before <= col_after for all rows."""
    before = pd.to_datetime(df[col_before], errors='coerce')
    after = pd.to_datetime(df[col_after], errors='coerce')
    valid_mask = before.notna() & after.notna()
    bad = (before[valid_mask] > after[valid_mask]).sum()
    if bad > 0:
        result.add(f"  FAIL: {bad} rows in {name} where {col_before} > {col_after}", False)
        return False
    return True


def _check_non_negative(df, col, name, result):
    if col not in df.columns:
        return True
    vals = pd.to_numeric(df[col], errors='coerce')
    neg_count = (vals < 0).sum()
    if neg_count > 0:
        result.add(f"  FAIL: {neg_count} negative {col} values in {name}", False)
        return False
    return True


def _check_values_in_set(df, col, allowed, name, result):
    if col not in df.columns:
        return True
    vals = set(df[col].dropna().astype(str).str.strip().unique())
    invalid = vals - set(allowed)
    if invalid:
        result.add(f"  FAIL: Invalid {col} values in {name}: {invalid}", False)
        return False
    return True


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate():
    result = ValidationResult()
    result.add("=" * 56)
    result.add("DATA VALIDATION REPORT")
    result.add("SYNTHETIC DATA FOR SIH PS 26027")
    result.add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result.add("=" * 56)
    result.add("")

    # ----- Load datasets -----
    try:
        network = pd.read_csv(os.path.join(RAW_DIR, 'railway_network.csv'))
        timetable = pd.read_csv(os.path.join(RAW_DIR, 'railway_timetable.csv'))
        eng = pd.read_csv(os.path.join(RAW_DIR, 'engineering_maintenance.csv'))
        trc = pd.read_csv(os.path.join(RAW_DIR, 'traction_maintenance.csv'))
        sig = pd.read_csv(os.path.join(RAW_DIR, 'signalling_maintenance.csv'))
        forecast = pd.read_csv(os.path.join(RAW_DIR, 'goods_train_forecast.csv'))
        blocks = pd.read_csv(os.path.join(RAW_DIR, 'maintenance_block_history.csv'))
    except FileNotFoundError as e:
        result.add(f"FATAL: Could not load raw data -- {e}", False)
        result.add("\nOverall: FAIL")
        result.write(REPORT_PATH)
        return False

    datasets = {
        'railway_network': network,
        'railway_timetable': timetable,
        'engineering_maintenance': eng,
        'traction_maintenance': trc,
        'signalling_maintenance': sig,
        'goods_train_forecast': forecast,
        'maintenance_block_history': blocks,
    }

    # ===== Per-dataset record count =====
    for name, df in datasets.items():
        result.add(f"{name}: PASS ({len(df)} records)")
    result.add("")

    # ===== Build reference sets =====
    valid_sections = set(network['section_id'].astype(str).str.strip().unique())
    valid_src_stations = set(network['source_station_id'].astype(str).str.strip().unique())
    valid_dst_stations = set(network['destination_station_id'].astype(str).str.strip().unique())
    valid_stations = valid_src_stations | valid_dst_stations

    # ===== 1. Duplicate validation =====
    dup_pass = True
    dup_pass &= _check_no_duplicates(network, 'section_id', 'railway_network', result)
    dup_pass &= _check_no_duplicates(eng, 'task_id', 'engineering_maintenance', result)
    dup_pass &= _check_no_duplicates(trc, 'task_id', 'traction_maintenance', result)
    dup_pass &= _check_no_duplicates(sig, 'task_id', 'signalling_maintenance', result)
    dup_pass &= _check_no_duplicates(blocks, 'block_id', 'maintenance_block_history', result)
    result.add(f"Duplicate validation: {'PASS' if dup_pass else 'FAIL'}")

    # ===== 2. Missing value validation =====
    miss_pass = True
    for name, df in datasets.items():
        if name == 'railway_timetable':
            # section_id can be empty for last stop of each train
            miss_pass &= _check_no_missing(df, name, result, exclude_cols=['section_id'])
        else:
            miss_pass &= _check_no_missing(df, name, result)
    result.add(f"Missing value validation: {'PASS' if miss_pass else 'FAIL'}")

    # ===== 3. Foreign key validation =====
    fk_pass = True
    # Timetable station_id -> valid stations
    fk_pass &= _check_fk(timetable, 'station_id', valid_stations, 'railway_network stations', 'timetable', result)
    # Timetable section_id -> valid sections (excluding empty last-stop entries)
    tt_with_sec = timetable[timetable['section_id'].astype(str).str.strip() != ''].copy()
    if len(tt_with_sec) > 0:
        fk_pass &= _check_fk(tt_with_sec, 'section_id', valid_sections, 'railway_network', 'timetable', result)
    # Maintenance section_id
    for name, df in [('engineering_maintenance', eng), ('traction_maintenance', trc), ('signalling_maintenance', sig)]:
        fk_pass &= _check_fk(df, 'section_id', valid_sections, 'railway_network', name, result)
    # Forecast / block history section_id
    fk_pass &= _check_fk(forecast, 'section_id', valid_sections, 'railway_network', 'goods_train_forecast', result)
    fk_pass &= _check_fk(blocks, 'section_id', valid_sections, 'railway_network', 'maintenance_block_history', result)
    result.add(f"Foreign key validation: {'PASS' if fk_pass else 'FAIL'}")

    # ===== 4. Date validation =====
    date_pass = True
    for name, df in [('engineering_maintenance', eng), ('traction_maintenance', trc), ('signalling_maintenance', sig)]:
        date_pass &= _check_date_order(df, 'inspection_date', 'reported_date', name, result)
        date_pass &= _check_date_order(df, 'reported_date', 'due_date', name, result)
    # Timetable: arrival <= departure (for non-terminal stops)
    # Note: midnight crossover is valid (e.g., arrival 23:53, departure 00:12)
    tt_mid = timetable.copy()
    max_seq = tt_mid.groupby(['train_id', 'service_date'])['station_sequence'].transform('max')
    tt_mid = tt_mid[(tt_mid['station_sequence'] > 1) & (tt_mid['station_sequence'] < max_seq)]
    if len(tt_mid) > 0:
        arr = pd.to_datetime(tt_mid['arrival_time'], format='%H:%M', errors='coerce')
        dep = pd.to_datetime(tt_mid['departure_time'], format='%H:%M', errors='coerce')
        valid_both = arr.notna() & dep.notna()
        arr_hour = arr.dt.hour
        dep_hour = dep.dt.hour
        # Midnight crossover: arrival in late evening (>= 22) and departure in early morning (<= 4)
        midnight_cross = (arr_hour >= 22) & (dep_hour <= 4)
        bad = ((arr[valid_both] > dep[valid_both]) & ~midnight_cross[valid_both]).sum()
        if bad > 0:
            date_pass = False
            result.add(f"  FAIL: {bad} timetable rows where arrival > departure", False)
    result.add(f"Date validation: {'PASS' if date_pass else 'FAIL'}")

    # ===== 5. Value range validation =====
    vr_pass = True
    # Non-negative checks
    for col in ['distance_km', 'number_of_tracks', 'maximum_speed_kmph', 'section_capacity']:
        vr_pass &= _check_non_negative(network, col, 'railway_network', result)
    for df, name in [(eng, 'engineering_maintenance'), (trc, 'traction_maintenance'), (sig, 'signalling_maintenance')]:
        vr_pass &= _check_non_negative(df, 'estimated_duration_hours', name, result)
        vr_pass &= _check_non_negative(df, 'required_team_size', name, result)
    vr_pass &= _check_non_negative(forecast, 'expected_goods_trains', 'goods_train_forecast', result)
    vr_pass &= _check_non_negative(blocks, 'block_duration_hours', 'maintenance_block_history', result)
    vr_pass &= _check_non_negative(blocks, 'train_delay_minutes', 'maintenance_block_history', result)

    # Categorical checks
    vr_pass &= _check_values_in_set(network, 'track_type',
                                     ['Single', 'Double', 'Triple', 'Quadruple'], 'railway_network', result)
    vr_pass &= _check_values_in_set(timetable, 'train_type',
                                     ['Express', 'Superfast', 'Passenger', 'EMU', 'MEMU', 'Freight', 'Special'],
                                     'railway_timetable', result)
    vr_pass &= _check_values_in_set(timetable, 'direction', ['UP', 'DOWN'], 'railway_timetable', result)
    for df, name in [(eng, 'engineering_maintenance'), (trc, 'traction_maintenance'), (sig, 'signalling_maintenance')]:
        vr_pass &= _check_values_in_set(df, 'criticality',
                                         ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], name, result)
        vr_pass &= _check_values_in_set(df, 'safety_impact', ['HIGH', 'MEDIUM', 'LOW'], name, result)
        vr_pass &= _check_values_in_set(df, 'status',
                                         ['Pending', 'Scheduled', 'In Progress', 'Completed', 'Cancelled'],
                                         name, result)
    vr_pass &= _check_values_in_set(forecast, 'traffic_density',
                                     ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'], 'goods_train_forecast', result)

    # Severity range 1-5
    for df, name in [(eng, 'engineering_maintenance'), (trc, 'traction_maintenance'), (sig, 'signalling_maintenance')]:
        bad_sev = ((df['severity'] < 1) | (df['severity'] > 5)).sum()
        if bad_sev:
            vr_pass = False
            result.add(f"  FAIL: {bad_sev} severity values outside 1-5 in {name}", False)

    # Block efficiency 0-1
    bad_eff = ((blocks['block_efficiency'] < 0) | (blocks['block_efficiency'] > 1)).sum()
    if bad_eff:
        vr_pass = False
        result.add(f"  FAIL: {bad_eff} block_efficiency values outside 0-1", False)

    # completed <= planned
    bad_cp = (blocks['maintenance_tasks_completed'] > blocks['planned_tasks']).sum()
    if bad_cp:
        vr_pass = False
        result.add(f"  FAIL: {bad_cp} blocks where completed > planned", False)

    result.add(f"Value range validation: {'PASS' if vr_pass else 'FAIL'}")

    # ===== Overall =====
    result.add("")
    overall = result.all_pass
    result.add(f"Overall: {'PASS' if overall else 'FAIL'}")

    result.write(REPORT_PATH)

    # Also print to console
    for line in result.lines:
        print(line)

    return overall


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
