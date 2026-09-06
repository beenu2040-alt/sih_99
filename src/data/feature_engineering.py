# =============================================================================
# SYNTHETIC DATA FOR SIH PS 26027 -- Feature Engineering
# Produces three ML/optimizer-ready datasets from processed data:
#   1. maintenance_ml_dataset.csv   (XGBoost-ready)
#   2. optimization_input.csv       (OR-Tools-ready)
#   3. train_occupancy.csv          (section time-window occupancy)
# =============================================================================

import os
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def ensure_dirs():
    os.makedirs(PROC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asset_age(asset_id):
    """Deterministic synthetic age 0.5-30 years derived from asset_id hash."""
    h = int(hashlib.md5(asset_id.encode()).hexdigest(), 16)
    return round(0.5 + (h % 2950) / 100.0, 1)  # 0.5 to 30.0


def _safety_encode(val):
    return {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(str(val).strip(), 1)


def _criticality_score(val):
    return {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(str(val).strip(), 1)


# ---------------------------------------------------------------------------
# 1. Maintenance ML Dataset
# ---------------------------------------------------------------------------

def build_ml_dataset():
    """Merge all 3 maintenance datasets and compute ML features."""
    print("[1/3] Building maintenance_ml_dataset.csv ...")

    # Load processed maintenance datasets
    eng = pd.read_csv(os.path.join(PROC_DIR, 'engineering_maintenance_processed.csv'))
    trc = pd.read_csv(os.path.join(PROC_DIR, 'traction_maintenance_processed.csv'))
    sig = pd.read_csv(os.path.join(PROC_DIR, 'signalling_maintenance_processed.csv'))

    # Add department labels
    eng['department'] = 'Engineering'
    trc['department'] = 'Traction'
    sig['department'] = 'Signalling'

    # Standardize columns: engineering has no asset_type
    if 'asset_type' not in eng.columns:
        eng['asset_type'] = 'Track'

    # Block requirement columns
    eng['power_block_required'] = False
    eng['signal_block_required'] = False
    eng['track_block_required'] = True

    if 'power_block_required' not in trc.columns:
        trc['power_block_required'] = True
    trc['signal_block_required'] = False
    trc['track_block_required'] = False

    sig['power_block_required'] = False
    if 'signal_block_required' not in sig.columns:
        sig['signal_block_required'] = True
    sig['track_block_required'] = False

    # Unify
    common_cols = [
        'task_id', 'department', 'asset_id', 'section_id', 'asset_type',
        'task_type', 'defect_type', 'severity', 'criticality',
        'inspection_date', 'reported_date', 'due_date',
        'estimated_duration_hours', 'required_team_size', 'safety_impact',
        'status', 'power_block_required', 'signal_block_required', 'track_block_required',
    ]
    combined = pd.concat([eng[common_cols], trc[common_cols], sig[common_cols]],
                         ignore_index=True)

    # ----- Compute features -----

    # days_to_deadline
    reported = pd.to_datetime(combined['reported_date'], errors='coerce')
    due = pd.to_datetime(combined['due_date'], errors='coerce')
    combined['days_to_deadline'] = (due - reported).dt.days.fillna(30).astype(int)

    # asset_age
    combined['asset_age'] = combined['asset_id'].apply(_asset_age)

    # failure_history: count of tasks per asset across the whole dataset
    asset_counts = combined.groupby('asset_id')['task_id'].count().rename('failure_history')
    combined = combined.merge(asset_counts, on='asset_id', how='left')
    combined['failure_history'] = combined['failure_history'].clip(upper=10).astype(int)

    # maintenance_frequency: tasks per month per section
    section_counts = combined.groupby('section_id')['task_id'].count()
    section_freq = (section_counts / 12.0).rename('maintenance_frequency').round(2)
    combined = combined.merge(section_freq, on='section_id', how='left')

    # safety_impact_encoded
    combined['safety_impact_encoded'] = combined['safety_impact'].apply(_safety_encode)

    # ----- Traffic density from forecast -----
    try:
        forecast = pd.read_csv(os.path.join(PROC_DIR, 'goods_train_forecast_processed.csv'))
        density_map_values = {'LOW': 0.2, 'MEDIUM': 0.45, 'HIGH': 0.7, 'VERY_HIGH': 0.9}
        forecast['density_num'] = forecast['traffic_density'].map(density_map_values).fillna(0.3)
        sec_density = forecast.groupby('section_id')['density_num'].mean().rename('train_traffic_density').round(3)
        combined = combined.merge(sec_density, on='section_id', how='left')
        combined['train_traffic_density'] = combined['train_traffic_density'].fillna(0.3)
    except FileNotFoundError:
        combined['train_traffic_density'] = 0.3

    # expected_train_conflicts: duration x traffic density x scaling factor
    combined['expected_train_conflicts'] = (
        combined['estimated_duration_hours'] * combined['train_traffic_density'] * 5.0
    ).round(1)

    # ----- historical_failure_rate from block history -----
    try:
        blocks = pd.read_csv(os.path.join(PROC_DIR, 'maintenance_block_history_processed.csv'))
        block_rate = blocks.groupby('section_id').agg(
            total_blocks=('block_id', 'count'),
            total_conflicts=('train_conflicts', 'sum')
        )
        block_rate['historical_failure_rate'] = (block_rate['total_conflicts'] / block_rate['total_blocks']).round(2)
        combined = combined.merge(block_rate[['historical_failure_rate']], on='section_id', how='left')
        combined['historical_failure_rate'] = combined['historical_failure_rate'].fillna(2.0)
    except FileNotFoundError:
        combined['historical_failure_rate'] = 2.0

    # ----- Priority score (0-100) -----
    # Weighted: severity (30%), criticality (20%), safety (15%), urgency (20%), traffic (15%)
    severity_norm = (6 - combined['severity']) / 4.0  # 1->1.25, 5->0.25
    crit_norm = combined['criticality'].apply(_criticality_score) / 4.0
    safety_norm = combined['safety_impact_encoded'] / 3.0
    urgency_norm = (1.0 - combined['days_to_deadline'].clip(0, 90) / 90.0)
    traffic_norm = combined['train_traffic_density']

    combined['priority_score'] = (
        severity_norm * 30 + crit_norm * 20 + safety_norm * 15 +
        urgency_norm * 20 + traffic_norm * 15
    ).round(1).clip(0, 100)

    # priority_class
    combined['priority_class'] = pd.cut(
        combined['priority_score'],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )

    # Select final columns
    ml_cols = [
        'task_id', 'department', 'section_id', 'asset_id', 'asset_type',
        'task_type', 'defect_type', 'severity', 'criticality',
        'days_to_deadline', 'asset_age', 'failure_history',
        'maintenance_frequency', 'estimated_duration_hours',
        'required_team_size', 'safety_impact_encoded',
        'train_traffic_density', 'expected_train_conflicts',
        'historical_failure_rate', 'priority_score', 'priority_class',
        'power_block_required', 'signal_block_required', 'track_block_required',
        'status',
    ]
    ml_df = combined[ml_cols].copy()
    ml_df.to_csv(os.path.join(PROC_DIR, 'maintenance_ml_dataset.csv'), index=False)
    print(f"    --> {len(ml_df)} records, {len(ml_cols)} features")
    print(f"    Priority distribution:")
    print(ml_df['priority_class'].value_counts().to_string())
    return combined


# ---------------------------------------------------------------------------
# 2. Optimization Input
# ---------------------------------------------------------------------------

def build_optimization_input(combined_df):
    """Filter to Pending/Scheduled tasks and build optimizer-ready dataset."""
    print("\n[2/3] Building optimization_input.csv ...")

    # Only actionable tasks
    active = combined_df[combined_df['status'].isin(['Pending', 'Scheduled'])].copy()

    if len(active) == 0:
        print("    WARNING: No Pending/Scheduled tasks found!")
        active = combined_df.head(100).copy()

    # earliest_start: reported_date at 06:00
    active['earliest_start'] = pd.to_datetime(active['reported_date'], errors='coerce').dt.strftime('%Y-%m-%d') + ' 06:00'
    # latest_end: due_date at 22:00
    active['latest_end'] = pd.to_datetime(active['due_date'], errors='coerce').dt.strftime('%Y-%m-%d') + ' 22:00'
    # deadline
    active['deadline'] = pd.to_datetime(active['due_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # train_conflict_cost: higher for busy sections
    active['train_conflict_cost'] = (active['train_traffic_density'] * 100 * active['estimated_duration_hours']).round(1)

    # traffic_density (numeric)
    active['traffic_density'] = active['train_traffic_density']

    # can_combine: True if other departments have pending tasks on same section within 7 days
    due_dt = pd.to_datetime(active['due_date'], errors='coerce')
    active_copy = active[['section_id', 'department', 'due_date']].copy()
    active_copy['due_dt'] = pd.to_datetime(active_copy['due_date'], errors='coerce')

    # Vectorized approach for can_combine
    can_combine_list = []
    for idx, row in active.iterrows():
        same_section = active_copy[
            (active_copy['section_id'] == row['section_id']) &
            (active_copy['department'] != row['department'])
        ]
        if len(same_section) == 0:
            can_combine_list.append(False)
            continue
        row_due = pd.to_datetime(row['due_date'], errors='coerce')
        if pd.isna(row_due):
            can_combine_list.append(False)
            continue
        nearby = same_section[
            (same_section['due_dt'] >= row_due - pd.Timedelta(days=7)) &
            (same_section['due_dt'] <= row_due + pd.Timedelta(days=7))
        ]
        can_combine_list.append(len(nearby) > 0)
    active['can_combine'] = can_combine_list

    opt_cols = [
        'task_id', 'section_id', 'department', 'priority_score',
        'estimated_duration_hours', 'earliest_start', 'latest_end',
        'deadline', 'train_conflict_cost', 'traffic_density',
        'power_block_required', 'signal_block_required', 'track_block_required',
        'can_combine',
    ]
    opt_df = active[opt_cols].copy()
    opt_df = opt_df.rename(columns={'estimated_duration_hours': 'duration_hours'})

    opt_df.to_csv(os.path.join(PROC_DIR, 'optimization_input.csv'), index=False)
    print(f"    --> {len(opt_df)} actionable tasks for optimizer")
    return opt_df


# ---------------------------------------------------------------------------
# 3. Train Occupancy
# ---------------------------------------------------------------------------

def build_train_occupancy():
    """Build section-level occupancy from timetable + forecast."""
    print("\n[3/3] Building train_occupancy.csv ...")

    # Load timetable
    try:
        tt = pd.read_csv(os.path.join(PROC_DIR, 'railway_timetable_processed.csv'))
    except FileNotFoundError:
        tt = pd.read_csv(os.path.join(RAW_DIR, 'railway_timetable.csv'))

    # Load network for section list
    try:
        net = pd.read_csv(os.path.join(PROC_DIR, 'railway_network_processed.csv'))
    except FileNotFoundError:
        net = pd.read_csv(os.path.join(RAW_DIR, 'railway_network.csv'))

    section_ids = net['section_id'].unique().tolist()
    section_capacity = dict(zip(net['section_id'], net['section_capacity']))

    # Get representative dates from forecast
    try:
        fcst = pd.read_csv(os.path.join(PROC_DIR, 'goods_train_forecast_processed.csv'))
        dates = sorted(fcst['forecast_date'].unique())
    except FileNotFoundError:
        dates = sorted(tt['service_date'].unique()[:30])

    # 2-hour windows
    windows = [(f"{h*2:02d}:00", f"{(h*2+2) % 24:02d}:00") for h in range(12)]

    # Pre-compute timetable counts per section/date/window
    tt_valid = tt[tt['section_id'].astype(str).str.strip() != ''].copy()
    tt_valid['dep_hour'] = pd.to_datetime(tt_valid['departure_time'], format='%H:%M', errors='coerce').dt.hour

    # Group by section_id and service_date
    tt_grouped = {}
    for (sec, sdate), grp in tt_valid.groupby(['section_id', 'service_date']):
        tt_grouped[(sec, sdate)] = grp['dep_hour'].tolist()

    records = []
    for date_str in dates:
        for sid in section_ids:
            hours_list = tt_grouped.get((sid, date_str), [])
            cap = section_capacity.get(sid, 40)
            for w_start, w_end in windows:
                start_h = int(w_start.split(':')[0])
                end_h = start_h + 2

                # Count trains in this window
                train_count = sum(1 for h in hours_list if start_h <= h < end_h)
                occupied = train_count > 0
                density = round(min(1.0, train_count / max(1, cap / 12.0)), 3)

                records.append({
                    'section_id': sid,
                    'date': date_str,
                    'time_window_start': w_start,
                    'time_window_end': w_end,
                    'occupied': occupied,
                    'train_count': train_count,
                    'traffic_density': density,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(PROC_DIR, 'train_occupancy.csv'), index=False)
    print(f"    --> {len(df)} occupancy records")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING -- SIH PS 26027")
    print("=" * 60 + "\n")

    ensure_dirs()

    # Step 1: ML dataset
    combined = build_ml_dataset()

    # Step 2: Optimizer input
    build_optimization_input(combined)

    # Step 3: Train occupancy
    build_train_occupancy()

    print("\n" + "=" * 60)
    print("Feature engineering complete. Files in data/processed/:")
    print("  - maintenance_ml_dataset.csv")
    print("  - optimization_input.csv")
    print("  - train_occupancy.csv")
    print("=" * 60)


if __name__ == '__main__':
    main()
