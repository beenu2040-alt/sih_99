# =============================================================================
# SYNTHETIC DATA FOR SIH PS 26027 -- Preprocessing Pipeline
# Cleans, normalizes, and enriches all 7 raw datasets into processed versions.
# =============================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

REFERENCE_DATE = pd.Timestamp('2025-06-15')


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def _strip_categorical(df, columns):
    """Strip whitespace and normalize casing on categorical columns."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# 1. Railway Network
# ---------------------------------------------------------------------------
def process_network():
    print("[1/7] Processing railway_network ...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'railway_network.csv'))

    df = _strip_categorical(df, ['track_type', 'section_id'])
    df['electrified'] = df['electrified'].astype(bool)

    # Derived: section length category
    df['section_length_category'] = pd.cut(
        df['distance_km'],
        bins=[0, 15, 30, np.inf],
        labels=['Short', 'Medium', 'Long']
    )

    df.to_csv(os.path.join(OUT_DIR, 'railway_network_processed.csv'), index=False)
    print(f"    --> {len(df)} records saved")
    return df


# ---------------------------------------------------------------------------
# 2. Railway Timetable
# ---------------------------------------------------------------------------
def process_timetable(valid_sections, valid_stations):
    print("[2/7] Processing railway_timetable ...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'railway_timetable.csv'))

    df = _strip_categorical(df, ['train_type', 'direction', 'station_id', 'section_id', 'train_id'])

    # Ensure time format
    df['arrival_time'] = df['arrival_time'].astype(str).str.strip()
    df['departure_time'] = df['departure_time'].astype(str).str.strip()

    # Remove rows with invalid foreign keys
    before = len(df)
    df = df[df['station_id'].isin(valid_stations)]
    # section_id can be empty string for last stop
    mask_valid_sec = df['section_id'].isin(valid_sections) | (df['section_id'].astype(str).str.strip() == '')
    df = df[mask_valid_sec]
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with invalid foreign keys")

    # Derived: dwell time (minutes)
    arr = pd.to_datetime(df['arrival_time'], format='%H:%M', errors='coerce')
    dep = pd.to_datetime(df['departure_time'], format='%H:%M', errors='coerce')
    df['dwell_time_minutes'] = ((dep - arr).dt.total_seconds() / 60).fillna(0).clip(lower=0).astype(int)

    # Derived: is_peak_hour
    hour = pd.to_datetime(df['departure_time'], format='%H:%M', errors='coerce').dt.hour
    df['is_peak_hour'] = hour.isin([7, 8, 9, 10, 16, 17, 18, 19])

    df.to_csv(os.path.join(OUT_DIR, 'railway_timetable_processed.csv'), index=False)
    print(f"    --> {len(df)} records saved")
    return df


# ---------------------------------------------------------------------------
# 3-5. Maintenance datasets
# ---------------------------------------------------------------------------
def process_maintenance(filename, label, valid_sections):
    print(f"[{label}] Processing {filename} ...")
    df = pd.read_csv(os.path.join(RAW_DIR, filename))

    cat_cols = ['criticality', 'safety_impact', 'status', 'task_type', 'defect_type']
    if 'asset_type' in df.columns:
        cat_cols.append('asset_type')
    df = _strip_categorical(df, cat_cols)

    # Ensure date format
    for dcol in ['inspection_date', 'reported_date', 'due_date']:
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce').dt.strftime('%Y-%m-%d')

    # Remove invalid FK
    before = len(df)
    df = df[df['section_id'].isin(valid_sections)]
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with invalid section_id")

    # Derived: days_until_due (from reported_date)
    reported = pd.to_datetime(df['reported_date'], errors='coerce')
    due = pd.to_datetime(df['due_date'], errors='coerce')
    df['days_until_due'] = (due - reported).dt.days.fillna(0).astype(int)

    # Derived: urgency_category
    df['urgency_category'] = pd.cut(
        df['days_until_due'],
        bins=[-np.inf, 7, 30, np.inf],
        labels=['Urgent', 'Normal', 'Low']
    )

    out_name = filename.replace('.csv', '_processed.csv')
    df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    print(f"    --> {len(df)} records saved")
    return df


# ---------------------------------------------------------------------------
# 6. Goods Train Forecast
# ---------------------------------------------------------------------------
def process_forecast(valid_sections):
    print("[6/7] Processing goods_train_forecast ...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'goods_train_forecast.csv'))

    df = _strip_categorical(df, ['traffic_density', 'section_id'])
    df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Remove invalid FK
    before = len(df)
    df = df[df['section_id'].isin(valid_sections)]
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with invalid section_id")

    # Derived: is_peak_window
    start_hour = df['time_window_start'].str.split(':').str[0].astype(int)
    df['is_peak_window'] = start_hour.isin([6, 8, 10, 16, 18])

    df.to_csv(os.path.join(OUT_DIR, 'goods_train_forecast_processed.csv'), index=False)
    print(f"    --> {len(df)} records saved")
    return df


# ---------------------------------------------------------------------------
# 7. Maintenance Block History
# ---------------------------------------------------------------------------
def process_blocks(valid_sections):
    print("[7/7] Processing maintenance_block_history ...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'maintenance_block_history.csv'))

    df = _strip_categorical(df, ['departments_involved', 'section_id'])
    df['block_date'] = pd.to_datetime(df['block_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Remove invalid FK
    before = len(df)
    df = df[df['section_id'].isin(valid_sections)]
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with invalid section_id")

    # Derived: efficiency_category
    df['efficiency_category'] = pd.cut(
        df['block_efficiency'],
        bins=[-np.inf, 0.4, 0.6, 0.8, np.inf],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )

    df.to_csv(os.path.join(OUT_DIR, 'maintenance_block_history_processed.csv'), index=False)
    print(f"    --> {len(df)} records saved")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE -- SIH PS 26027")
    print("=" * 60 + "\n")

    ensure_dirs()

    # 1. Network first (needed for FK validation)
    df_net = process_network()
    valid_sections = set(df_net['section_id'].unique())
    valid_stations = set(df_net['source_station_id'].unique()) | set(df_net['destination_station_id'].unique())

    # 2. Timetable
    process_timetable(valid_sections, valid_stations)

    # 3-5. Maintenance
    process_maintenance('engineering_maintenance.csv', '3/7', valid_sections)
    process_maintenance('traction_maintenance.csv', '4/7', valid_sections)
    process_maintenance('signalling_maintenance.csv', '5/7', valid_sections)

    # 6. Forecast
    process_forecast(valid_sections)

    # 7. Blocks
    process_blocks(valid_sections)

    print("\n" + "=" * 60)
    print("All 7 processed datasets saved to data/processed/")
    print("=" * 60)


if __name__ == '__main__':
    main()
