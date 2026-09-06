# =============================================================================
# SYNTHETIC DATA FOR SIH PS 26027
# AI-Powered Automatic Block Planning to Maximize Asset Availability
# for Train Operations on Indian Railways
#
# This script generates 7 synthetic CSV datasets under data/raw/.
# This data does NOT represent actual Indian Railways infrastructure.
# =============================================================================

import os
import sys
import random
import datetime
import hashlib
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'seed': 42,
    'num_stations': 50,
    'num_sections': 75,
    'num_trains': 200,
    'num_assets': 1000,
    'num_engineering_tasks': 4000,
    'num_traction_tasks': 3000,
    'num_signalling_tasks': 3000,
    'history_months': 12,
    'start_date': '2025-01-01',
    'end_date': '2025-12-31',
    'num_historical_blocks': 2000,
    'good_block_ratio': 0.6,
}


def load_config(config_path):
    """Load YAML config or fall back to defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_cfg = yaml.safe_load(f)
            if file_cfg:
                cfg.update(file_cfg)
        print(f"  Loaded config from {config_path}")
    else:
        print(f"  Using default configuration (could not load {config_path})")
    return cfg


# ---------------------------------------------------------------------------
# Station names (synthetic, Indian-sounding â€“ NOT real station names)
# ---------------------------------------------------------------------------

STATION_NAMES = [
    "Rajpur Junction", "Krishnanagar", "Devgarh", "Sundarpur", "Rameshwar",
    "Bhimtal", "Sultanpur Road", "Vikramgarh", "Chandrapur", "Shivpuri",
    "Kashipur", "Madhuban", "Dharmapur", "Gopalganj", "Kamalpur",
    "Narayanpur", "Bharatgarh", "Lalgarh", "Sitapur City", "Haripur",
    "Anandpur Sahib", "Kalyanpur", "Vijaypur", "Ratangarh", "Shyamnagar",
    "Durgapur East", "Pratapgarh", "Mohanpur", "Bhavnagar Term", "Ramanagar",
    "Kishangarh", "Jagatpur", "Nandigram Halt", "Tulsipur", "Govindpur",
    "Keshavpur", "Dinapur Cantonment", "Amritpur", "Fatehpur Sikri", "Indrapur",
    # Branch line stations (41-50)
    "Jagdishpur Jn", "Kailashpur", "Lakshmipur", "Manikpur Jn",
    "Niranjanpur", "Omkareshwar Road", "Prakashpur", "Qadirganj",
    "Raghunathpur", "Srinagar Town",
]


# ---------------------------------------------------------------------------
# 1. Railway Network
# ---------------------------------------------------------------------------

def generate_network(cfg):
    """Generate railway_network.csv with structured trunk + branch topology."""
    print("[1/7] Generating railway network â€¦")

    stations = []
    for i in range(1, cfg['num_stations'] + 1):
        stations.append({
            'station_id': f'ST{i:03d}',
            'station_name': STATION_NAMES[i - 1],
        })

    sections = []
    section_idx = 0

    # --- Main trunk: ST001 â†’ ST002 â†’ â€¦ â†’ ST040 (39 sections) ---
    for i in range(1, 40):
        section_idx += 1
        n_tracks = random.choices([1, 2, 3, 4], weights=[10, 50, 25, 15])[0]
        track_type_map = {1: "Single", 2: "Double", 3: "Triple", 4: "Quadruple"}
        dist = round(random.uniform(12, 42), 1)
        sections.append({
            'section_id': f'SEC{section_idx:03d}',
            'section_name': f"{STATION_NAMES[i-1]} - {STATION_NAMES[i]}",
            'source_station_id': f'ST{i:03d}',
            'destination_station_id': f'ST{i+1:03d}',
            'source_station': STATION_NAMES[i - 1],
            'destination_station': STATION_NAMES[i],
            'distance_km': dist,
            'track_type': track_type_map[n_tracks],
            'number_of_tracks': n_tracks,
            'electrified': random.random() < 0.85,
            'maximum_speed_kmph': random.choice([100, 110, 120, 130, 140, 160]),
            'section_capacity': 15 + n_tracks * 12 + random.randint(0, 8),
            'asset_density': round(random.uniform(1.5, 4.5), 2),
        })

    # --- Branch 1: ST010 â†’ ST041 â†’ ST042 â†’ ST043 (3 sections) ---
    branch1_pairs = [(10, 41), (41, 42), (42, 43)]
    for src, dst in branch1_pairs:
        section_idx += 1
        n_tracks = random.choices([1, 2], weights=[60, 40])[0]
        sections.append(_make_section(section_idx, src, dst, n_tracks))

    # --- Branch 2: ST020 â†’ ST044 â†’ ST045 â†’ ST046 â†’ ST047 (4 sections) ---
    branch2_pairs = [(20, 44), (44, 45), (45, 46), (46, 47)]
    for src, dst in branch2_pairs:
        section_idx += 1
        n_tracks = random.choices([1, 2], weights=[50, 50])[0]
        sections.append(_make_section(section_idx, src, dst, n_tracks))

    # --- Branch 3: ST030 â†’ ST048 â†’ ST049 â†’ ST050 (3 sections) ---
    branch3_pairs = [(30, 48), (48, 49), (49, 50)]
    for src, dst in branch3_pairs:
        section_idx += 1
        n_tracks = random.choices([1, 2], weights=[70, 30])[0]
        sections.append(_make_section(section_idx, src, dst, n_tracks))

    # --- Additional cross-link / parallel sections to reach 75 ---
    extra_pairs = [
        (5, 15), (15, 25), (25, 35), (1, 10), (10, 20), (20, 30),
        (30, 40), (35, 40), (5, 41), (43, 15), (47, 30), (50, 35),
        (2, 8), (8, 14), (14, 22), (22, 28), (28, 34), (34, 39),
        (3, 12), (12, 18), (18, 26), (26, 33), (33, 38),
        (7, 17), (17, 27),
        (40, 44),
    ]
    for src, dst in extra_pairs:
        if section_idx >= 75:
            break
        section_idx += 1
        n_tracks = random.choices([1, 2, 3], weights=[30, 50, 20])[0]
        sections.append(_make_section(section_idx, src, dst, n_tracks))

    df_sections = pd.DataFrame(sections)

    # --- Assign assets to sections ---
    assets = []
    all_section_ids = df_sections['section_id'].tolist()
    for i in range(1, cfg['num_assets'] + 1):
        if i <= 400:
            dept = "Engineering"
        elif i <= 700:
            dept = "Traction"
        else:
            dept = "Signalling"
        # Distribute assets roughly proportional to section length (proxy: idx)
        sec = all_section_ids[(i - 1) % len(all_section_ids)]
        assets.append({
            'asset_id': f'AST{i:04d}',
            'department': dept,
            'assigned_section': sec,
        })

    # Build adjacency for timetable routing
    adjacency = {}
    for _, row in df_sections.iterrows():
        src = row['source_station_id']
        dst = row['destination_station_id']
        adjacency.setdefault(src, []).append((dst, row['section_id']))
        adjacency.setdefault(dst, []).append((src, row['section_id']))  # bidirectional

    print(f"    â†’ {len(df_sections)} sections, {len(stations)} stations, {len(assets)} assets")
    return df_sections, stations, assets, adjacency


def _make_section(idx, src_num, dst_num, n_tracks):
    track_type_map = {1: "Single", 2: "Double", 3: "Triple", 4: "Quadruple"}
    dist = round(random.uniform(8, 38), 1)
    return {
        'section_id': f'SEC{idx:03d}',
        'section_name': f"{STATION_NAMES[src_num - 1]} - {STATION_NAMES[dst_num - 1]}",
        'source_station_id': f'ST{src_num:03d}',
        'destination_station_id': f'ST{dst_num:03d}',
        'source_station': STATION_NAMES[src_num - 1],
        'destination_station': STATION_NAMES[dst_num - 1],
        'distance_km': dist,
        'track_type': track_type_map[n_tracks],
        'number_of_tracks': n_tracks,
        'electrified': random.random() < 0.75,
        'maximum_speed_kmph': random.choice([80, 100, 110, 120]),
        'section_capacity': 12 + n_tracks * 10 + random.randint(0, 5),
        'asset_density': round(random.uniform(1.0, 3.5), 2),
    }


# ---------------------------------------------------------------------------
# 2. Railway Timetable
# ---------------------------------------------------------------------------

TRAIN_TYPE_CONFIG = {
    'Superfast':  {'priority': 1, 'speed_range': (100, 140), 'dwell': (2, 5)},
    'Express':    {'priority': 2, 'speed_range': (80, 120),  'dwell': (3, 8)},
    'Passenger':  {'priority': 3, 'speed_range': (50, 80),   'dwell': (5, 15)},
    'EMU':        {'priority': 3, 'speed_range': (60, 90),   'dwell': (2, 4)},
    'MEMU':       {'priority': 3, 'speed_range': (50, 80),   'dwell': (3, 5)},
    'Freight':    {'priority': 5, 'speed_range': (40, 60),   'dwell': (5, 20)},
    'Special':    {'priority': 2, 'speed_range': (60, 100),  'dwell': (3, 10)},
}


def _build_route(adjacency, start_station, length, direction, visited=None):
    """Build a route of `length` stations by BFS-like walk through adjacency."""
    route = [start_station]
    if visited is None:
        visited = {start_station}
    current = start_station
    for _ in range(length - 1):
        neighbors = adjacency.get(current, [])
        # Filter to unvisited
        candidates = [(n, s) for n, s in neighbors if n not in visited]
        if not candidates:
            break
        nxt, sec = random.choice(candidates)
        route.append(nxt)
        visited.add(nxt)
        current = nxt
    return route


def generate_timetable(cfg, df_sections, adjacency):
    """Generate railway_timetable.csv with realistic train movements."""
    print("[2/7] Generating timetable â€¦")

    num_trains = cfg['num_trains']
    # Use ~30 representative dates
    start = datetime.date.fromisoformat(cfg['start_date'])
    end = datetime.date.fromisoformat(cfg['end_date'])
    all_dates = pd.date_range(start, end, freq='12D').date.tolist()

    # Build section lookup for fast matching
    sec_lookup = {}
    for _, row in df_sections.iterrows():
        sec_lookup[(row['source_station_id'], row['destination_station_id'])] = row['section_id']
        sec_lookup[(row['destination_station_id'], row['source_station_id'])] = row['section_id']

    # Pre-generate train definitions
    train_defs = []
    type_weights = [0.20, 0.15, 0.20, 0.10, 0.10, 0.20, 0.05]
    type_names = list(TRAIN_TYPE_CONFIG.keys())
    peak_hours = [7, 8, 9, 10, 16, 17, 18, 19]

    for t in range(1, num_trains + 1):
        t_type = random.choices(type_names, weights=type_weights)[0]
        tcfg = TRAIN_TYPE_CONFIG[t_type]
        direction = "UP" if t % 2 == 0 else "DOWN"

        # Route: choose a start station and walk 5-20 stations
        route_len = random.randint(5, min(20, 40))
        if direction == "UP":
            start_st = f'ST{random.randint(1, 30):03d}'
        else:
            start_st = f'ST{random.randint(10, 40):03d}'
        route = _build_route(adjacency, start_st, route_len, direction)

        # Base departure hour â€“ bias toward peak
        if random.random() < 0.45:
            base_hour = random.choice(peak_hours)
        else:
            base_hour = random.randint(0, 23)

        train_defs.append({
            'train_id': f'TRN{t:03d}',
            'train_number': 10000 + t,
            'train_type': t_type,
            'priority_class': tcfg['priority'],
            'speed_range': tcfg['speed_range'],
            'dwell_range': tcfg['dwell'],
            'direction': direction,
            'route': route,
            'base_hour': base_hour,
        })

    # Generate records
    records = []
    for svc_date in all_dates:
        for tdef in train_defs:
            route = tdef['route']
            if len(route) < 2:
                continue
            base_min = tdef['base_hour'] * 60 + random.randint(0, 59)
            current_min = base_min

            for seq, station in enumerate(route):
                arr_min = current_min
                dwell = random.randint(*tdef['dwell_range']) if seq < len(route) - 1 else 0
                dep_min = arr_min + dwell if seq > 0 else arr_min
                if seq == 0:
                    arr_min = dep_min  # first station: arr == dep
                if seq == len(route) - 1:
                    dep_min = arr_min  # last station: dep == arr

                # Section to next station
                sec_id = ""
                if seq < len(route) - 1:
                    key = (station, route[seq + 1])
                    sec_id = sec_lookup.get(key, "")

                speed = random.randint(*tdef['speed_range'])

                records.append({
                    'train_id': tdef['train_id'],
                    'train_number': tdef['train_number'],
                    'train_type': tdef['train_type'],
                    'service_date': svc_date.isoformat(),
                    'station_id': station,
                    'station_sequence': seq + 1,
                    'arrival_time': f"{(arr_min // 60) % 24:02d}:{arr_min % 60:02d}",
                    'departure_time': f"{(dep_min // 60) % 24:02d}:{dep_min % 60:02d}",
                    'section_id': sec_id,
                    'direction': tdef['direction'],
                    'scheduled_speed': speed,
                    'priority_class': tdef['priority_class'],
                })

                # Travel to next station
                if seq < len(route) - 1:
                    key = (station, route[seq + 1])
                    dist = 25.0  # default
                    for _, r in df_sections.iterrows():
                        if (r['source_station_id'], r['destination_station_id']) == key or \
                           (r['destination_station_id'], r['source_station_id']) == key:
                            dist = r['distance_km']
                            break
                    travel_min = max(10, int(dist / (speed / 60)))
                    current_min = dep_min + travel_min

    df = pd.DataFrame(records)
    print(f"    â†’ {len(df)} timetable records across {len(all_dates)} service dates")
    return df


# ---------------------------------------------------------------------------
# 3-5. Maintenance datasets
# ---------------------------------------------------------------------------

ENGINEERING_DEFECTS = {
    "Rail Inspection":            ["Rail crack", "Surface defect", "Gauge deviation", "Worn rail head", "Corrugation"],
    "Rail Replacement":           ["Broken rail", "Severely worn rail", "End-post fracture", "Rail end batter"],
    "Track Geometry Correction":  ["Alignment defect", "Cross-level error", "Twist fault", "Gauge irregularity"],
    "Sleeper Replacement":        ["Cracked sleeper", "Decayed wooden sleeper", "Broken concrete sleeper", "Missing sleeper"],
    "Ballast Maintenance":        ["Fouled ballast", "Insufficient ballast", "Vegetation growth", "Drainage blockage"],
    "Turnout Maintenance":        ["Switch tongue wear", "Crossing nose defect", "Check rail gap", "Point mechanism fault"],
    "Welding":                    ["Weld crack", "Misaligned weld", "Weld depression", "Thermit weld defect"],
    "Track Alignment":            ["Lateral misalignment", "Vertical misalignment", "Cant deficiency", "Buckling risk"],
}

TRACTION_ASSET_TYPES = ["OHE", "Pantograph interface", "Insulator", "Catenary",
                        "Transformer", "Feeder", "Sectioning equipment", "Electrical switchgear"]
TRACTION_DEFECTS = {
    "OHE Inspection":           ["Contact wire wear", "Dropper defect", "Height deviation", "Stagger fault"],
    "Insulator Replacement":    ["Flashover damage", "Cracked insulator", "Pollution deposit", "Mechanical failure"],
    "Catenary Maintenance":     ["Catenary sag", "Tension deviation", "Splice joint fault", "Registration arm damage"],
    "Transformer Servicing":    ["Oil leak", "Bushing defect", "Winding fault", "Cooling system failure"],
    "Feeder Repair":            ["Cable damage", "Joint failure", "Overheating", "Insulation breakdown"],
    "Sectioning Post Maintenance": ["Interrupter fault", "Isolator defect", "Earthing issue", "Jumper damage"],
    "Switchgear Maintenance":   ["Circuit breaker fault", "Contactor wear", "Relay malfunction", "Arc damage"],
    "OHE Mast Repair":          ["Foundation settlement", "Corrosion", "Bracket damage", "Guy wire slack"],
}

SIGNALLING_ASSET_TYPES = ["Signal", "Point Machine", "Track Circuit", "Axle Counter",
                          "Interlocking", "Relay", "Communication Equipment", "Level Crossing Equipment"]
SIGNALLING_DEFECTS = {
    "Signal Inspection":           ["Lamp failure", "Aspect error", "Lens damage", "Signal post tilt"],
    "Relay Replacement":           ["Contact oxidation", "Coil failure", "Timing drift", "Mechanical wear"],
    "Track Circuit Maintenance":   ["Insulation failure", "Bonding defect", "Receiver fault", "Feed-end issue"],
    "Point Machine Servicing":     ["Motor fault", "Clutch slip", "Detection rod issue", "Lock bar defect"],
    "Interlocking Test":           ["Logic error", "Route locking fault", "Signal interlocking mismatch", "Overlap fault"],
    "Axle Counter Calibration":    ["Sensor misalignment", "Count error", "Reset failure", "Cable fault"],
    "Communication Equipment Repair": ["Fiber break", "Radio fault", "Control phone failure", "OFC splice loss"],
    "Level Crossing Maintenance":  ["Boom barrier fault", "Bell failure", "Gate motor issue", "Lighting defect"],
}


def _severity_to_criticality(severity):
    if severity == 1:
        return "CRITICAL"
    elif severity == 2:
        return "HIGH"
    elif severity == 3:
        return "MEDIUM"
    else:
        return "LOW"


def _severity_to_safety(severity):
    if severity <= 2:
        return "HIGH"
    elif severity == 3:
        return "MEDIUM"
    else:
        return "LOW"


def _due_date_gap(severity):
    """Days between reported_date and due_date â€” shorter for higher severity."""
    if severity == 1:
        return random.randint(3, 7)
    elif severity == 2:
        return random.randint(7, 15)
    elif severity == 3:
        return random.randint(14, 30)
    elif severity == 4:
        return random.randint(25, 60)
    else:
        return random.randint(40, 90)


def generate_maintenance(cfg, assets, dept, count, prefix, task_defect_map,
                         asset_type_list=None, block_field=None):
    """Generate a maintenance dataset for one department."""
    dept_label = dept.capitalize()
    step_map = {'Engineering': 3, 'Traction': 4, 'Signalling': 5}
    step_num = step_map.get(dept, '?')
    print(f"[{step_num}/7] Generating {dept_label} maintenance â€¦")

    dept_assets = [a for a in assets if a['department'] == dept]
    if not dept_assets:
        raise ValueError(f"No assets found for department {dept}")

    # Assign fixed asset_type per asset (if applicable)
    asset_type_map = {}
    if asset_type_list:
        for a in dept_assets:
            idx = int(a['asset_id'].replace('AST', ''))
            asset_type_map[a['asset_id']] = asset_type_list[idx % len(asset_type_list)]

    task_types = list(task_defect_map.keys())
    start_date = datetime.date.fromisoformat(cfg['start_date'])
    end_date = datetime.date.fromisoformat(cfg['end_date'])
    date_range_days = (end_date - start_date).days

    severity_weights = [0.10, 0.20, 0.35, 0.25, 0.10]  # 1-5
    status_options = ['Pending', 'Scheduled', 'In Progress', 'Completed', 'Cancelled']
    status_weights = [0.25, 0.20, 0.10, 0.35, 0.10]

    records = []
    for i in range(1, count + 1):
        asset = random.choice(dept_assets)
        task_type = random.choice(task_types)
        defect_options = task_defect_map[task_type]
        defect = random.choice(defect_options)

        severity = random.choices([1, 2, 3, 4, 5], weights=severity_weights)[0]
        criticality = _severity_to_criticality(severity)
        safety = _severity_to_safety(severity)

        # Dates
        reported = start_date + datetime.timedelta(days=random.randint(0, date_range_days - 10))
        inspection = reported - datetime.timedelta(days=random.randint(0, 3))
        due = reported + datetime.timedelta(days=_due_date_gap(severity))

        rec = {
            'task_id': f'{prefix}{i:04d}',
            'asset_id': asset['asset_id'],
            'section_id': asset['assigned_section'],
            'task_type': task_type,
            'defect_type': defect,
            'severity': severity,
            'criticality': criticality,
            'inspection_date': inspection.isoformat(),
            'reported_date': reported.isoformat(),
            'due_date': due.isoformat(),
            'estimated_duration_hours': round(random.uniform(1.0, 12.0), 1),
            'required_team_size': random.randint(2, 15),
            'safety_impact': safety,
            'status': random.choices(status_options, weights=status_weights)[0],
        }

        if asset_type_list:
            rec['asset_type'] = asset_type_map[asset['asset_id']]

        if block_field == 'power_block_required':
            # Higher probability for OHE, Catenary, Feeder work
            high_power = rec.get('asset_type', '') in ('OHE', 'Catenary', 'Feeder', 'Sectioning equipment')
            rec['power_block_required'] = random.random() < (0.90 if high_power else 0.40)
        elif block_field == 'signal_block_required':
            high_signal = rec.get('asset_type', '') in ('Signal', 'Point Machine', 'Track Circuit', 'Interlocking')
            rec['signal_block_required'] = random.random() < (0.85 if high_signal else 0.35)

        records.append(rec)

    df = pd.DataFrame(records)

    # Reorder columns so asset_type and block fields are in the right place
    if asset_type_list:
        cols = list(df.columns)
        # Move asset_type after section_id
        cols.remove('asset_type')
        idx = cols.index('section_id') + 1
        cols.insert(idx, 'asset_type')
        df = df[cols]

    print(f"    â†’ {len(df)} {dept_label} maintenance tasks")
    return df


# ---------------------------------------------------------------------------
# 6. Goods Train Forecast
# ---------------------------------------------------------------------------

def generate_forecasts(cfg, df_sections):
    """Generate goods_train_forecast.csv with realistic traffic patterns."""
    print("[6/7] Generating goods train forecasts â€¦")

    # Representative dates: 1st & 15th of each month + 6 random
    dates = []
    for month in range(1, 13):
        dates.append(datetime.date(2025, month, 1))
        dates.append(datetime.date(2025, month, 15))
    for _ in range(6):
        dates.append(datetime.date(2025, random.randint(1, 12), random.randint(1, 28)))
    dates = sorted(set(dates))

    # Section importance: trunk sections (SEC001-SEC039) are busier
    section_ids = df_sections['section_id'].tolist()
    section_importance = {}
    for sid in section_ids:
        num = int(sid.replace('SEC', ''))
        if num <= 39:
            section_importance[sid] = random.uniform(0.6, 1.0)  # trunk
        else:
            section_importance[sid] = random.uniform(0.2, 0.6)  # branch/cross

    windows = [(f"{h*2:02d}:00", f"{(h*2+2) % 24:02d}:00") for h in range(12)]

    records = []
    for date in dates:
        weekday = date.weekday()
        is_weekend = weekday >= 5
        for sid in section_ids:
            importance = section_importance[sid]
            for w_start, w_end in windows:
                hour = int(w_start.split(':')[0])

                # Time-of-day factor: goods trains run more at night / early morning
                if hour in (0, 2, 4, 22):
                    time_factor = 1.3
                elif hour in (10, 12, 14):
                    time_factor = 0.7
                else:
                    time_factor = 1.0

                weekend_factor = 0.7 if is_weekend else 1.0
                base = importance * time_factor * weekend_factor

                exp_trains = max(0, int(np.random.poisson(base * 4)))
                exp_trains = min(exp_trains, 8)
                load = round(min(100.0, max(0.0, base * 50 + random.uniform(-10, 10))), 2)

                if load < 25:
                    density = "LOW"
                elif load < 50:
                    density = "MEDIUM"
                elif load < 75:
                    density = "HIGH"
                else:
                    density = "VERY_HIGH"

                records.append({
                    'forecast_date': date.isoformat(),
                    'section_id': sid,
                    'time_window_start': w_start,
                    'time_window_end': w_end,
                    'expected_goods_trains': exp_trains,
                    'expected_traffic_load': load,
                    'traffic_density': density,
                    'forecast_confidence': round(random.uniform(0.60, 0.99), 2),
                })

    df = pd.DataFrame(records)
    print(f"    â†’ {len(df)} forecast records across {len(dates)} dates")
    return df


# ---------------------------------------------------------------------------
# 7. Maintenance Block History
# ---------------------------------------------------------------------------

def generate_block_history(cfg, df_sections):
    """Generate maintenance_block_history.csv with good and poor blocks."""
    print("[7/7] Generating maintenance block history â€¦")

    num_blocks = cfg['num_historical_blocks']
    good_ratio = cfg['good_block_ratio']
    section_ids = df_sections['section_id'].tolist()
    dept_combos = [
        "Engineering", "Traction", "Signalling",
        "Engineering,Traction", "Engineering,Signalling",
        "Traction,Signalling", "Engineering,Traction,Signalling",
    ]

    records = []
    for i in range(1, num_blocks + 1):
        date = datetime.date(2025, random.randint(1, 12), random.randint(1, 28))
        is_good = random.random() < good_ratio

        # Block window
        start_hour = random.randint(0, 20)
        duration = random.randint(1, 5) if is_good else random.randint(2, 6)
        end_hour = min(start_hour + duration, 23)
        actual_duration = end_hour - start_hour

        planned = random.randint(2, 8) if is_good else random.randint(1, 10)

        if is_good:
            efficiency = round(random.uniform(0.70, 1.0), 2)
            completed = max(1, int(planned * efficiency))
            train_conflicts = random.randint(0, 4)
            delay = random.randint(0, 30)
        else:
            efficiency = round(random.uniform(0.15, 0.55), 2)
            completed = max(0, int(planned * efficiency))
            train_conflicts = random.randint(3, 15)
            delay = random.randint(20, 120)

        cancelled = max(0, planned - completed - random.randint(0, max(0, planned - completed)))

        records.append({
            'block_id': f'BLK{i:04d}',
            'block_date': date.isoformat(),
            'section_id': random.choice(section_ids),
            'block_start': f"{start_hour:02d}:00",
            'block_end': f"{end_hour:02d}:00",
            'block_duration_hours': actual_duration,
            'tasks_in_block': planned,
            'departments_involved': random.choice(dept_combos),
            'train_conflicts': train_conflicts,
            'train_delay_minutes': delay,
            'maintenance_tasks_completed': completed,
            'planned_tasks': planned,
            'cancelled_tasks': cancelled,
            'block_efficiency': efficiency,
        })

    df = pd.DataFrame(records)
    print(f"    â†’ {len(df)} historical blocks ({int(good_ratio*100)}% well-planned)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR â€” SIH PS 26027")
    print("AI-Powered Automatic Block Planning")
    print("=" * 60)

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config', 'data_config.yaml')
    out_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path)

    # Seed for reproducibility
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    print(f"  Random seed: {seed}\n")

    # 1. Network
    df_net, stations, assets, adjacency = generate_network(cfg)
    df_net.to_csv(os.path.join(out_dir, 'railway_network.csv'), index=False)

    # 2. Timetable
    df_tt = generate_timetable(cfg, df_net, adjacency)
    df_tt.to_csv(os.path.join(out_dir, 'railway_timetable.csv'), index=False)

    # 3. Engineering maintenance
    df_eng = generate_maintenance(
        cfg, assets, "Engineering", cfg.get('num_engineering_tasks', 4000),
        "ENG", ENGINEERING_DEFECTS
    )
    df_eng.to_csv(os.path.join(out_dir, 'engineering_maintenance.csv'), index=False)

    # 4. Traction maintenance
    df_trc = generate_maintenance(
        cfg, assets, "Traction", cfg.get('num_traction_tasks', 3000),
        "TRC", TRACTION_DEFECTS,
        asset_type_list=TRACTION_ASSET_TYPES,
        block_field='power_block_required'
    )
    df_trc.to_csv(os.path.join(out_dir, 'traction_maintenance.csv'), index=False)

    # 5. Signalling maintenance
    df_sig = generate_maintenance(
        cfg, assets, "Signalling", cfg.get('num_signalling_tasks', 3000),
        "SIG", SIGNALLING_DEFECTS,
        asset_type_list=SIGNALLING_ASSET_TYPES,
        block_field='signal_block_required'
    )
    df_sig.to_csv(os.path.join(out_dir, 'signalling_maintenance.csv'), index=False)

    # 6. Goods train forecast
    df_fcst = generate_forecasts(cfg, df_net)
    df_fcst.to_csv(os.path.join(out_dir, 'goods_train_forecast.csv'), index=False)

    # 7. Block history
    df_blk = generate_block_history(cfg, df_net)
    df_blk.to_csv(os.path.join(out_dir, 'maintenance_block_history.csv'), index=False)

    print("\n" + "=" * 60)
    print("All 7 datasets generated successfully in data/raw/")
    print("=" * 60)


if __name__ == '__main__':
    main()

