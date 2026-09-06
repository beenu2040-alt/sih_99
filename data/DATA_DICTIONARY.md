# Data Dictionary for SIH PS 26027

This document details the schema for all datasets (raw and processed) generated for the AI-Powered Automatic Block Planning project.

## Raw Datasets (`data/raw/`)

### 1. railway_network.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| section_id | string | Unique section identifier | SEC001-SEC075 | SEC015 | Referenced by all other datasets |
| section_name | string | Human-readable section name | - | Rajpur Jn - Krishnanagar | Derived from station names |
| source_station_id | string | Origin station ID | ST001-ST050 | ST001 | References station_id |
| destination_station_id | string | Destination station ID | ST001-ST050 | ST002 | References station_id |
| source_station | string | Origin station name | - | Rajpur Junction | - |
| destination_station | string | Destination station name | - | Krishnanagar | - |
| distance_km | float | Section length in km | 8.0-45.0 | 23.5 | - |
| track_type | string | Track classification | Single/Double/Triple/Quadruple | Double | - |
| number_of_tracks | int | Number of parallel tracks | 1-4 | 2 | - |
| electrified | boolean | Whether section has electric traction | True/False | True | Affects traction maintenance |
| maximum_speed_kmph | int | Speed limit | 80-160 | 130 | Affects timetable speed |
| section_capacity | int | Max trains per day | 20-60 | 40 | Affects traffic density |
| asset_density | float | Assets per kilometer | 0.5-5.0 | 2.3 | - |

### 2. railway_timetable.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| train_id | string | Unique train identifier | TRN001-TRN200 | TRN150 | References train routes |
| train_number | int | 5-digit train number | - | 12301 | - |
| train_type | string | Type of train | Express/Superfast/Passenger/EMU/MEMU/Freight/Special | Superfast | - |
| service_date | date | Date of service | YYYY-MM-DD | 2024-03-15 | - |
| station_id | string | Station where this entry applies | ST001-ST050 | ST023 | References railway_network |
| station_sequence | int | Order of station in train's route | 1-N | 3 | - |
| arrival_time | time | Arrival time | HH:MM | 14:30 | - |
| departure_time | time | Departure time | HH:MM, >= arrival_time | 14:35 | - |
| section_id | string | Section between this station and next | SEC001-SEC075 | SEC045 | References railway_network |
| direction | string | Direction of travel | UP/DOWN | UP | - |
| scheduled_speed | float | Scheduled speed in km/h | - | 85.5 | - |
| priority_class | int | Priority class | 1(highest)-5(lowest) | 2 | - |

### 3. engineering_maintenance.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| task_id | string | Unique task identifier | ENG0001-ENG4000 | ENG0150 | - |
| asset_id | string | Unique asset identifier | AST0001-AST0400 | AST0120 | References engineering assets |
| section_id | string | Section where maintenance occurs | SEC001-SEC075 | SEC045 | References railway_network |
| task_type | string | Type of maintenance task | Rail Inspection/Rail Replacement/Track Geometry Correction/Sleeper Replacement/Ballast Maintenance/Turnout Maintenance/Welding/Track Alignment | Rail Inspection | - |
| defect_type | string | Specific defect description | - | Broken rail | - |
| severity | int | Severity level | 1(critical)-5(minor) | 1 | - |
| criticality | string | Criticality level | CRITICAL/HIGH/MEDIUM/LOW | CRITICAL | - |
| inspection_date | date | Date of inspection | YYYY-MM-DD | 2024-03-01 | - |
| reported_date | date | Date reported | YYYY-MM-DD, >= inspection_date | 2024-03-02 | - |
| due_date | date | Due date for maintenance | YYYY-MM-DD, >= reported_date | 2024-03-10 | - |
| estimated_duration_hours | float | Estimated duration | 1.0-12.0 | 4.5 | - |
| required_team_size | int | Number of personnel required | 2-15 | 5 | - |
| safety_impact | string | Impact on safety | HIGH/MEDIUM/LOW | HIGH | - |
| status | string | Task status | Pending/Scheduled/In Progress/Completed/Cancelled | Pending | - |

### 4. traction_maintenance.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| task_id | string | Unique task identifier | TRC0001-TRC3000 | TRC0500 | - |
| asset_id | string | Unique asset identifier | AST0401-AST0700 | AST0500 | References traction assets |
| section_id | string | Section where maintenance occurs | SEC001-SEC075 | SEC045 | References railway_network |
| task_type | string | Type of maintenance task | - | - | - |
| defect_type | string | Specific defect description | - | Broken pantograph | - |
| severity | int | Severity level | 1(critical)-5(minor) | 2 | - |
| criticality | string | Criticality level | CRITICAL/HIGH/MEDIUM/LOW | HIGH | - |
| inspection_date | date | Date of inspection | YYYY-MM-DD | 2024-03-01 | - |
| reported_date | date | Date reported | YYYY-MM-DD, >= inspection_date | 2024-03-02 | - |
| due_date | date | Due date for maintenance | YYYY-MM-DD, >= reported_date | 2024-03-10 | - |
| estimated_duration_hours | float | Estimated duration | 1.0-12.0 | 3.0 | - |
| required_team_size | int | Number of personnel required | 2-15 | 4 | - |
| safety_impact | string | Impact on safety | HIGH/MEDIUM/LOW | HIGH | - |
| status | string | Task status | Pending/Scheduled/In Progress/Completed/Cancelled | Pending | - |
| asset_type | string | Type of traction asset | OHE/Pantograph interface/Insulator/Catenary/Transformer/Feeder/Sectioning equipment/Electrical switchgear | OHE | - |
| power_block_required | boolean | Requires power block | True/False | True | - |

### 5. signalling_maintenance.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| task_id | string | Unique task identifier | SIG0001-SIG3000 | SIG1200 | - |
| asset_id | string | Unique asset identifier | AST0701-AST1000 | AST0800 | References signalling assets |
| section_id | string | Section where maintenance occurs | SEC001-SEC075 | SEC045 | References railway_network |
| task_type | string | Type of maintenance task | - | - | - |
| defect_type | string | Specific defect description | - | Signal failure | - |
| severity | int | Severity level | 1(critical)-5(minor) | 1 | - |
| criticality | string | Criticality level | CRITICAL/HIGH/MEDIUM/LOW | CRITICAL | - |
| inspection_date | date | Date of inspection | YYYY-MM-DD | 2024-03-01 | - |
| reported_date | date | Date reported | YYYY-MM-DD, >= inspection_date | 2024-03-02 | - |
| due_date | date | Due date for maintenance | YYYY-MM-DD, >= reported_date | 2024-03-10 | - |
| estimated_duration_hours | float | Estimated duration | 1.0-12.0 | 2.5 | - |
| required_team_size | int | Number of personnel required | 2-15 | 3 | - |
| safety_impact | string | Impact on safety | HIGH/MEDIUM/LOW | HIGH | - |
| status | string | Task status | Pending/Scheduled/In Progress/Completed/Cancelled | Pending | - |
| asset_type | string | Type of signalling asset | Signal/Point Machine/Track Circuit/Axle Counter/Interlocking/Relay/Communication Equipment/Level Crossing Equipment | Signal | - |
| signal_block_required | boolean | Requires signal block | True/False | True | - |

### 6. goods_train_forecast.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| forecast_date | date | Date of forecast | YYYY-MM-DD | 2024-03-15 | - |
| section_id | string | Section for forecast | SEC001-SEC075 | SEC045 | References railway_network |
| time_window_start | time | Start of time window | HH:MM (2-hour windows) | 10:00 | - |
| time_window_end | time | End of time window | HH:MM | 12:00 | - |
| expected_goods_trains | int | Expected number of trains | 0-8 | 3 | - |
| expected_traffic_load | float | Expected load percentage | 0-100 | 75.5 | - |
| traffic_density | string | Traffic density level | LOW/MEDIUM/HIGH/VERY_HIGH | HIGH | - |
| forecast_confidence | float | Confidence score | 0.6-0.99 | 0.85 | - |

### 7. maintenance_block_history.csv
| Column | Type | Description | Values | Example | Relationships |
|--------|------|-------------|--------|---------|---------------|
| block_id | string | Unique block identifier | BLK0001-BLK2000 | BLK0050 | - |
| block_date | date | Date of block | YYYY-MM-DD | 2024-03-15 | - |
| section_id | string | Section of block | SEC001-SEC075 | SEC045 | References railway_network |
| block_start | time | Start time of block | HH:MM | 14:00 | - |
| block_end | time | End time of block | HH:MM | 16:30 | - |
| block_duration_hours | float | Duration of block | 1.0-6.0 | 2.5 | - |
| tasks_in_block | int | Number of tasks | 1-10 | 4 | - |
| departments_involved | string | Departments involved | Comma-separated: Engineering/Traction/Signalling | Engineering,Signalling | - |
| train_conflicts | int | Number of train conflicts | 0-15 | 2 | - |
| train_delay_minutes | int | Total delay in minutes | 0-120 | 45 | - |
| maintenance_tasks_completed | int | Completed tasks | <= planned_tasks | 3 | - |
| planned_tasks | int | Planned tasks | 1-10 | 4 | - |
| cancelled_tasks | int | Cancelled tasks | 0 to planned-completed | 1 | - |
| block_efficiency | float | Efficiency metric | 0.0-1.0 | 0.75 | - |

## Processed Datasets (`data/processed/`)

Processed datasets contain the same columns as their raw counterparts, with the addition of the following computed fields:

### railway_network_processed.csv
* **section_length_category** (string): Categorization of section length (Short/Medium/Long).

### railway_timetable_processed.csv
* **dwell_time_minutes** (int): Time spent at the station.
* **is_peak_hour** (boolean): Indicates if the time falls during peak hours.

### *_maintenance_processed.csv (All maintenance datasets)
* **days_until_due** (int): Number of days until the maintenance task is due.
* **urgency_category** (string): Categorization of urgency based on days_until_due and criticality.

### goods_train_forecast_processed.csv
* **is_peak_window** (boolean): Indicates if the 2-hour window corresponds to a peak traffic period.

### maintenance_block_history_processed.csv
* **efficiency_category** (string): Categorization of block efficiency based on block_efficiency score.

## ML & Optimizer Datasets (`data/processed/`)

### maintenance_ml_dataset.csv
| Column | Description |
|--------|-------------|
| task_id | Unique task identifier |
| department | Engineering, Traction, or Signalling |
| section_id | Section identifier |
| asset_id | Asset identifier |
| asset_type | Type of asset |
| task_type | Type of maintenance task |
| defect_type | Defect description |
| severity | Severity level (1-5) |
| criticality | Criticality level |
| days_to_deadline | Days remaining until due date |
| asset_age | Age of the asset in years |
| failure_history | Number of past failures |
| maintenance_frequency | Frequency of maintenance actions |
| estimated_duration_hours | Estimated duration in hours |
| required_team_size | Number of personnel required |
| safety_impact_encoded | Numerically encoded safety impact |
| train_traffic_density | Density of train traffic on the section |
| expected_train_conflicts | Number of anticipated train conflicts |
| historical_failure_rate | Rate of historical failures |
| priority_score | Calculated priority score (Target variable) |
| priority_class | Target class |
| power_block_required | Boolean |
| signal_block_required | Boolean |
| track_block_required | Boolean |
| status | Current status |

### optimization_input.csv
| Column | Description |
|--------|-------------|
| task_id | Unique task identifier |
| section_id | Section identifier |
| department | Department responsible |
| priority_score | Priority score from ML model |
| duration_hours | Estimated duration |
| earliest_start | Earliest possible start time |
| latest_end | Latest acceptable end time |
| deadline | Hard deadline |
| train_conflict_cost | Penalty cost for train conflicts |
| traffic_density | Density of traffic |
| power_block_required | Boolean |
| signal_block_required | Boolean |
| track_block_required | Boolean |
| can_combine | Boolean indicating if task can be combined in block |

### train_occupancy.csv
| Column | Description |
|--------|-------------|
| section_id | Section identifier |
| date | Date of occupancy |
| time_window_start | Start of time window |
| time_window_end | End of time window |
| occupied | Boolean indicating if section is occupied |
| train_count | Number of trains |
| traffic_density | Density classification |
