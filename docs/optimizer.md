# OR-Tools CP-SAT Maintenance Block Optimizer

Technical documentation for the optimization layer of SIH PS 26027 — *AI-Powered Automatic Block Planning to Maximize Asset Availability for Train Operations on Indian Railways*.

---

## 1. What OR-Tools Does

[Google OR-Tools](https://developers.google.com/optimization) is an open-source optimization toolkit. This project uses its **CP-SAT** (Constraint Programming – Satisfiability) solver, which is a state-of-the-art constraint programming solver backed by SAT (Boolean satisfiability) techniques.

OR-Tools takes a set of **decision variables**, **constraints**, and an **objective function** and finds an optimal (or near-optimal) assignment of values to the variables.

## 2. Why CP-SAT

CP-SAT is ideal for maintenance scheduling because:

- It handles **discrete time** natively (integer variables).
- It provides **interval variables** and built-in `NoOverlap` constraints for resource scheduling.
- It supports **optional tasks** (tasks that may or may not be scheduled).
- It scales well with parallelism (`num_workers`).
- It handles complex combinatorial constraints that greedy or heuristic approaches miss.
- It can produce **provably optimal** solutions or report the optimality gap.

## 3. Input Datasets

| File | Records | Description |
|------|---------|-------------|
| `data/processed/optimization_input.csv` | 4,612 | Tasks with time windows, durations, resource requirements |
| `data/processed/maintenance_predictions.csv` | 10,000 | ML-predicted priority scores and classes |
| `data/processed/train_occupancy.csv` | 26,100 | Section-level 2-hour occupancy windows |

## 4. ML → Optimizer Interface

The XGBoost model produces:

```
task_id → predicted_priority_score (float, 0–100)
task_id → predicted_priority_class (CRITICAL/HIGH/MEDIUM/LOW)
```

The optimizer joins these predictions with `optimization_input.csv` on `task_id` and uses `predicted_priority_score` as the primary scheduling priority. The original `priority_score` column (training label) is only used as a fallback if predictions are missing.

## 5. Decision Variables

For each task `t`:

| Variable | Type | Domain | Description |
|----------|------|--------|-------------|
| `scheduled[t]` | BoolVar | {0, 1} | Whether the task is scheduled |
| `start[t]` | IntVar | [earliest_start, latest_end − duration] | Start time (integer minutes from epoch) |
| `end[t]` | IntVar | [earliest_start + duration, latest_end] | End time |
| `interval[t]` | IntervalVar | Optional, fixed-size | Present only when `scheduled = 1` |

## 6. Constraints

### 6.1 Time-Window Constraints
Every scheduled task must satisfy:
- `start >= earliest_start`
- `end <= min(latest_end, deadline)`

### 6.2 Duration Constraints
- `end = start + duration` (enforced via fixed-size IntervalVar)

### 6.3 Train Occupancy Constraints
For each task requiring an infrastructure block (track, signal, or power) on section `s`:
- For each occupied 2-hour window `[occ_s, occ_e)` on section `s` within the task's feasible range:
  - `end <= occ_s  OR  start >= occ_e`
  - Linearised using an auxiliary BoolVar

### 6.4 Resource No-Overlap Constraints
Tasks requiring the **same exclusive resource** on the **same section** cannot overlap:
- `track` block tasks on SEC001 → `NoOverlap([interval_T1, interval_T2, ...])`
- `signal` block tasks on SEC001 → `NoOverlap([...])`
- `power` block tasks on SEC001 → `NoOverlap([...])`

224 resource groups are created across 75 sections.

### 6.5 Dependency Constraints (Framework)
If task B depends on task A: `end(A) <= start(B)`. The current dataset does not contain dependencies, but the framework supports them.

## 7. Train Occupancy Logic

Train occupancy data provides 2-hour windows per section per date:

```
SEC001, 2025-03-15, 08:00–10:00, occupied=True
```

This is converted to absolute minutes from epoch and stored in a lookup:
```python
{"SEC001": [(start_min_1, end_min_1), (start_min_2, end_min_2), ...]}
```

Only tasks that require an infrastructure block (track, signal, or power) are constrained against occupancy. Tasks with no block requirement can proceed regardless of train presence.

## 8. Resource Constraints

Three exclusive resource types are modelled:

| Resource | Column | Count |
|----------|--------|-------|
| Track | `track_block_required` | 1,824 tasks |
| Power | `power_block_required` | 917 tasks |
| Signal | `signal_block_required` | 819 tasks |

Tasks requiring the same resource on the same section use CP-SAT `NoOverlap` intervals.

## 9. Block Grouping

After solving, scheduled tasks are grouped into **maintenance blocks**:

1. Tasks are grouped by section.
2. Within each section, tasks are sorted by start time.
3. Tasks are merged into a block if:
   - Same section
   - At least one has `can_combine = True`
   - Intervals overlap or are adjacent (gap ≤ 15 minutes)
4. The block spans from the earliest start to the latest end of its tasks.

The solver also includes **combination bonus variables** that incentivise placing compatible tasks in overlapping time windows.

## 10. Objective Function

The objective is a single **maximisation** target (minimisation components are negated):

```
MAXIMISE:
  Σ (priority_int × PRIORITY_WEIGHT / 100 × scheduled)
+ Σ (urgency_int × DEADLINE_WEIGHT / 100 × scheduled)
+ Σ (combination_bonus × CONSOLIDATION_WEIGHT)
+ Σ (priority_int × UNSCHEDULED_PENALTY / 100 × scheduled)   [rewards scheduling]
- Σ (conflict_cost_int × TRAIN_CONFLICT_WEIGHT / 100 × scheduled)
```

All coefficients are integers. Float scores are scaled via `PRIORITY_SCALE_FACTOR = 10` (e.g., 95.2 → 952).

## 11. Solver Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| Time limit | 60 seconds | `OPT_TIME_LIMIT` |
| Workers | min(8, CPU count) | `OPT_NUM_WORKERS` |
| Random seed | 42 | `OPT_RANDOM_SEED` |

## 12. Output Files

| File | Description |
|------|-------------|
| `artifacts/optimization/optimal_schedule.csv` | Task-level schedule with block assignments |
| `artifacts/optimization/maintenance_blocks.csv` | Block-level summary with constituent tasks |
| `artifacts/optimization/unscheduled_tasks.csv` | Tasks not scheduled, with reasons |
| `artifacts/optimization/optimization_summary.json` | Solver statistics and KPIs |
| `artifacts/optimization/solver_log.txt` | Human-readable solver log |
| `artifacts/optimization/validation_report.json` | Independent validation results |

## 13. Validation

The validator independently checks (does NOT trust CP-SAT alone):

- Time-window adherence
- Duration correctness
- Deadline compliance
- Train occupancy non-overlap
- Resource non-overlap
- Block consistency (tasks in a block match section)
- Task coverage (every input task accounted for)
- No duplicate scheduling

## 14. How to Run

```bash
# Full run with defaults
python -m src.optimizer.main

# Custom solver settings
python -m src.optimizer.main --time-limit 120 --workers 8 --seed 42
```

## 15. How to Test

```bash
pytest tests/optimizer -v
```

## 16. Known Assumptions

- All tasks are independent (no dependencies in current dataset).
- Priority scores from XGBoost are the sole priority input.
- Train occupancy is static (pre-computed, not reactive).
- Resource exclusivity is per-section: two tasks on different sections never conflict.
- The solver's 60-second time limit may produce FEASIBLE (not proven OPTIMAL) solutions for the full 4,612-task dataset.

## 17. Limitations

- **Synthetic data**: All datasets are synthetic for SIH demonstration. Results do not represent actual Indian Railways operations.
- **Block grouping is post-solve**: The in-model combination bonus incentivises grouping, but the actual block formation happens after solving. This may not find the globally optimal block structure.
- **Single planning horizon**: All tasks are scheduled in a single pass. Rolling-horizon scheduling is not implemented.
- **No crew/equipment constraints**: Team sizes and equipment availability are not modelled.

## 18. Future Improvements

- Rolling-horizon scheduling for continuous operations.
- Crew and equipment resource constraints.
- Multi-objective Pareto front exploration.
- Integration with real-time train timetable feeds.
- FastAPI endpoint for on-demand re-optimisation.
- Weather-dependent scheduling constraints.
- Historical performance feedback loop.
