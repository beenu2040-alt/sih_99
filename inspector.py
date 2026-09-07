import pandas as pd
import json
import os
from datetime import datetime, timedelta

def run_inspection():
    base_dir = "c:/hackathon/sih_99"
    
    # Load files
    opt_in_df = pd.read_csv(os.path.join(base_dir, "data/processed/optimization_input.csv"))
    occ_df = pd.read_csv(os.path.join(base_dir, "data/processed/train_occupancy.csv"))
    pred_df = pd.read_csv(os.path.join(base_dir, "data/processed/maintenance_predictions.csv"))
    sched_df = pd.read_csv(os.path.join(base_dir, "artifacts/optimization/optimal_schedule.csv"))
    blocks_df = pd.read_csv(os.path.join(base_dir, "artifacts/optimization/maintenance_blocks.csv"))
    
    unsched_path = os.path.join(base_dir, "artifacts/optimization/unscheduled_tasks.csv")
    if os.path.exists(unsched_path):
        unsched_df = pd.read_csv(unsched_path)
    else:
        unsched_df = pd.DataFrame()
        
    report = {}
    violations = {
        "train_conflicts": [],
        "resource_conflicts": [],
        "deadline_violations": [],
        "duration_violations": [],
        "duplicate_assignments": []
    }
    
    # 1. Total input tasks
    report["total_input_tasks"] = len(opt_in_df)
    
    # 2. Scheduled tasks
    report["scheduled_tasks"] = len(sched_df)
    
    # 3. Unscheduled tasks
    report["unscheduled_tasks"] = len(unsched_df)
    
    # 4. Total maintenance blocks
    report["total_maintenance_blocks"] = sched_df["block_id"].nunique()
    
    # 5. Tasks per block
    block_counts = sched_df.groupby("block_id").size()
    report["tasks_per_block_avg"] = round(block_counts.mean(), 2)
    
    # 6. Highest-priority scheduled tasks (Top 5)
    top_sched = sched_df.sort_values(by="predicted_priority_score", ascending=False).head(5)
    report["highest_priority_scheduled"] = top_sched[["task_id", "predicted_priority_score"]].to_dict('records')
    
    # 7. Highest-priority unscheduled tasks (Top 5)
    if not unsched_df.empty:
        top_unsched = unsched_df.sort_values(by="predicted_priority_score", ascending=False).head(5)
        report["highest_priority_unscheduled"] = top_unsched[["task_id", "predicted_priority_score"]].to_dict('records')
    else:
        report["highest_priority_unscheduled"] = []
        
    # Validation
    # Convert dates to datetime
    sched_df['start_time'] = pd.to_datetime(sched_df['start_time'])
    sched_df['end_time'] = pd.to_datetime(sched_df['end_time'])
    opt_in_df['deadline'] = pd.to_datetime(opt_in_df['deadline'])
    opt_in_df['earliest_start'] = pd.to_datetime(opt_in_df['earliest_start'])
    
    # Merge for validations
    sched_val = pd.merge(sched_df, opt_in_df[['task_id', 'deadline', 'earliest_start']], on='task_id', how='left')
    
    # 10. Deadline violations
    for _, row in sched_val.iterrows():
        deadline_eod = row['deadline'] + timedelta(days=1) - timedelta(seconds=1)
        if pd.notna(row['deadline']) and row['end_time'] > deadline_eod:
            violations["deadline_violations"].append({
                "task_id": row["task_id"],
                "block_id": row["block_id"],
                "section_id": row["section_id"],
                "scheduled_interval": f"{row['start_time']} to {row['end_time']}",
                "conflicting_interval": f"Deadline: {row['deadline'].date()}",
                "reason": "Scheduled end time is after the deadline."
            })
            
    # 11. Duration violations
    for _, row in sched_val.iterrows():
        scheduled_duration = (row['end_time'] - row['start_time']).total_seconds() / 3600
        # Give a small tolerance for floating point
        if scheduled_duration < row['duration_hours'] - 0.01:
            violations["duration_violations"].append({
                "task_id": row["task_id"],
                "block_id": row["block_id"],
                "section_id": row["section_id"],
                "scheduled_interval": f"{row['start_time']} to {row['end_time']} ({scheduled_duration:.2f}h)",
                "conflicting_interval": f"Required: {row['duration_hours']:.2f}h",
                "reason": "Scheduled duration is less than required duration."
            })
            
    # 12. Duplicate task assignments
    task_counts = sched_df['task_id'].value_counts()
    duplicates = task_counts[task_counts > 1]
    for task_id in duplicates.index:
        dups = sched_df[sched_df['task_id'] == task_id]
        blocks = dups['block_id'].tolist()
        intervals = [f"{r['start_time']} to {r['end_time']}" for _, r in dups.iterrows()]
        violations["duplicate_assignments"].append({
            "task_id": task_id,
            "block_id": str(blocks),
            "section_id": dups.iloc[0]['section_id'],
            "scheduled_interval": str(intervals),
            "conflicting_interval": "N/A",
            "reason": "Task assigned multiple times."
        })
        
    # 13. Block consolidation statistics
    report["block_consolidation"] = {
        "blocks_with_1_task": int((block_counts == 1).sum()),
        "blocks_with_multiple_tasks": int((block_counts > 1).sum()),
        "max_tasks_in_block": int(block_counts.max())
    }
    
    # 8. Train conflicts
    occ_df['date'] = pd.to_datetime(occ_df['date']).dt.date
    occ_df['window_start_time'] = pd.to_datetime(occ_df['time_window_start'], format='%H:%M').dt.time
    occ_df['window_end_time'] = pd.to_datetime(occ_df['time_window_end'], format='%H:%M').dt.time
    
    occ_sec = occ_df[occ_df['occupied'] == True].copy()
    occ_sec['datetime_start'] = pd.to_datetime(occ_sec['date'].astype(str) + ' ' + occ_sec['time_window_start'])
    occ_sec['datetime_end'] = pd.to_datetime(occ_sec['date'].astype(str) + ' ' + occ_sec['time_window_end'])
    occ_sec.loc[occ_sec['time_window_end'] == '00:00', 'datetime_end'] += timedelta(days=1)

    for sec in sched_val['section_id'].unique():
        sec_tasks = sched_val[sched_val['section_id'] == sec]
        sec_occ = occ_sec[occ_sec['section_id'] == sec]
        
        for _, row in sec_tasks.iterrows():
            needs_block = row['power_block_required'] or row['signal_block_required'] or row['track_block_required']
            if not needs_block:
                continue

            st = row['start_time']
            et = row['end_time']
            
            # Find overlaps
            overlaps = sec_occ[(sec_occ['datetime_start'] < et) & (sec_occ['datetime_end'] > st)]
            if not overlaps.empty:
                for _, occ_row in overlaps.iterrows():
                    violations["train_conflicts"].append({
                        "task_id": row["task_id"],
                        "block_id": row["block_id"],
                        "section_id": row["section_id"],
                        "scheduled_interval": f"{st} to {et}",
                        "conflicting_interval": f"{occ_row['datetime_start']} to {occ_row['datetime_end']} (Occupied)",
                        "reason": "Task requires infrastructure block and overlaps with scheduled train."
                    })

    # 9. Resource conflicts
    for sec in sched_val['section_id'].unique():
        sec_tasks = sched_val[sched_val['section_id'] == sec]
        
        for resource in ['power_block_required', 'signal_block_required', 'track_block_required']:
            res_tasks = sec_tasks[sec_tasks[resource] == True]
            for i in range(len(res_tasks)):
                for j in range(i+1, len(res_tasks)):
                    t1 = res_tasks.iloc[i]
                    t2 = res_tasks.iloc[j]
                    
                    if max(t1['start_time'], t2['start_time']) < min(t1['end_time'], t2['end_time']):
                        res_name = resource.replace('_block_required', '')
                        violations["resource_conflicts"].append({
                            "task_id": f"{t1['task_id']}, {t2['task_id']}",
                            "block_id": f"{t1['block_id']}, {t2['block_id']}",
                            "section_id": sec,
                            "resource_name": res_name,
                            "scheduled_interval": f"{t1['start_time']} to {t1['end_time']} vs {t2['start_time']} to {t2['end_time']}",
                            "conflicting_interval": "Overlap",
                            "reason": f"Overlapping tasks on the same section sharing resources: {res_name}"
                        })
                    
    report["violations"] = violations

    # Create the report object
    with open(os.path.join(base_dir, 'artifacts/optimization/schedule_inspection_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    # Print summary
    print("=== Schedule Inspection Summary ===")
    print(f"Total Input Tasks: {report['total_input_tasks']}")
    print(f"Scheduled Tasks: {report['scheduled_tasks']}")
    print(f"Unscheduled Tasks: {report['unscheduled_tasks']}")
    print(f"Total Maintenance Blocks: {report['total_maintenance_blocks']}")
    print(f"Average Tasks per Block: {report['tasks_per_block_avg']}")
    print(f"Block Consolidation: {report['block_consolidation']}")
    print(f"Top Priority Scheduled: {', '.join([str(x['task_id']) for x in report['highest_priority_scheduled']])}")
    print(f"Top Priority Unscheduled: {', '.join([str(x['task_id']) for x in report['highest_priority_unscheduled']])}")
    print("\n=== Violations ===")
    print(f"Train Conflicts: {len(violations['train_conflicts'])}")
    print(f"Resource Conflicts: {len(violations['resource_conflicts'])}")
    print(f"Deadline Violations: {len(violations['deadline_violations'])}")
    print(f"Duration Violations: {len(violations['duration_violations'])}")
    print(f"Duplicate Assignments: {len(violations['duplicate_assignments'])}")
    
    for v_type, v_list in violations.items():
        if v_list:
            print(f"\n{v_type.replace('_', ' ').title()}:")
            for v in v_list:
                resource_info = f" ({v.get('resource_name')})" if 'resource_name' in v else ""
                print(f"  Task {v['task_id']} in block {v['block_id']} at {v['section_id']}{resource_info}: {v['scheduled_interval']} conflicts with {v['conflicting_interval']}. Reason: {v['reason']}")

run_inspection()
