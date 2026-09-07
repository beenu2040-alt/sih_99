# Implementation Plan: Fix Validator and Inspector Logic

## Goal Description
The schedule inspection script previously reported 51 train conflicts and 14 resource conflicts. Our root cause analysis found these to be validator false positives due to overly strict, naive logic. We need to fix the validation logic so that it matches the CP-SAT optimizer's actual constraints, add regression tests to the project's test suite, and generate an updated inspection report.

## Proposed Changes

### `src/optimizer/validate.py`
We will update the `validate_schedule` function in `validate.py` (which is the canonical validator) so that:
1.  **Train Conflicts:** It already correctly uses `needs_block` to exempt tasks that don't need infrastructure blocks from train conflicts. We will preserve this logic. We will enhance it to provide detailed error reports (task_id, section, etc.) matching the requested format if needed.
2.  **Resource Conflicts:** We will update the logic to validate overlaps at the **block** level instead of just the task level, checking if blocks in the same section overlap *and* require the same resource. This aligns with the user's explicit instructions to check every pair of overlapping maintenance blocks and compare their resource requirements. (Alternatively, since tasks build the blocks, checking tasks by resource group is equivalent, but we will ensure the block-level logic is present or the detailed error matches the user's block overlap scenario). Actually, to "use the same resource grouping logic as the optimizer", we can just use `ctx.tasks_by_resource` which the optimizer builds. We will extract this logic into a helper function.

### `inspector.py`
We will rewrite `inspector.py` to be a thin wrapper that imports the pipeline (data loading, preprocessing, model building) and directly calls `validate_schedule` from `src.optimizer.validate`. It will output the detailed JSON violation report and terminal summary expected by the user. By reusing `validate.py`, we avoid duplicating the rules.

### `tests/optimizer/test_validation.py`
We will add TEST 1 through TEST 10 as requested by the user, covering all edge cases for train conflicts, resource conflicts, deadline, duration, and duplicate assignments.

## Verification Plan
1. Run the optimizer tests (`pytest tests/optimizer`).
2. Run the full test suite (`pytest tests`).
3. Run the updated `inspector.py` against the existing output files.
4. Verify the updated `schedule_inspection_report.json` reports 0 train conflicts and 0 resource conflicts, completely eliminating the false positives.
