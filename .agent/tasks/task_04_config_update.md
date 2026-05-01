# Task: Configuration Update for NERSC

## Objective
Update the configuration files to point to the correct data paths on NERSC and disable dummy mode, so the repository is ready to merge into an integration branch for production training.

## Working rules
- Read only the files listed below unless you discover a direct dependency.
- If you need to expand scope, stop and report why.
- Make the smallest possible coherent patch.
- Keep all changes confined to the modify list unless explicitly authorized.
- Ensure conda/mamba environment `aurora_mjo` is activated before running Python code (directory `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`).

## Read
- `configs/phase1_baseline.yaml`
- `configs/phase2_physics.yaml`
- `configs/phase2_rollout.yaml`
- `configs/phase3_longrun.yaml`

## Modify
- `configs/phase1_baseline.yaml`
- `configs/phase2_physics.yaml`
- `configs/phase2_rollout.yaml`
- `configs/phase3_longrun.yaml`

## Expected output
- All 4 YAML configurations correctly set to use real NERSC data instead of dummy data.

## Steps
1. Open `configs/phase1_baseline.yaml`.
2. Locate the `data` block. Change `use_dummy: true` to `use_dummy: false`.
3. Repeat this process for `phase2_physics.yaml`, `phase2_rollout.yaml`, and `phase3_longrun.yaml`.
4. Ensure that `root` remains `/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results`.

## Risks & Reminders
- Verify that `use_dummy: false` correctly shifts execution from the dummy dataset over to the `LANLMJODataset` dataloader verified in previous tasks.
