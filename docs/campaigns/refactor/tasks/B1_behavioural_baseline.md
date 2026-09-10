# B1 — **THE GATE.** Fingerprint current behaviour before anything moves

| | |
| --- | --- |
| **Phase** | B |
| **Depends on** | none strictly; run **before** C1 |
| **Base branch** | `epic/refactor` |
| **Budget** | 2–3 h, dominated by waiting on the cluster |
| **Touches** | `tests/fixtures/baseline/*.json` (new), `results/B1_result.md` |
| **Must not touch** | **anything else. This task is read-only against the code.** No `src/`, no `configs/`, no `scripts/`. |
| **Needs** | Perlmutter, the conda env, CFS read access, one GPU node for step 5 |

## Objective

At the end, there is a machine-comparable record of what this code does *today*,
committed to the repository, so that C4 can prove the refactor changed nothing.

## Why this is a separate task — and why it is the gate

`src/trainer.py` is 1138 lines with **no test coverage**. C1, C2 and C3 will move
every module, re-point 27 imports and restructure the entry point. Without a
before-picture, a behaviour change introduced there is undetectable until it
surfaces weeks later as an 11-hour SLURM slot producing garbage — and by then
nobody will connect it to this campaign.

**If this task does not complete, do not start C1.** Report `BLOCKED` and stop.
An honest `BLOCKED` here is far cheaper than a refactor with no baseline.

## What you may assume

- `configs/unified.yaml` parses and resolves four modes: `baseline`,
  `physics_informed`, `lora`, `combined`. (`00_CONTEXT.md` §2.3)
- All five `src` modules and `train.py` import in the conda env.
- `train.py --smoke-test` exists and uses a synthetic in-process loader, so it
  needs a GPU but **not** CFS data. (`00_CONTEXT.md` §2.3)
- Expected values to sanity-check against are in `03_DOMAIN_PRIORS.md`. In
  particular: 1980 has **1,462** samples; statics are `z` mean 3709.2466, `lsm`
  mean 0.3357, `slt` mean 0.6708; `small` has **41,008** trainable parameters.
- `export HDF5_USE_FILE_LOCKING=FALSE` is required before any NetCDF read. E1
  has not landed, so you must set it yourself.

**Run this under the existing conda env, not under `uv`.** The point is to
fingerprint the code as it is today in the environment it actually ran in. If you
run it under a freshly resolved `uv` environment you are measuring two changes at
once, and a C4 mismatch will be unattributable.

## Steps

1. Record provenance first: `git rev-parse HEAD`, `hostname`, `python -V`,
   `python -c "import torch; print(torch.__version__)"`, and `nvidia-smi -L` if
   on a GPU node. These go at the top of every artifact you write.

2. **Config resolution, all four modes.** For each mode, resolve the config
   exactly as `train.py` does and dump it to canonical JSON — sorted keys,
   2-space indent, so a diff is meaningful:

   ```bash
   for m in baseline physics_informed lora combined; do
     python -c "
   import json, sys
   sys.path.insert(0, '.')
   from train import load_config
   cfg = load_config('configs/unified.yaml', '$m')
   print(json.dumps(cfg, indent=2, sort_keys=True, default=str))
   " > tests/fixtures/baseline/config_$m.json
   done
   ```

   Also capture one `--override` case, because C2 must preserve that semantics
   exactly. Use `--override training.optimizer.lr=1e-5` and save as
   `config_baseline_override.json`.

   Note `default=str` — if any value is not JSON-serialisable, say which in your
   result file rather than silently stringifying something important.

3. **Model parameter counts, all four modes.** Build each model via
   `src.model.load_model` with that mode's `model` config block and record
   `total`, `trainable` and `frozen` parameter counts as exact integers, plus
   whether an MJO head and LoRA adapters were constructed.

   This can be done on CPU — `load_model` does not need CUDA to count
   parameters. Doing it on CPU makes it reproducible in C4 without a GPU
   allocation, which matters. If it does require CUDA, say so and use a GPU
   node.

   **Sanity-check against `03_DOMAIN_PRIORS.md` §7 before you believe it.**
   `baseline` should show trainable ≈ 41,008 for `model_type: small`. If
   trainable equals total, `freeze_backbone` did not run and you must
   investigate before recording — a fingerprint of broken behaviour is still a
   valid fingerprint, but it must be *labelled* as such, because C4 will
   otherwise faithfully preserve a bug.

4. **Dataset index fingerprint for one year.** With
   `HDF5_USE_FILE_LOCKING=FALSE` set, construct
   `LANLMJODataset(start_year=1980, end_year=1980, root_dir=<the CFS root>,
   slt_path=<the slt path>)` and record:

   - `len(dataset)` — expect **1462**
   - the full per-variable alignment report printed by `__init__`, verbatim
   - probed pressure-level indices
   - for each static var: shape, mean, std, and **finite fraction**
   - for sample 0 and sample `len-1`: the metadata init time, and for every
     surface and atmospheric variable its shape, dtype, min, max, mean and
     finite fraction
   - **whether any `UserWarning` containing `Using zeros` was emitted** — capture
     warnings explicitly with `warnings.catch_warnings(record=True)` rather than
     relying on them appearing in stderr

   That last item is the live check for the trap in `02_UPSTREAM_CONTRACT.md`
   §4.3. **If it fires, say so prominently in your `SUMMARY` line** — it changes
   what B2 means and it changes E1's urgency.

   Write to `tests/fixtures/baseline/dataset_1980.json` plus the raw alignment
   report as `dataset_1980_alignment.txt`.

5. **Smoke-test trajectory.** On a GPU node, with a fixed seed:

   ```bash
   python train.py --config configs/unified.yaml --mode baseline --smoke-test \
     2>&1 | tee tests/fixtures/baseline/smoke_baseline.log
   ```

   Extract into `smoke_baseline.json`: per-step loss values in order, the number
   of steps, and **the non-finite-gradient skip count**.

   **Read `03_DOMAIN_PRIORS.md` §7 before interpreting this.** A previous probe
   recorded `nonfinite_grad_skips: 30` across `steps_completed: 30` — every step
   skipped. If your run does the same, the grad guard is working and the run
   learned nothing. **That is a red alarm to report, not a pass** — but it is
   still a valid fingerprint, and C4 must reproduce it. Record the ratio
   explicitly.

6. **State the comparison tolerance.** Config JSON and parameter counts are
   compared for **exact** equality in C4. Loss values are not: cuDNN kernel
   selection and reduction order are not bitwise deterministic across
   environments. Record a **relative tolerance of 1e-4** for float
   trajectories, and say in your result file that this is the tolerance C4 must
   use. If you have evidence for a tighter or looser figure — for example from
   running the smoke test twice and diffing — use it and show the evidence.

   Running the smoke test twice is cheap and worth it: it tells C4 how much of
   any future difference is just noise.

7. Commit everything under `tests/fixtures/baseline/`. Total should be well
   under 1 MB — if the smoke log is large, truncate it to the loss lines and say
   you did.

## Definition of Done

- [ ] Provenance (commit SHA, hostname, python, torch, GPU) recorded at the top
      of every artifact
- [ ] `config_{baseline,physics_informed,lora,combined}.json` written, canonical
      JSON, sorted keys
- [ ] `config_baseline_override.json` written for
      `--override training.optimizer.lr=1e-5`
- [ ] Parameter counts for all four modes as exact integers (total / trainable /
      frozen), plus MJO-head and LoRA presence; whether CPU or GPU stated
- [ ] `baseline` trainable count compared against the expected 41,008, with the
      comparison stated
- [ ] `dataset_1980.json` written; `len(dataset)` recorded and compared against
      the expected 1,462
- [ ] Per-variable alignment report captured verbatim
- [ ] Static var shape / mean / std / finite-fraction recorded and compared
      against `03_DOMAIN_PRIORS.md` §3
- [ ] **`Using zeros` warning presence explicitly recorded** — and surfaced in
      `SUMMARY` if present
- [ ] Smoke-test loss trajectory and non-finite-grad skip count captured; the
      skip/step ratio stated
- [ ] Comparison tolerance for floats stated explicitly (default: relative 1e-4)
- [ ] Total committed fixture size stated and under 1 MB
- [ ] **No file outside `tests/fixtures/baseline/` and `results/` modified** —
      `git status --porcelain` pasted to prove it
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing anything.** If you find a bug — including a firing `Using zeros`
  warning, or every step being grad-skipped — **record it and stop**. This task
  captures reality, including broken reality. Fixing it here would destroy the
  only baseline.
- Any training run beyond the smoke test.
- Running under `uv` (see What you may assume).
- Multi-year dataset construction. One year is enough for a fingerprint and
  costs minutes instead of hours.
