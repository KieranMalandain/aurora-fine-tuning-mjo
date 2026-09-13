# Context

Everything an agent needs that is **already true**. Written so a fresh context
window needs nothing else. If this file and the code disagree, the code is
right and this file is stale — say so in your result file.

The long-form handoff note that preceded this campaign is
[`cleanup-2026-09.md`](cleanup-2026-09.md). It is the source for most of what
follows and remains worth reading in full for the git-consolidation history.
This file is the operative summary.

---

## 1. What this repo is

`aurora-fine-tuning-mjo` fine-tunes **Microsoft Aurora** (1.3B-parameter 3D Swin
Transformer) for sub-seasonal prediction of the **Madden–Julian Oscillation**.
The approach treats MJO forecasting as a physics-consistent initial value
problem — prognostic state stepping, not statistical anomaly regression — with a
target of RMM correlation r > 0.5 at 30-day lead.

Training runs on **NERSC Perlmutter**, 1 node × 4 × A100 80GB, under SLURM.
Input data is the 1° NERSC/LANL ERA5 preprocessing output on CFS, upsampled on
GPU to Aurora's native 0.25° (720×1440) grid.

The repository was untouched for roughly two months, then went through a
nine-task git consolidation in September 2026 that collapsed four worktrees and
three divergent branches into a single `main`. That consolidation is **done**.
This campaign is what comes after it.

---

## 2. What is already true

### 2.1 Repository shape, as of the start of this campaign

73 tracked files, ~4.9 MB packed, one branch (`main`), one worktree, no stash.

```text
.agent/          rules/ (6 files) and workflows/ (5 files); tasks/ was pruned
configs/         unified.yaml — four modes: baseline, physics_informed, lora, combined
docs/            architecture, documentation, project-brief, known-gaps,
                 nersc-dataset-information, experiment-registry, git-policy,
                 development-workflow, evaluation-spec, compute-environments,
                 antigravity-usage, papers/, images/, campaigns/refactor/
scripts/         18 files: calc_norm_stats, compute_rmm, evaluate_mjo,
                 scan_for_bad_values, download_era5 (EMPTY), download_slt,
                 explore_nersc_data, verify_dataset_loader, verify_shapes,
                 plot_loss, test_num_workers, smoke_test_{freeze,mjo_head,rollout}
slurm_scripts/   train_auto.slurm, submit_chain.sh, eval.slurm, test_train.slurm
src/             __init__.py (EMPTY), dataset, model, loss, trainer, checkpoint
tools/           diagnose_val_nan, probe_model_size, repro_ima_matrix
notebooks/       t_nb.py (EMPTY)
root             train.py, test_dataset.py, inspect_vars.py, environment.yml,
                 AURORA_MJO_GAMEPLAN.md, AGENTS.md, GEMINI.md, README.md
```

Python line counts, largest first — this is the real weight of the refactor:

| File | Lines |
| --- | --- |
| `src/trainer.py` | 1138 |
| `scripts/evaluate_mjo.py` | 984 |
| `scripts/compute_rmm.py` | 653 |
| `src/model.py` | 494 |
| `src/dataset.py` | 423 |
| `train.py` | 411 |
| `tools/probe_model_size.py` | 411 |
| `tools/diagnose_val_nan.py` | 337 |
| `src/checkpoint.py` | 312 |
| `src/loss.py` | 258 |
| everything else | ~1500 |
| **total** | **~6950** |

There are **27 first-party `from src.…` import statements** across the repo.
Those are the edges the package move has to re-point.

### 2.2 The environment, today

- Conda env at `/pscratch/sd/k/kam352/conda_envs/aurora_mjo`
- Python 3.10, torch 2.5.1+cu121, xarray 2025.6.1, numpy 2.2.6
- `microsoft-aurora` installed **via pip**, not conda
- `environment.yml` is a loose, unpinned spec and is **not** the source of
  truth. The ground truth for dependencies is
  `$HOME/aurora-backup/aurora_mjo-explicit-*.txt` (a `conda list --explicit`
  export), with `-full.yml` and `-list.txt` alongside it.

There is **no** `pyproject.toml`, no lockfile, no `tests/` directory, no CI, no
pre-commit, no linter config, and no type checker.

### 2.3 What works right now

Verified at the end of the consolidation:

- All 27 Python files parse.
- All 3 YAML files load.
- All five `src` modules and `train.py` import in the real conda env.
- `configs/unified.yaml` parses and resolves all four modes.

`train.py` already has a real CLI surface: `--config`, `--mode`,
`--override KEY=VALUE` (repeatable, dot-notation), `--smoke-test`,
`--resume auto|none|PATH`. Mode resolution is a `_deep_merge` of the top-level
config with `modes.<name>`. **This works. Do not throw it away** — the campaign
formalises it, it does not replace it.

`slurm_scripts/train_auto.slurm` is a genuinely careful piece of work: 11.5 h
walltime against an in-process 11.0 h limit, `--signal=B:USR1@1800` forwarding
to the trainer for a checkpoint-and-exit-99 contract, a `DONE` marker so
over-provisioned chains no-op, a `STOP_TRAINING` kill switch, and `-c 64`
because an earlier version requested 8 CPUs for 16 dataloader workers and
thrashed. Read the comments before you touch it.

### 2.4 What is known broken

Six items, all diagnosed, none yet fixed:

1. **`src/trainer.py:307`** — `from src.dummy_dataset import MJODataset,
   load_and_combine_files`. `src/dummy_dataset.py` was deleted during
   consolidation, but the import survived. It sits inside the `if use_dummy:`
   branch, so it is a lazy import on a dead path: `unified.yaml` sets
   `use_dummy: false`, so nothing hits it today. It is still a landmine.
2. **`slurm_scripts/eval.slurm`** — references the deleted
   `configs/phase1_baseline.yaml`, passes `--override data.use_dummy=true`
   (which detonates item 1), and hardcodes
   `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python` — wrong path, the
   real one is `conda_envs`, not `conda/envs`. This file is non-functional. It
   was deliberately retained as a draft because the 25-line version is the only
   copy that exists anywhere.
3. **Stale references to deleted files** — `phase1_baseline.yaml`,
   `phase2_physics.yaml`, `phase3_longrun.yaml`, `dummy_dataset`,
   `slurm_scripts/train.slurm`. Present in 12 files: `README.md`,
   `docs/documentation.md`, `docs/experiment-registry.md`,
   `scripts/evaluate_mjo.py`, `scripts/verify_shapes.py`,
   `.agent/rules/01-project-context.md`, `configs/unified.yaml` (in a comment),
   `slurm_scripts/eval.slurm`, `slurm_scripts/test_train.slurm`,
   `src/trainer.py`, `AURORA_MJO_GAMEPLAN.md`, and this campaign's own
   `cleanup-2026-09.md`.
4. **`README.md` describes the wrong machine** — it talks about Yale Bouchet and
   H200s. The work happened on Perlmutter with 4 × A100 80G.
5. **Tracked files that should be ignored** — `checkpoints/baseline/metrics.jsonl`,
   `checkpoints/lora/metrics.jsonl`, `test_outputs/diagnose_val_nan_outputs.txt`,
   `tools/probe_results/{huge,small}.json`, `docs/verify_output.txt`,
   `docs/verify_output_slt.txt`. Note the `.gitignore` currently has explicit
   negations to *keep* the `metrics.jsonl` files — that was intentional once.
   Decide, don't just delete (see `01_TARGET_STATE.md` D7).
6. **`msl` norm stats in `unified.yaml` are placeholders.** They are Aurora's
   own built-in `sp` values, not stats computed from this dataset. The config
   comment says so. See §3 Lesson 1.

### 2.5 Empty files

`src/__init__.py`, `scripts/download_era5.py`, `notebooks/t_nb.py` are all
zero bytes. The first is legitimate; the other two are debris.

---

## 3. Lessons this repo has already paid for

Each of these cost real time on a machine with a ~50-hour batch queue latency.
**Do not re-learn them.** Every one of them is a candidate regression test.

### Lesson 1 — `msl` is surface pressure wearing an MSL costume

`dataset.py` maps Aurora's `msl` channel to LANL's `ps` (surface pressure),
because the archive has no true mean-sea-level pressure. But Aurora normalises
that channel with `location=100958, scale=1332` — MSL-calibrated. Surface
pressure over high terrain therefore lands at roughly **−36 σ**.

This is the **confirmed trigger** for 100%-non-finite validation loss.

`unified.yaml` now overrides the normalisation with `ps`-appropriate stats,
which pulls the worst case to about −4.6 σ — "good enough to unblock". Those
override values are **explicitly marked PLACEHOLDER** and must be replaced with
stats actually computed over the training years.

**Rule:** never trust a normalisation constant that was not computed from this
dataset. **Guard:** `scripts/calc_norm_stats.py` exists to compute them; the
placeholder is flagged in a comment in `configs/unified.yaml`.

### Lesson 2 — a finite loss can still produce non-finite gradients

Under bf16, `GradScaler` is disabled, so `scaler.step()` degrades to a bare
`optimizer.step()` with no inf-check. `clip_grad_norm_` will happily compute a
NaN total norm and propagate it through the clip without raising. And Adam
updates its moment buffers *before* applying the learning rate — so the buffers
get poisoned even during warmup at lr ≈ 0. That is why the observed failure mode
was NaN-**forever** rather than NaN-once.

**Guard: this one is already fixed.** `src/trainer.py` around line 917 has an
explicit per-parameter `torch.isfinite(p.grad).all()` check, made collective
across DDP ranks via `_collective_all_finite` so all ranks step-or-skip
together, with a `_nonfinite_grad_steps` counter and a `[grad-guard]` warning.
**Read the comment block above it before touching that code.** The campaign's
job here is to put a *test* around it, not to reimplement it.

### Lesson 3 — HDF5 file locking against a parallel filesystem

`xr.open_dataset` on CFS fails with

```text
OSError: [Errno -101] NetCDF: HDF error: '.../e5.oper.an.sfc.128_167_2t....nc'
```

on files that exist and are readable. The cause is HDF5 advisory locking on a
parallel filesystem. The fix is one environment variable:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

Verified 8 September 2026: with it set, the same file opens cleanly
(180×360×124, variable `t2`). It **is** set in `slurm_scripts/train_auto.slurm`.
It is **not** set at process start in Python, and it is not set in the other
slurm scripts.

**Rule:** an environment variable that is load-bearing for correctness does not
belong only in a shell script.

### Lesson 4 — the fallback that hides Lesson 3 and silently zeroes the physics

`src/dataset.py::_load_static_vars` wraps the invariant `z` (geopotential) and
`lsm` (land-sea mask) loads in `except Exception` and substitutes **zero
tensors**, emitting only a `UserWarning`:

```python
except Exception as e:
    warnings.warn(f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros.")
```

If the July runs did not have `HDF5_USE_FILE_LOCKING=FALSE` set, they trained on
zero geopotential and a zero land-sea mask while printing two warnings. That
would compound with Lesson 1. **This is unconfirmed** — it is resolved by
grepping the July SLURM logs for `Using zeros`, which is task **B2**.

**Rule:** a fallback that produces physically meaningless input must be a hard
failure, not a warning. Either way the outcome of B2 does not change what the
code should do — it only changes whether any prior result means anything.

### Lesson 5 — a glob is not a year filter

The pre-v3 dataset built one global index from the *reference* surface
variable's file list and reused `(file_idx, local_time_idx)` pairs for every
other variable, assuming identical chunking and identical time axes. Measurement
against Perlmutter logs killed that assumption: the 1980–2015 "train" dataset
reported **54,060** timesteps. 1980–2015 is 52,596. 54,060 is exactly 1980–**2016**
— and 2016 is the first *validation* year. The glob was pulling a validation
year into training, and nothing enforced the requested range on what the glob
returned.

`dataset.py` v3 makes this structurally impossible: per-variable
`{timestamp -> (file_idx, local_idx)}` maps built from each variable's own `time`
coordinate, hard year-range enforcement at build time, and a sample timeline
that is the sorted **intersection** of all variables' timestamps with exact 6-hour
spacing enforced across every timestep a sample needs.

**Rule:** derive sample counts from the requested range and assert against them.
**Guard:** the v3 alignment report printed by `__init__`. This is the single
highest-value regression test in the repo — see `03_DOMAIN_PRIORS.md`, where the
expected counts are exact and arithmetic.

### Lesson 6 — gradient checkpointing crashes this machine

`gradient_checkpointing: false` everywhere in `unified.yaml`, because
checkpointing **deterministically** triggers an illegal memory access on
Perlmutter. The consequence is real: the full 1.3B model does not fit on
4 × A100 80G without it, which is *why* `model_type: "small"` is currently
selected. `tools/repro_ima_matrix.py` exists to search for a crash-free
configuration; `training.sdpa_backend: "math"` is the documented first-line
workaround.

**Rule:** do not flip `gradient_checkpointing` to `true` as a "cleanup". The
`false` is load-bearing and comments say so.

### Lesson 7 — plan against measured state, not remembered state

Task 08 of the consolidation aborted correctly: its conflict set had been
derived from divergence data collected *before* task 07 landed five new commits,
and those commits created fresh overlaps in four doc files. The abort guard
caught it and cost nothing.

**Rule:** re-measure before you act on an inherited plan. That includes this
campaign's plan.

---

## 4. The third-party name collision — read this before writing any import

`microsoft-aurora` **occupies the top-level `aurora` import name.** It is
imported nine times across six first-party files:

```text
src/dataset.py:55      from aurora import Batch, Metadata
src/model.py:30        from aurora import Aurora, AuroraSmallPretrained
src/model.py:31        from aurora.batch import Batch
src/model.py:32        from aurora.model.lora import LoRA, LoRARollout
src/model.py:33        from aurora.normalisation import locations, scales
src/trainer.py:231     from aurora.batch import Batch, Metadata
scripts/evaluate_mjo.py:449          from aurora.batch import Metadata
scripts/smoke_test_freeze.py:30      from aurora.model.lora import LoRA, LoRARollout
scripts/smoke_test_mjo_head.py:19    from aurora import Batch, Metadata
```

A first-party package at `src/aurora/` installed as `aurora` would shadow all of
it. The failure would be an `ImportError` on `Batch` in the best case and a
silently wrong resolution order in the worst.

**The first-party package is therefore `aurora_mjo`, at `src/aurora_mjo/`.** See
`01_TARGET_STATE.md` D1 and `QUESTIONS.md` Q-01.

---

## 5. What is out of scope for this campaign

Written down so that a helpful agent does not implement it.

- **The science.** Findings 3 (rollout feeds unclamped predictions back in,
  bypassing Aurora's `apply_rollout_input_clipping`), 5 (physics-informed mode
  freezes the entire backbone, so the moisture-budget loss scores fields whose
  decoders cannot learn) and 6 (the `msl` output head is frozen and
  MSL-calibrated, so after the input renorm it denormalises with the wrong
  stats) are **not** in this campaign. Finding 5 in particular needs a human
  scientific decision, not a refactor.
- **Small vs full Aurora.** `model_type: "small"` was chosen under deadline
  pressure and explicitly flagged as provisional. Re-running
  `tools/probe_model_size.py` to settle it on evidence is a later campaign.
- **Computing the real `ps` norm stats.** This campaign makes the placeholder
  *impossible to use by accident* (task D3). It does not compute the
  replacement, because that needs a real multi-year pass over CFS.
- **Any training run whose purpose is a result.** The only run this campaign
  performs is the acceptance smoke test in F2.
- **History rewriting.** The hygiene audit found 4.59 MiB packed and no large
  binaries in history. No filter-repo, no rebase of `main`, no squash.
- **Deleting anything under §6 of `cleanup-2026-09.md`'s recovery table.** In
  particular `origin/archive/pre-antigravity-baseline` is remote-only, never
  merged, and marked **do not delete**.