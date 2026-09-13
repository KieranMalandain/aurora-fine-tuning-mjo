# Aurora–MJO: repository consolidation and forward plan

**Status:** draft handoff note, September 2026. Written to prime a fresh working context.
Not authoritative — the "Plan going forward" section in particular is a sketch to be
expanded.

**Repo:** `git@github.com:KieranMalandain/aurora-fine-tuning-mjo.git`
**Working copy:** `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo` (Perlmutter, NERSC)
**Env:** `/pscratch/sd/k/kam352/conda_envs/aurora_mjo` — Python 3.10, torch 2.5.1+cu121,
xarray 2025.6.1, numpy 2.2.6, `microsoft-aurora` via pip

---

## 1. What the project is

Fine-tuning Microsoft Aurora (1.3B parameter 3D Swin Transformer) for sub-seasonal prediction
of the Madden–Julian Oscillation. The approach treats MJO forecasting as a physics-consistent
initial value problem — prognostic state stepping rather than statistical anomaly regression —
with the goal of skillful prediction at 30+ day lead times (target: RMM correlation r > 0.5
at 30 days).

Data is the 1° NERSC/LANL ERA5 preprocessing output on CFS, upsampled to Aurora's 0.25°
(720×1440) grid.

---

## 2. Starting state, 8 September 2026

The repository had been untouched for roughly two months and was spread across four git
worktrees on `/pscratch` with uncommitted work in three of them:

```
Aurora/
├── slt/                                    soil-type data (out of scope)
├── aurora-fine-tuning-mjo/                 main
└── worktrees/
    ├── human-integration/                  integration/antigravity   (21 files dirty)
    ├── campaign-fixes/                     agent/campaign-fixes      (2 files dirty)
    └── agent-simul-training-one/           agent/simul-training-one  (11 files dirty)
```

Plus one stash, 1.7 GB of untracked checkpoints, and no clear record of which branch held the
current implementation.

### Actual topology, once measured

```
main (07a9ce0)
├── integration/antigravity (b0cff2d)          main +4
│   └── agent/simul-training-one (6862fdf)     antigravity +1   ← superset
└── agent/campaign-fixes (b9ffe63)             main +2
```

`integration/antigravity` was entirely contained in `agent/simul-training-one`. Only two lines
had genuinely diverged from `main`, not three.

---

## 3. What was done

Nine sequential tasks, each dispatched to a separate agent with read-only reconnaissance first
and no write operations until the state was fully understood. Reports live in
`/pscratch/sd/k/kam352/Aurora/cleanup-reports/`.

| # | Task | Result |
|---|---|---|
| 01 | Backup tags + `git bundle` | 4 tags, bundle to `$HOME` (off scratch) |
| 02 | Per-checkout inventory | Found the dirty state and the duplicate stash |
| 03 | Divergence + conflict prediction | Established the real topology; 9 predicted conflicts |
| 04 | Hygiene audit | 4.59 MiB packed, no large binaries tracked → no history rewrite needed |
| 05 | Preserve uncommitted work | 3 `wip/snapshot/*` tags via `git stash create`, patches, tarball |
| 06 | Content comparison of the two lineages | Established that campaign-fixes is content-newest |
| 07 | Land WIP as commits | 5 new commits across 3 branches; stash dropped; empty stray file removed |
| 08 | Build consolidation branch | Aborted correctly — predicted conflict set was stale |
| 09 | Consolidation retry | Completed; all verification passed |

Task 08's abort was a planning error: the conflict set had been derived from pre-task-07
divergence data, and task 07's new commits created fresh overlaps in four doc files and
`slurm_scripts/train.slurm`. The abort guard caught it and cost nothing.

---

## 4. The central decision

**`agent/campaign-fixes` wins every conflict in `src/`, `train.py`, `configs/`,
`.gitignore` and `scripts/calc_norm_stats.py`.**

In the git graph `campaign-fixes` branched from `main` and appeared to have lost the NERSC
dataloader and soil-type work. In content it had not — the work was hand-carried forward. The
evidence:

- Its `dataset.py` has `slt_path`, the `slt_data.nc` default and the `Step00/ERA5.invariant`
  globbing, none of which exist on `main`.
- Its `_load_static_vars` docstring reads *"v3: NaN/Inf are zeroed for ALL three statics
  (previously only slt was sanitized)"* — a direct reference to the version it derived from.
- Structurally it is a superset of simul's version plus `_to_seconds_i64`,
  `_build_aligned_index`, `_clean` and `_read_var_at_times`.
- `configs/unified.yaml`'s `lora` mode is named `phase2_rollout_lora`, knowingly superseding
  simul's config file of that name.

Corroborating detail: antigravity's uncommitted `src/loss.py` change is byte-identical to
what `6862fdf` committed, so work was demonstrably being copied between branches rather than
developed independently.

**One deliberate exception:** `slurm_scripts/eval.slurm` was kept from our side despite
campaign-fixes deleting it, because the 25-line version is the only copy anywhere. It turns
out to be non-functional (see §7) but is retained as a draft.

Also decided: history preserved throughout. Real merges, no rebase, no squash. There were only
five non-merge commits across all three branches, so squashing would have gained nothing.

---

## 5. Findings inherited from the July campaign

From `AURORA_MJO_GAMEPLAN.md`, written under deadline pressure ("Perlmutter goes offline in
~2 days", batch queue unusable at ~50 h scheduling latency, everything in chained 4 h
`salloc` sessions). This document is the most substantive analysis in the repository.

**Finding 1 — `msl` is surface pressure normalised as MSL.** `dataset.py` maps `msl` to
LANL's `ps` because the archive has no true MSL, but Aurora normalises that channel with
`location=100958, scale=1332`. High terrain lands at −36σ. Confirmed trigger for
100%-non-finite validation. `unified.yaml` now overrides with `ps`-appropriate stats, but
those values are marked **PLACEHOLDER** and must be replaced with real computed stats.

**Finding 2 — no gradient-finiteness check at the optimizer boundary.** Under bf16
`GradScaler` is disabled, so `scaler.step()` degrades to a bare `optimizer.step()` with no
inf-check. `clip_grad_norm_` computes a NaN total norm without rejecting it. Adam updates its
moment buffers before applying lr, so the buffers are poisoned even during warmup at lr≈0 —
which is why the failure mode was NaN-forever rather than NaN-once.

**Finding 3 — rollout feeds unclamped predictions back in**, bypassing Aurora's
`apply_rollout_input_clipping`.

**Finding 5 — physics-informed mode freezes the entire backbone**, so the moisture-budget
loss scores fields whose decoders cannot learn. Flagged as needing a human decision.

**Finding 6 — the `msl` output head is frozen and MSL-calibrated**, so after the input
renorm it denormalises with the wrong stats.

Also noted: `gradient_checkpointing: false` everywhere in `unified.yaml` because
checkpointing deterministically triggers an illegal memory access on Perlmutter. The full
1.3B model does not fit on 4×A100 80G without it.

---

## 6. HDF5 file locking — resolved

The long-running blocker on this project. `xr.open_dataset` on CFS fails with:

```
OSError: [Errno -101] NetCDF: HDF error: '.../e5.oper.an.sfc.128_167_2t...nc'
```

on files that exist and are readable. Cause is HDF5 file locking against the parallel
filesystem. Fix:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

Verified 8 September 2026: with the variable set, the same file opens cleanly (180×360×124,
variable `t2`). This needs to be set in the slurm scripts and ideally at process start in
`train.py`, not left to the interactive shell.

**Unconfirmed but important:** `dataset.py` catches this error for the invariant `z` and `lsm`
files and substitutes **zero tensors** with only a `UserWarning`. If the July runs did not
have the variable set, they trained on zero geopotential and a zero land-sea mask while
printing two warnings. This would compound with Finding 1. Confirm by grepping July slurm logs
for `Using zeros`. Either way the fallback should become a hard failure.

---

## 7. Current state

One branch (`main`), one worktree, no stash, 73 tracked files, 4.9 MB packed.

```
.agent/          rules/ and workflows/ (tasks/ pruned)
configs/         unified.yaml — four modes: baseline, physics_informed, lora, combined
docs/            architecture, documentation, project-brief, known-gaps, nersc-dataset-*,
                 experiment-registry, papers/, images/
scripts/         calc_norm_stats, compute_rmm, evaluate_mjo, scan_for_bad_values,
                 download_era5, download_slt, explore_nersc_data, verify_dataset_loader,
                 verify_shapes, plot_loss, test_num_workers, smoke_test_{freeze,mjo_head,rollout}
slurm_scripts/   train_auto.slurm, submit_chain.sh, eval.slurm, test_train.slurm
src/             dataset, model, loss, trainer, checkpoint
tools/           diagnose_val_nan, probe_model_size, repro_ima_matrix
train.py, test_dataset.py, inspect_vars.py, environment.yml, AURORA_MJO_GAMEPLAN.md
```

Verified: all 27 Python files parse, all 3 YAML files load, all five `src` modules and
`train.py` import in the real env, `unified.yaml` parses with four modes.

### Known broken

1. **`src/trainer.py:307`** — `from src.dummy_dataset import MJODataset, load_and_combine_files`.
   `src/dummy_dataset.py` was deleted by campaign-fixes but the import remains. Lazy import
   inside the `use_dummy` branch, so `unified.yaml` (`use_dummy: false`) is unaffected; the
   dummy path is dead.
2. **`slurm_scripts/eval.slurm`** — references deleted `configs/phase1_baseline.yaml`, passes
   `--override data.use_dummy=true` (hits item 1), and hardcodes
   `/pscratch/sd/k/kam352/conda/envs/aurora_mjo/bin/python`, which is the wrong path (actual:
   `conda_envs`, not `conda/envs`).
3. **Stale references throughout** to `phase1_baseline.yaml`, `phase2_physics.yaml`,
   `phase3_longrun.yaml`, `dummy_dataset`, `slurm_scripts/train.slurm` — in `README.md`,
   `docs/documentation.md`, `docs/experiment-registry.md`, `scripts/evaluate_mjo.py`,
   `scripts/verify_shapes.py`, `.agent/rules/01-project-context.md`.
4. **`README.md` describes the wrong machine** — Yale Bouchet and H200s; the actual work was
   Perlmutter and 4×A100 80G.
5. **`checkpoints/` and `test_outputs/` are tracked** and should be gitignored.
6. **`msl` norm stats in `unified.yaml` are placeholders** (Aurora's built-in `sp` values).

---

## 8. Recovery points

| Ref | Points at |
|---|---|
| `backup/pre-cleanup/main` | `07a9ce0` |
| `backup/pre-cleanup/antigravity` | `b0cff2d` |
| `backup/pre-cleanup/campaign-fixes` | `b9ffe63` |
| `backup/pre-cleanup/simul-training` | `6862fdf` |
| `wip/snapshot/human-integration` | 21-file dirty tree (`05e8515`) |
| `wip/snapshot/campaign-fixes` | pre-restore `metrics.jsonl` |
| `wip/snapshot/simul-training-one` | 11-file dirty tree |
| `v0.1.0-pre-agents`, `v0.1.1-agent-scaffold`, `v0.1.2-pre-campaign-fixes` | pre-existing |
| `origin/archive/pre-antigravity-baseline` | `800455a` — remote only, never merged, **do not delete** |

`$HOME/aurora-backup/` holds four incremental bundles, the preserved-patches tarball, and
three conda env exports (`-full.yml`, `-explicit.txt`, `-list.txt`). The explicit export is
the ground truth for dependencies, not `environment.yml`.

**Not covered by any backup:** the 1.7 GB of `.pt` checkpoints, deliberately deleted as
artifacts of a run the gameplan declares fully non-finite.

---

## 9. Plan going forward

Sketch, to be expanded. Ordered so diagnostics precede tooling churn — reversing the order
makes it much harder to tell whether a change fixed something or moved the failure.

### Phase A — Unblock and make failures loud

- Set `HDF5_USE_FILE_LOCKING=FALSE` in the slurm scripts and at process start in `train.py`.
- Turn the `z`/`lsm`/`slt` zero-substitution fallback into a hard failure.
- Grep July logs for `Using zeros` to determine whether past runs were affected.
- Fix `src/trainer.py:307`: either restore a dummy dataset path or remove it entirely.
- Rewrite `slurm_scripts/eval.slurm` against `unified.yaml`.
- Compute the real `ps` mean/std and replace the placeholder `msl` norm stats.

### Phase B — Dependencies

- Migrate conda → `uv`, using `aurora_mjo-explicit-*.txt` as the source of truth.
- Constraint to work outward from: `microsoft-aurora` on Python 3.10 with torch 2.5.1+cu121.
- Add `pyproject.toml`; retire `environment.yml`.

### Phase C — Test suite

More starting material than expected. `scripts/smoke_test_freeze.py`,
`smoke_test_mjo_head.py`, `smoke_test_rollout.py`, `verify_dataset_loader.py`,
`verify_shapes.py` and `test_dataset.py` already exist — this is conversion work, not
greenfield. Priority targets: dataset construction against real paths, static-var shapes and
finiteness, config-mode resolution, the gradient-finiteness guard from Finding 2.

### Phase D — Code review and CLI

- Dead-code pass; reconcile the stale references in §7.
- `.gitignore` for `checkpoints/`, `test_outputs/`, `tools/probe_results/`,
  `docs/verify_output*.txt`; untrack what's already committed.
- Coherent CLI over `unified.yaml`'s mode/override system.
- Rewrite `README.md` for Perlmutter.

### Phase E — Back to science

- Address Findings 2, 3, 5 and 6 properly rather than as deadline workarounds.
- Re-run the model-size probe (`tools/probe_model_size.py`) to settle small vs full Aurora
  on evidence rather than under time pressure.
- Baseline → evaluate → LoRA rollout, in the gameplan's priority order.

---

## 10. Open questions

1. Were the July runs affected by the zero-substitution bug? Determines whether any prior
   result means anything.
2. Small vs full Aurora — the gameplan chose small under deadline pressure and explicitly
   flagged the decision as provisional.
3. Physics-informed mode is scientifically incoherent as currently wired (Finding 5). Fix the
   freezing, or drop the phase?
4. Is `docs/experiment-registry.md` still accurate? It was not examined during consolidation.
5. `slt_data.nc` lives on `/pscratch` and is a hard dependency of `dataset.py`.
   `scripts/download_slt.py` is the only record of how it was produced. Move it somewhere
   permanent.