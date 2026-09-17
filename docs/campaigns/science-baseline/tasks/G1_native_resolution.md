# G1 — Native 1° ingestion; delete both upsamplers

| | |
| --- | --- |
| **Phase** | G |
| **Depends on** | none. **This is the critical path — nothing else in the campaign starts until it lands.** |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h |
| **Compute** | Perlmutter login node for CFS reads. **No `sbatch`.** No GPU. |
| **Touches** | `src/aurora_mjo/dataset.py`, `src/aurora_mjo/trainer.py` (only `_upsample_batch_gpu` and `_prep_batch`), `src/aurora_mjo/loss.py` (only grid-size constants), `configs/unified.yaml`, `tests/test_dataset_loader.py`, `tests/test_shapes.py`, `tests/test_static_vars.py`, `tests/fixtures/synthetic_archive*/`, `scripts/download_slt.py`, `data/static/slt_1deg.nc` (new), `docs/PROJECT_STATE.md` |
| **Must not touch** | `_build_aligned_index` or anything in the per-variable timestamp machinery (Lesson 5). `src/aurora_mjo/rmm/`. `src/aurora_mjo/model.py`. `src/aurora_mjo/checkpoint.py`. Any `results/*.md`. `tests/fixtures/baseline/`, `tests/fixtures/postrefactor/`. |

## Objective

The dataset returns 1° fields on the grid the archive actually uses, Aurora
receives them unmodified, and `_upsample_to_aurora` and `_upsample_batch_gpu` no
longer exist. `Batch.metadata.lat` and `.lon` are read from the NetCDF
coordinates rather than constructed. `slt` is regridded to 1° by nearest
neighbour and committed, removing the purgeable-scratch dependency.

## Why this is a separate task

Every other change in the campaign is easier to diagnose on a loader that reads
what it claims to read. Doing this alongside the loss rewrite would produce a
run where two things changed and neither can be attributed.

It is also the change with the largest downstream effect: a 16× reduction in grid
points is what makes the 1.3B model affordable (G2), and the grid-metadata fix
removes three latent defects at once (`00_CONTEXT.md` R7.5).

## What you may assume

- **Aurora accepts 180×360 and 181×360 directly.** Measured by live forward pass
  on `microsoft-aurora==1.8.0`: both crop to 180×360, patch grid `(4, 45, 90)`,
  forward completes, output finite (`00_CONTEXT.md` R5). You do not need to
  re-verify this; you need to verify it works *on real archive data*.
- The Aurora team recommended exactly this (microsoft/aurora#184, 2026-05-13).
- `45 × 90` is not a multiple of the `(2, 6, 12)` window. Aurora's Swin3D pads to
  `48 × 96` and crops back (`aurora/model/swin3d.py:350`). This is expected.
- The v3 timestamp-alignment machinery is correct and tested
  (`../refactor/results/D1_result.md`, `D3_result.md`). It is resolution-agnostic.

## Steps

1. **Read the grid before writing any code.** Open one reference file per
   variable on CFS and record `lat` and `lon` coordinate values. Assert latitude
   is descending and that all eleven variables agree to float32 exactness.
   Paste the first and last five values of each into your result file. This
   answers **Q-16** — record it there too, and add it to
   `03_DOMAIN_PRIORS.md` §2.1.
2. Replace the hard-coded `self.lat = torch.linspace(90, -90, 720)` and
   `self.lon = torch.linspace(0, 360, 1441)[:-1]` (`dataset.py:153-154`) with the
   values read from the archive. **No `linspace` for grid construction.**
3. Delete `_upsample_to_aurora` (`dataset.py:85`). Statics are loaded at native
   1° from the same archive as everything else.
4. Regrid `slt` from the 0.25° `slt_data.nc` to 1° by **nearest neighbour**
   (it is categorical, 0–7; bilinear would produce soil type 3.7). Write to
   `data/static/slt_1deg.nc` and commit it — ~65 kB as int8. Point
   `data.slt_path` at it. Add a provenance header to `scripts/download_slt.py`
   saying what the committed file was derived from. This closes **Q-07** and
   **Q-17** as a side effect; say so in your result file.
5. Delete `_upsample_batch_gpu` (`trainer.py:155`) and the `needs_upsample`
   branch in `_prep_batch` (`trainer.py:898`).
6. Update every grid-size constant and assertion: `loss.py` latitude-length
   checks, the `MoistureBudgetLoss` grid-mismatch guard, and any `720` / `1440`
   literal in `src/`. Grep for both numbers and justify every survivor.
7. Regenerate the synthetic NetCDF fixtures at 1°. They are already 1° on disk
   (the archive is 1°) — check whether anything downstream of them assumed the
   upsample.
8. Update `tests/test_shapes.py`, `tests/test_dataset_loader.py`,
   `tests/test_static_vars.py` to the new shapes. Add a test asserting
   `metadata.lat` is descending and equals the archive values.
9. **Verify sample counts are unchanged.** 1980 must still give **1,462** and
   1981 **1,458** (`03_DOMAIN_PRIORS.md` §1). If they moved, you broke Lesson 5;
   stop and report `RED`.
10. Run a single forward pass on one real 1980 sample on the login node with
    `model_type: small` and confirm the output is finite. Paste the shapes.

## Definition of Done

- [ ] Archive `lat` / `lon` read from NetCDF; first and last five of each pasted;
      descending-latitude assertion in place; **Q-16** answered in `QUESTIONS.md`
      and `03_DOMAIN_PRIORS.md` §2.1 updated
- [ ] `_upsample_to_aurora` and `_upsample_batch_gpu` **deleted**;
      `grep -rn "_upsample" src/ scripts/ tests/` returns nothing
- [ ] `grep -rn "720\|1440" src/` — every remaining hit justified in the result file
- [ ] `data/static/slt_1deg.nc` committed, nearest-neighbour regridded, integer
      values 0–7 only; `data.slt_path` points at it; no `/pscratch` path remains
      in `configs/unified.yaml`
- [ ] Dataset returns `[1, 2, 180, 360]` surface and `[1, 2, 13, 180, 360]`
      atmospheric; shapes pasted
- [ ] **Sample counts unchanged: 1,462 for 1980 and 1,458 for 1981**; alignment
      report pasted
- [ ] One real 1980 sample forwards through `AuroraSmallPretrained` with finite
      output; shapes and min/max pasted
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section written per `04_AGENT_PROTOCOL.md` §5 — grid, smoke
      losses, peak memory
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Anything in `_build_aligned_index`.** If you believe it needs to change to
  support 1°, you have misdiagnosed something — it does not know about
  resolution. Record it as an Observation and stop.
- The loss weighting (H1), the model class (G2), normalisation statistics (G3).
- Adding `tp6h` or `mslhf` to the loader — that is H3.
- Re-enabling `gradient_checkpointing`. Lesson 6; only G2 may touch it.
- Fixtures under `tests/fixtures/baseline/` or `postrefactor/` — they are
  historical record and stay byte-identical.
