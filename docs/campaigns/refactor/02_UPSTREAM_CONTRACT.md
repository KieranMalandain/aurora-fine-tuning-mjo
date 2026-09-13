# Upstream contract

**Binding.** What the upstream data archive guarantees, what it does not, and
the alignments that silently return wrong answers rather than raising.

"Upstream" here is the NERSC/LANL ERA5 preprocessing output, plus two files this
project depends on that are *not* part of it.

---

## 1. Read-only discipline

The ERA5 archive at

```text
/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
```

belongs to another group's CFS allocation. **This repo reads it and never
writes to it, ever, under any circumstance, including "just to add a fixed
file".** If a value upstream is wrong, it is fixed upstream, by its owners. It is
never patched in place, and a corrected copy never lands inside the archive
tree.

Derived artefacts — normalisation statistics, RMM indices, fixtures — are
written to project-owned paths under `/pscratch/sd/k/kam352/` or into the repo.

Opening any file from this root with an xarray mode other than read is a defect.
`xr.open_dataset(...)` without a write mode is the only acceptable access.

---

## 2. What upstream provides

Eleven variables, mapped in `src/aurora_mjo/dataset.py` as
`{aurora_name: (step_subdir, glob, native_var_name)}`. **The native names differ
from the Aurora names.** This table is the mapping; read it before assuming a
variable name.

### Surface variables

| Aurora name | Step subdir | Native name | Notes |
| --- | --- | --- | --- |
| `2t` | `Step02/…6hrInst/T2` | `t2` | |
| `10u` | `Step02/…6hrInst/U10` | `u10` | |
| `10v` | `Step02/…6hrInst/V10` | `v10` | |
| `msl` | `Step02/…6hrInst/PS` | `ps` | **PROXY. See §4.1.** |
| `ttr` | `Step03/…6hrInst/meanTNLWFLX` | `mtnlwrf` | top net LW flux; sign-negative |
| `tcwv` | `Step02/…6hrInst/tcwv` | `tcwv` | |

### Atmospheric variables

| Aurora name | Step subdir | Native name |
| --- | --- | --- |
| `z` | `Step01/…6hrInst/gopt` | `z` |
| `q` | `Step01/…6hrInst/sphu` | `q` |
| `t` | `Step01/…6hrInst/tprt` | `t` |
| `u` | `Step01/…6hrInst/uWnd` | `u` |
| `v` | `Step01/…6hrInst/vWnd` | `v` |

The full subdirectory stem is `ERA5.remap_180x360MODIS_6hrInst`.

### Static / invariant fields

| Field | Source | Notes |
| --- | --- | --- |
| `z` (surface geopotential) | `Step00/ERA5.invariant` glob | **§4.3** |
| `lsm` (land-sea mask) | `Step00/ERA5.invariant` glob | **§4.3** |
| `slt` (soil type) | `slt_data.nc` — **NOT in the archive** | **§4.4** |

---

## 3. What upstream guarantees

| Guarantee | Evidence | Where it can break |
| --- | --- | --- |
| 1° regular lat/lon grid, 180 × 360 | `verify_dataset_loader.py` output; `docs/verify_output.txt` | Nothing observed. Aurora needs 720 × 1440, so the trainer upsamples on GPU; static vars are upsampled once at dataset init. |
| 6-hourly instantaneous, 4 timesteps/day | directory names contain `6hrInst`; sample-count arithmetic in `03_DOMAIN_PRIORS.md` matches exactly | A missing file silently shortens a variable's timeline. v3 handles this by intersection — see §4.2. |
| Each file carries a real `time` coordinate | v3 index is built from `time` coords rather than filenames | If a file's `time` were wrong the index would be wrong and nothing would raise. Unverified whether this has ever happened. |
| 13 Aurora pressure levels are available | `_probe_pressure_level_indices()` probes rather than assumes | Level *order* is probed, not assumed. Do not hardcode level indices. |
| Files are readable by this project's allocation | routine | **Only with `HDF5_USE_FILE_LOCKING=FALSE`.** See §4.5. |

Note the pattern: every guarantee in this table is one the code **measures**
rather than trusts. That is deliberate and it is the direct product of Lesson 5.
Preserve it. A refactor that replaces a probe with a constant is a regression
even if the constant is currently correct.

---

## 4. Alignments that silently return wrong answers

These are the traps. Each one produces a *plausible* wrong answer rather than an
exception, which is what makes them expensive.

### 4.1 `msl` is not `msl`

**The trap.** Aurora's `msl` channel is fed LANL's `ps` (surface pressure)
because the archive has no true mean-sea-level pressure. Aurora then normalises
that channel with its built-in MSL constants (`location=100958, scale=1332`).
Surface pressure over high terrain lands at about **−36 σ**.

**What the wrong answer looks like.** 100% non-finite validation loss. Confirmed
trigger, not hypothesis.

**The right handling.** `configs/unified.yaml` overrides
`model.norm_stats.msl`, which forces Aurora's global `locations`/`scales` dicts
to use pressure-appropriate values. **The current override values are Aurora's
own built-in `sp` stats, used as a stopgap — they are PLACEHOLDER and are not
computed from this dataset.** They pull the worst case to ≈ −4.6 σ, which is
enough to unblock and not enough to trust.

**In this campaign.** D3 adds a test that fails if the placeholder values are
still present *and* the config is being used for anything other than a smoke
test. Computing the real values is out of scope (`00_CONTEXT.md` §5).

**Do not** "fix" this by removing the override — that restores the −36 σ
failure. **Do not** change the `msl → ps` mapping to a different variable
without a human decision; there is no MSL field to map to.

### 4.2 Per-variable file lists and time axes are not identical

**The trap.** The pre-v3 code built one global `(file_idx, local_time_idx)`
index from a single reference surface variable and reused it for all eleven,
assuming identical chunking and identical time axes across variables. That
assumption is **false** — measured, not suspected.

**What the wrong answer looks like.** Two ways, both bad. First, the observed
one: the 1980–2015 train set reported 54,060 samples where the range gives
52,596, because the glob pulled in 2016 — the first *validation* year — so
training silently leaked validation data. Second, the suspected mechanism for
the non-finite validation losses: if file lists can differ by a whole year they
can differ in chunking, and a shared index then reads variable A at timestep *t*
alongside variable B at some other timestep, producing a physically incoherent
sample that looks structurally fine.

**The right handling** — v3, and this is load-bearing:

1. Every variable gets its **own** `{timestamp -> (file_idx, local_idx)}` map,
   built from the `time` coordinate in its own files.
2. Timestamps outside `[start_year, end_year]` are dropped **at build time**,
   enforcing the year range independently of what the glob returned.
3. The sample timeline is the sorted **intersection** of all variables'
   timestamps. A sample is valid only if every timestep it needs
   (`t−6h, t, t+6h, …, t+k·6h`) exists in that intersection at exact 6-hour
   spacing. Gaps remove samples rather than producing misaligned reads.
4. `__init__` prints a per-variable alignment report — files, timesteps,
   dropped-out-of-range, coverage — so data drift is visible in the first 90
   seconds of a job instead of surfacing as NaN eleven hours later.

**In this campaign.** The alignment report and the intersection logic move
verbatim. D3 asserts the exact sample counts from `03_DOMAIN_PRIORS.md`. If you
find yourself simplifying the per-variable index into a shared one because it
"looks redundant", stop — read the 40-line comment at the top of `dataset.py`
first. It exists for you.

### 4.3 The static-variable fallback substitutes zeros

**The trap.** `_load_static_vars` catches `Exception` around the invariant `z`
and `lsm` loads and substitutes **zero tensors**, warning only:

```python
except Exception as e:
    warnings.warn(f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros.")
```

**What the wrong answer looks like.** Training proceeds normally on zero surface
geopotential and a zero land-sea mask. The model learns a planet with no
topography and no continents. Loss curves look unremarkable. Two `UserWarning`
lines scroll past in an 11-hour log.

Because §4.5's HDF5 locking error is exactly the kind of exception this catches,
**it is plausible that the July runs trained on zeroed statics**, compounding
with §4.1. Whether they did is task **B2** — grep the July SLURM logs for
`Using zeros`.

**The right handling.** Hard failure with a message naming the file and the
`HDF5_USE_FILE_LOCKING` fix. Task **E1**. Note that `_clean` — which zeroes
NaN/Inf across all three statics, per its own v3 docstring — is a *different*
thing and is legitimate; do not conflate them.

### 4.4 `slt_data.nc` is not upstream and is not backed up

**The trap.** `slt` (soil type) is a hard dependency of `dataset.py`, defaulting
to `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc` with a `UserWarning` if
`slt_path` is not passed. It is **not** in the LANL archive. It lives on
`/pscratch`, which is **purgeable scratch space**. The only record of how it was
produced is `scripts/download_slt.py`, which is 21 lines.

**What the wrong answer looks like.** Two flavours. Either `/pscratch` purges it
and training dies at dataset init — annoying but loud. Or the `_load_static_vars`
fallback swallows it and §4.3 happens.

**In this campaign.** E1 makes the failure loud. Moving the file somewhere
durable and verifying `download_slt.py` reproduces it is `QUESTIONS.md` Q-07 —
it needs a human decision about where "durable" is.

### 4.5 CFS reads require `HDF5_USE_FILE_LOCKING=FALSE`

**The trap.** `xr.open_dataset` fails on readable, existing files:

```text
OSError: [Errno -101] NetCDF: HDF error: '.../e5.oper.an.sfc.128_167_2t....nc'
```

HDF5 advisory locking against a parallel filesystem. Verified 8 September 2026:
with the variable set, the same file opens cleanly (180×360×124, variable `t2`).

**Where it is currently set.** `slurm_scripts/train_auto.slurm` only. Not in
Python, not in `eval.slurm`, not in `test_train.slurm`, not in
`submit_chain.sh`, and not in an interactive `salloc` unless the human remembers.

**In this campaign.** E1 sets it in `src/aurora_mjo/env.py` at process start,
before the first xarray import, and adds it to every SLURM script. A guard that
lives only in a shell script is not a guard.

---

## 5. What upstream does NOT promise

So nobody builds on an accident.

- **No true MSL field.** There is no variable to swap in. §4.1 is permanent
  until the archive changes.
- **No guarantee that all eleven variables cover the same year range.** The
  intersection in §4.2 is not defensive over-engineering; it is load-bearing.
- **No guarantee of uniform file chunking** across variables or across years.
- **No stability guarantee on the directory layout.** `Step00`–`Step03` and the
  `ERA5.remap_180x360MODIS_6hrInst` stem are another group's preprocessing
  convention. They can change without notice. Everything the code needs from
  them is expressed as a configurable root plus per-variable subpaths, and it
  must stay that way — do not inline the absolute path anywhere.
- **No promise about the `slt` file at all.** §4.4.
- **No promise that `/pscratch` content persists.** It is purgeable. Anything
  that matters gets copied off it. `cleanup-2026-09.md` §8 records that the
  backup bundles were deliberately written to `$HOME`, not `/pscratch`, for
  exactly this reason.