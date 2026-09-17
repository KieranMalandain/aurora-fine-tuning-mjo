# G3 — True 1980–2015 normalisation statistics; Welford into the package

| | |
| --- | --- |
| **Phase** | G |
| **Depends on** | G1 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 3 h plus job wall-clock |
| **Compute** | **CPU only.** One SLURM CPU job over 36 years of CFS data, ~1–2 h. `sbatch` permitted with approval. No GPU. |
| **Touches** | `src/aurora_mjo/stats.py` (new), `scripts/calc_norm_stats.py`, `configs/norm_stats_1980_2015.yaml` (new), `configs/unified.yaml`, `src/aurora_mjo/model.py` (only the `norm_stats` injection), `tests/test_norm_stats.py`, `tests/test_stats.py` (new), `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§4 only) |
| **Must not touch** | `src/aurora_mjo/dataset.py`. `src/aurora_mjo/loss.py`. `src/aurora_mjo/trainer.py`. `src/aurora_mjo/rmm/`. Any `results/*.md`. |

## Objective

Every variable the model normalises has a mean and standard deviation computed
from **this dataset over 1980–2015**, committed with provenance. The placeholder
`msl` constants are gone and are impossible to reintroduce silently. The Welford
accumulator lives in the package and has a unit test.

## Why this is a separate task

`../refactor/handoff-2026-09-refactor.md` §10.1 called this "the single most
critical next action". `00_CONTEXT.md` §5 explains why it is third rather than
first — but it is still mandatory, and it is a prerequisite for H1, which
normalises the loss with exactly these constants.

Keeping it separate from H1 also means the statistics can be checked on their own
terms. A wrong constant and a wrong weight in the same commit is unattributable.

## What you may assume

- G1 landed: data is read at native 1° from the grid the archive uses
  (`results/G1_result.md`).
- The current `msl` override — mean 96,667.9822, std 9,504.6359 — is Aurora's own
  built-in `sp` statistics, **not** computed from this dataset. The config comment
  says so and `../refactor/` Lesson 1 is the incident.
- `tests/test_norm_stats.py::test_placeholder_stats_warning_or_failure` is
  currently `xfail` and exists to track exactly this
  (`../refactor/PROJECT_STATE.md` §3).
- `03_DOMAIN_PRIORS.md` §4 lists Aurora's built-in scales. Your output should
  land near them for variables Aurora also has; a **large** disagreement is a
  finding, not automatically a bug, but a factor-of-two disagreement on a common
  variable means check the accumulator first.

## Steps

1. Extract the Welford accumulation from `scripts/calc_norm_stats.py` into
   `src/aurora_mjo/stats.py`. It is currently non-importable and therefore
   untestable — flagged as an Observation by D3 of the previous campaign.
   `scripts/calc_norm_stats.py` becomes a thin CLI, matching the
   `scripts/`-are-run-never-imported rule.
2. Write `tests/test_stats.py`: streaming Welford against `numpy.std` on a known
   distribution, in chunks, to float64 tolerance. Include a chunk-boundary case
   and a single-element case.
3. Compute statistics over **1980–2015 only** for: `2t`, `10u`, `10v`, `ps`,
   `ttr` (`mtnlwrf`), `tcwv`, and per-level for `z`, `q`, `t`, `u`, `v` across all
   13 Aurora levels. **Per level, not pooled** — `q` spans four orders of
   magnitude from 50 hPa to 1000 hPa and a pooled statistic is meaningless
   (`03_DOMAIN_PRIORS.md` §4).
4. Write `configs/norm_stats_1980_2015.yaml` with a provenance header: date,
   commit SHA, year range, file count, total sample count, and the archive root.
   A statistics file that cannot say what it was computed from is a placeholder
   with better manners.
5. Point `configs/unified.yaml` at it. Remove the placeholder block and the
   PLACEHOLDER comment.
6. In `model.py`, the norm-stat injection currently mutates the module-global
   `aurora.normalisation.locations` / `scales` dicts (`model.py:438`). Aurora
   exposes `surf_stats` as a constructor argument for this. **Switch to
   `surf_stats` for the surface variables.** Process-global mutation leaks across
   pytest runs and is a latent source of order-dependent test failures.
   Atmospheric per-level stats have no equivalent constructor hook — if global
   mutation is unavoidable there, say so explicitly and record it as a limitation.
7. Invert `test_placeholder_stats_warning_or_failure` from `xfail` to a passing
   test asserting the placeholder values (96,667.9822 / 9,504.6359) do **not**
   appear anywhere in `configs/`.
8. Update `03_DOMAIN_PRIORS.md` §4 with the measured values alongside Aurora's,
   and comment on any variable where they diverge by more than ~30%.
9. Sanity-check `ps`: the whole of Lesson 1 is that surface pressure over high
   terrain lands at −36 σ under MSL constants. With true `ps` statistics,
   compute the σ-value of a Tibetan-plateau grid point (~52,000–55,000 Pa) and
   paste it. It should be roughly −4 to −5 σ, not −36.

## Definition of Done

- [ ] `src/aurora_mjo/stats.py` exists; `scripts/calc_norm_stats.py` is a thin CLI
- [ ] `tests/test_stats.py` passes, including chunk-boundary and single-element
      cases; output pasted
- [ ] `configs/norm_stats_1980_2015.yaml` committed with a full provenance header;
      header pasted
- [ ] Per-level statistics present for all five atmospheric variables across all
      13 levels; `q` spread across four orders of magnitude confirmed
- [ ] Placeholder values absent from `configs/`;
      `grep -rn "96667\|9504.6" configs/` returns nothing
- [ ] `test_placeholder_stats_warning_or_failure` inverted from `xfail` to
      passing; before/after pytest lines pasted
- [ ] Surface variables normalised via Aurora's `surf_stats` constructor argument,
      not global dict mutation; any remaining global mutation named and justified
- [ ] Tibetan-plateau σ-value computed and pasted, with the old and new values
- [ ] `03_DOMAIN_PRIORS.md` §4 updated; divergences from Aurora's constants > 30%
      called out
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section written — `msl` statistics old vs new
- [ ] `docs/PROJECT_STATE.md` updated; OPEN-01 closed
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Using these statistics in the loss. That is H1, and it is a different change.
- Computing statistics for `tp6h` or `mslhf` — H3 owns those variables and will
  need its own, in physical units, for the `E − P` target.
- Statistics over any year outside 1980–2015. The test-year quarantine
  (`04_AGENT_PROTOCOL.md` §4) is absolute and the leakage contract
  (`02_SCIENTIFIC_CONTRACT.md` §6.3) forbids fitting anything on validation.
- Changing what the model does with the statistics beyond the `surf_stats`
  mechanism swap.
