# G4 — **GATE.** Physical-sanity fingerprint at 1°

| | |
| --- | --- |
| **Phase** | G |
| **Depends on** | G1, G2, G3, G5 (all `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 3 h |
| **Compute** | **2 GPU-hours.** One short GPU job. `sbatch` permitted with approval. |
| **Touches** | `tests/fixtures/science-baseline/` (new), `tests/test_physical_ranges.py` (new), `scripts/verify_dataset_loader.py`, `docs/PROJECT_STATE.md`, `03_DOMAIN_PRIORS.md` (§3 only) |
| **Must not touch** | Any file under `src/`. `tests/fixtures/baseline/` or `tests/fixtures/postrefactor/`. Any `results/*.md`. **If you find a bug, you do not fix it — you report `RED`.** |

## Objective

Every field the loader produces at 1° is checked against the physical ranges in
`03_DOMAIN_PRIORS.md` §3, the sample counts are confirmed unchanged, and a new
fixture set is captured so later tasks have something to compare against.

## Why this is a separate task, and why it is a gate

`refactor` had B1/C4: capture behaviour, change code, re-assert behaviour is
identical. That pattern is unavailable here — G1 and G3 change the numbers on
purpose, so equivalence would be a failure rather than a success.

What replaces it is **physical plausibility**. A loader reading the wrong
variable, the wrong level, or the wrong units produces numbers that are perfectly
self-consistent and physically absurd, and nothing downstream would catch it. The
gate asks: is 2-metre temperature actually a temperature?

It is a gate because Phase H and Phase J both build directly on this loader and
both run in parallel. A defect that survives G4 is a defect that surfaces two
weeks later in two places at once.

**This task must not fix anything.** An agent that finds a problem and fixes it
has destroyed the gate — the whole value is an independent check by something
that did not write the code. Report `RED`, name the task that should fix it, stop.

## What you may assume

- G1: native 1°, grid read from the archive (`results/G1_result.md`).
- G2: `model_type: full` available and measured (`results/G2_result.md`).
- G3: true normalisation statistics committed (`results/G3_result.md`).
- G5: SST present as a persisted static (`results/G5_result.md`). It is a fourth
  static and must be range-checked like the other three.
- `03_DOMAIN_PRIORS.md` §3 gives physical ranges. They are **LITERATURE and
  deliberately generous** — a violation means a unit error or a mis-mapped
  variable, not an unusual day.
- `03_DOMAIN_PRIORS.md` §1 gives sample counts that must **not** have moved.

## Steps

1. Load one sample from each of 1980, 1998 and 2015 — spread across the training
   period, including one strong-ENSO year, so a seasonal or decadal artefact has
   a chance to show.
2. For every one of the eleven variables plus the three statics, record: shape,
   dtype, min, max, mean, standard deviation, and non-finite count. Check each
   against `03_DOMAIN_PRIORS.md` §3.
3. Run the three named traps explicitly and state each verdict in one line:
   - `ttr` mean is **negative** (positive means the sign convention flipped and
     every RMM phase will be rotated 180°).
   - surface `z` mean ÷ 9.80665 ≈ **378 m** (if `z` mean itself is ≈ 378, units
     flipped).
   - `q` spans **four orders of magnitude** across the 13 levels (if not, the
     29→13 level subset is mis-indexed).
4. Confirm `lsm` is in [0, 1] and `slt` contains **only integers 0–7** — G1
   regridded it by nearest neighbour, and a non-integer value proves bilinear
   crept back in.
5. Confirm sample counts: **1,462** for 1980, **1,458** for 1981. If either
   moved, Lesson 5 is broken — `RED`, immediately.
6. Normalise each variable with the G3 statistics and confirm the resulting
   σ-values are sane: bulk of the distribution within ±5 σ, worst case within
   ±10 σ. **This is the direct test that Lesson 1 is dead.** Paste the worst
   σ-value per variable and where it occurs.
7. Run one forward pass with `model_type: full` on a real sample. Confirm output
   shapes, finiteness, and that predicted fields are within the same physical
   ranges as the inputs. An untrained injected head will produce poor `ttr` and
   `tcwv` — that is expected; `NaN` is not.
8. Write `tests/fixtures/science-baseline/` with the recorded statistics as JSON,
   mirroring the structure of `tests/fixtures/baseline/`.
9. Write `tests/test_physical_ranges.py` asserting the §3 bounds against the
   synthetic fixtures, so the check runs in CI forever rather than once.
10. Write the discontinuity section comparing every moved number against B1,
    with the reason each moved.

## Definition of Done

- [ ] Full statistics table for 11 variables + 3 statics × 3 sample years pasted
- [ ] Every value inside `03_DOMAIN_PRIORS.md` §3 bounds, **or** the violation
      named, localised to a variable, and the task that should fix it identified
- [ ] Three named traps checked, one-line verdict each
- [ ] `lsm` ∈ [0, 1]; `slt` integer-valued 0–7 only; **SST ∈ [271, 310] K over
      ocean**, and the land-fill value's σ-value pasted
- [ ] Sample counts 1,462 (1980) and 1,458 (1981) confirmed
- [ ] Worst σ-value per variable under G3 statistics pasted; all within ±10 σ;
      Tibetan `ps` case called out explicitly
- [ ] One `model_type: full` forward pass on real data, finite, shapes pasted
- [ ] `tests/fixtures/science-baseline/` committed
- [ ] `tests/test_physical_ranges.py` passing in CI against synthetic fixtures
- [ ] `tests/fixtures/baseline/` and `postrefactor/` **byte-identical**;
      `git diff --stat` on them pasted showing no change
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section covering every moved number vs B1
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing anything.** See above. This is the whole point.
- Touching any file under `src/`.
- Evaluating model *skill*. G4 asks whether the plumbing is physical, not whether
  the model is good. J4 is the skill baseline.
- Updating the historical fixture sets.
