# Domain priors

Expectations to sanity-check a measurement against. **Priors, not truth** — the
data wins. But a large disagreement means check your query before you believe
your result.

Provenance is recorded for every row. A prior with no provenance is a guess
wearing a table. Rows marked **MEASURED** come from committed output in this
repo and are the strongest kind; rows marked **DERIVED** are arithmetic; rows
marked **LITERATURE** are textbook geophysics and are the weakest.

---

## 1. Sample counts — exact, and the campaign's best regression test

These are arithmetic, not estimates. Data is 6-hourly instantaneous: **4
timesteps per day**. A single-step training sample needs three consecutive
timesteps (`t−6h`, `t`, `t+6h`), so

```text
timesteps = days_in_range × 4
samples   = timesteps − (k + 1)      where k = max_rollout_steps
```

| Range | Years | Leaps | Days | Timesteps | Samples (k=1) | Provenance |
| --- | --- | --- | --- | --- | --- | --- |
| 1980–2015 (train) | 36 | 9 | 13,149 | **52,596** | 52,594 | DERIVED; 52,596 confirmed in `dataset.py` header |
| 2016–2019 (val) | 4 | 1 | 1,461 | **5,844** | 5,842 | DERIVED |
| 2020–2023 (test) | 4 | 1 | 1,461 | **5,844** | 5,842 | DERIVED |
| 1980 only | 1 | 1 | 366 | **1,464** | **1,462** | **MEASURED** — `docs/verify_output.txt` |
| 2000 only | 1 | 1 | 366 | 1,464 | 1,462 | DERIVED |
| 1980–2016 | 37 | 10 | 13,515 | **54,060** | — | **MEASURED as the bug** — see below |

**The 1980 row is the anchor.** `docs/verify_output.txt` reports exactly 1,462
samples for 1980, and 1,464 − 2 = 1,462. The formula is confirmed against real
output, not assumed.

**The last row is Lesson 5 in numbers.** The pre-v3 loader reported 54,060
samples for a range requested as 1980–2015. 54,060 is exactly 1980–**2016**, and
2016 is the first validation year. If any count you measure equals 54,060 for a
train split, the year filter has failed and you are training on validation data.

**If you measure fewer samples than the formula gives,** that is not necessarily
a bug — the v3 timeline is the *intersection* of all eleven variables'
timestamps, so a gap in any one variable legitimately removes samples. It is
information: record the shortfall and which variable's coverage caused it. The
per-variable alignment report printed at dataset init tells you.

**If you measure more,** something is wrong. The intersection cannot exceed the
range.

---

## 2. Grid, shapes and dtypes — all MEASURED

From `docs/verify_output_slt.txt`, which is the current-state verify run.
Use these as exact assertions in tests.

| Quantity | Value |
| --- | --- |
| Native grid | 180 × 360 (1°) |
| Aurora grid | 720 × 1440 (0.25°) |
| Pressure levels | 13: `(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)` hPa |
| `metadata.lat` | `torch.Size([720])` |
| `metadata.lon` | `torch.Size([1440])` |
| Input surface var | `[1, 2, 720, 1440]` — batch, **2 timesteps**, lat, lon |
| Input atmos var | `[1, 2, 13, 720, 1440]` |
| Target surface var | `[1, 720, 1440]` — no time axis |
| Target atmos var | `[1, 13, 720, 1440]` |
| Static var | `[720, 1440]` |
| First sample init time, 1980 | `1980-01-01 06:00:00` (not 00:00 — index 0 needs `t−6h`) |

The 2-timestep input axis is Aurora's requirement, not a choice. The absent time
axis on targets is why `collate_fn` exists.

---

## 3. Static variables — MEASURED, and one live trap

From `docs/verify_output_slt.txt`:

| Field | Mean | Std | Units | Interpretation |
| --- | --- | --- | --- | --- |
| `z` | 3709.2466 | 8216.3809 | m² s⁻² | surface geopotential |
| `lsm` | 0.3357 | 0.4484 | fraction | land-sea mask |
| `slt` | 0.6708 | 1.1682 | category 0–7 | soil type |

Each of these is checkable, and the checks are worth writing down:

- **`z` is geopotential, not geopotential height.** 3709 / 9.80665 ≈ **378 m**
  mean elevation on an unweighted 1° grid. That is plausible (ocean at 0, land
  averaging ~840 m, and high-latitude cells over-represented in an unweighted
  mean by Antarctica). If it were metres, a 3,709 m global mean elevation would
  be absurd. **A factor-of-9.81 error here would be silent and catastrophic** —
  if you ever see `z` mean ≈ 378, the units flipped.
- **`lsm` 0.3357 matches Earth's ~29% land fraction** inflated by the unweighted
  grid. A value near 0.29 would also be fine; a value near 0.0 means Lesson 4.
- **`slt` is categorical 0–7** (0 = ocean/no soil). Mean 0.67 with std 1.17 is
  the signature of a mostly-zero field with land values in 1–7 — right shape.

### The trap: `docs/verify_output.txt` is stale and shows a broken state

The two committed verify outputs differ on exactly one line:

```text
verify_output.txt      slt: shape=[1, 720, 1440], mean=0.0000, std=0.0000
verify_output_slt.txt  slt: shape=[720, 1440],    mean=0.6708, std=1.1682
```

`verify_output.txt` is the **pre-soil-type-work** run and shows `slt` fully
zeroed. It is superseded, it is not labelled as superseded, and anyone reading it
today would conclude `slt` is broken. **Task F1 archives it** under
`docs/archive/` with a header saying what it superseded. Do not delete it — it is
the before-picture of a fixed bug.

### A structural asymmetry worth knowing before you touch `_load_static_vars`

`z` and `lsm` come from the CFS archive at 1° and go through
`_upsample_to_aurora`. `slt` comes from `slt_data.nc` — a different, non-archive
file (`02_UPSTREAM_CONTRACT.md` §4.4) — is **already at 0.25°**, and is
truncated with `_ensure_2d(...)[:720, :]` rather than upsampled. This looks
inconsistent because the two sources genuinely differ. It is not a bug. Do not
"unify" it.

---

## 4. Normalisation constants — the arithmetic behind Lesson 1

Aurora's built-in constants versus what the data needs:

| Channel | Aurora built-in | Config override (PLACEHOLDER) | Provenance |
| --- | --- | --- | --- |
| `msl` | location 100,958, scale 1,332 | mean 96,667.98, std 9,504.64 | Aurora's own `sp` stats |
| `ttr` | — | mean −226.0498, std 49.2158 | `configs/unified.yaml` |
| `tcwv` | — | mean 18.2967, std 16.3265 | `configs/unified.yaml` |

**Do the arithmetic once and it stops being abstract.** Tibetan-plateau surface
pressure at ~5,300 m is about 52,000–55,000 Pa.

```text
under Aurora's msl constants:  (52,000 − 100,958) / 1,332  = −36.8 σ
under the sp override:         (55,000 −  96,668) / 9,505  =  −4.4 σ
```

That is the whole of Lesson 1. Note **both** terms are wrong, not just one: the
location is ~4,300 Pa too high *and* the scale is **7× too small**, because MSL
varies by ~1,300 Pa globally while surface pressure varies by ~9,500 Pa once you
include topography. Any replacement stats must fix both.

**A prior on the replacement values.** Real `ps` computed over 1980–2015 should
land near mean 96,000–99,000 Pa and std 9,000–10,000 Pa. If a computed std comes
back near 1,300 you have computed MSL or masked out land. If a computed mean
comes back near 101,325 you have computed MSL.

**The area-weighting trap, which will bite whoever computes these.** An
unweighted mean over a regular 180 × 360 lat/lon grid over-weights the poles,
because cells have equal count but shrinking area. `tcwv`'s override of 18.30
kg m⁻² sits well below the accepted global mean of **24–25 kg m⁻²** (LITERATURE),
and unweighted polar over-representation is the most likely explanation — dry
polar air dragging the mean down. Before replacing any constant, **state whether
your statistic is area-weighted by cos(latitude) or not**, and match whatever
convention Aurora's own constants use. Getting this wrong shifts every input by a
fraction of a sigma across the whole globe, uniformly, invisibly.

---

## 5. Geophysical ranges — LITERATURE, for smell-testing a field

Global values unless stated. These exist so that "is this field plausible" has an
answer that is not a vibe.

| Field | Units | Expected | If it disagrees |
| --- | --- | --- | --- |
| `2t` | K | ~288 area-weighted; ~278–282 unweighted | <200 or >330 → units or fill values |
| `msl` (really `ps`) | Pa | 96,000–99,000 mean; 50,000–105,000 range | see §4 |
| `ttr` (`mtnlwrf`) | W m⁻² | **negative**, ~−225 to −240 | positive → sign convention flipped |
| `tcwv` | kg m⁻² | 24–25 weighted; 40–60 tropics; <5 polar | negative → fill value leaked |
| `q` @1000 hPa | kg kg⁻¹ | ~0.010–0.018 tropics | >0.1 → g/kg not kg/kg |
| `q` @200 hPa | kg kg⁻¹ | ~1e−5–1e−6 | exactly 0 → clipped |
| `t` @500 hPa | K | ~253 | |
| `t` @200 hPa | K | ~215–220 | |
| `z` @500 hPa | m² s⁻² | ~54,000 (≈5,500 gpm) | ~5,500 → gpm not m²s⁻² |
| `z` @1000 hPa | m² s⁻² | ~1,000 | |
| `u` @200 hPa | m s⁻² | jets 30–50; tropics −10 to +10 | |
| `u` @850 hPa tropics | m s⁻¹ | easterlies, ~−5 | |

**The `z` rows are the ones to actually check.** Both surface `z` (§3) and
pressure-level `z` are geopotential in m² s⁻², and both have a plausible-looking
geopotential-height twin exactly 9.80665× smaller. This is the classic silent
corruption in this domain.

---

## 6. MJO priors — for evaluation work

| Quantity | Expected | Provenance | If it disagrees |
| --- | --- | --- | --- |
| RMM1, RMM2 | standardised, unit variance by construction | LITERATURE (Wheeler & Hendon 2004) | std ≠ ~1 → normalisation step missed |
| Amplitude | `sqrt(RMM1² + RMM2²)` | definition | |
| "Active MJO" | amplitude > 1 | convention | |
| Fraction of days active | ~50–60% | LITERATURE | ≪40% or ≫70% → normalisation wrong |
| Phase | 1–8 from `atan2(RMM2, RMM1)`, eastward | convention | check the phase-1 origin and the rotation sense; an off-by-one-phase error is easy and invisible |
| Period | 30–60 days (20–100 broad) | LITERATURE | |
| Eastward phase speed | ~5 m s⁻¹ | LITERATURE | |
| OLR anomaly amplitude, tropics | ±20–40 W m⁻² | LITERATURE | |

**On the project target.** `r > 0.5` at 30-day lead is the stated goal. For
calibration: operational dynamical models were reaching `r = 0.5` at roughly
**22–27 days** as of the mid-2020s. The target is therefore at or beyond the
state of the art. That is fine as an ambition, but if an early evaluation reports
`r = 0.5` at 30 days, **suspect a leak before celebrating** — the most likely
causes are a validation year in the training range (Lesson 5) or RMM targets
computed using statistics from the full period rather than the training period
only. Both have happened in this domain.

**Do not** compute RMM normalisation, EOFs, or climatology over the full record
and then evaluate on held-out years. That is leakage, and
`.agent/rules/02-research-standards.md` already prohibits it: chronological
splits only, train-period normalisation only.

---

## 7. Compute and memory — MEASURED, from `tools/probe_results/`

| Quantity | `small` | `huge` |
| --- | --- | --- |
| Status | OK | **OOM** |
| Peak memory | 39.71 GiB | 78.41 GiB (of 79.2) |
| Mean step time | 0.694 s | — |
| Steps completed | 30 | 3 |
| Trainable params | 41,008 | 81,968 |

Priors that follow, and one alarm:

- **`small` fits in ~40 GiB of 79.2**; `huge` does not fit at all without
  gradient checkpointing, which crashes this machine (Lesson 6). This is the
  evidence base for `model_type: "small"`.
- **Trainable parameter counts are tiny — 41k and 82k** — because
  `freeze_backbone` freezes the whole Aurora backbone and leaves only the MJO
  head and/or LoRA adapters trainable. If a fingerprint measurement reports
  trainable params in the millions, freezing has broken. If it reports the same
  number as total params, `freeze_backbone` did not run at all.
- **`small` recorded `nonfinite_grad_skips: 30` across `steps_completed: 30`.**
  **Every single step was skipped.** The grad guard (Lesson 2) worked perfectly
  and the run therefore learned exactly nothing. Treat
  `nonfinite_grad_skips == steps` as a **red alarm, not a pass** — it means the
  guard is holding back a completely broken forward pass, most likely Lesson 1.
  Any future probe or smoke test must report this ratio, and F2's acceptance
  criteria include it.
- ~0.69 s/step on 4 × A100 for `small`, single-step. Useful for sanity-checking
  a walltime estimate; do not treat it as a benchmark, it was measured under
  unknown contention.