# Domain priors

Expectations to sanity-check a measurement against. **Priors, not truth** — the
data wins. But a large disagreement means check your query before you believe
your result.

Provenance is recorded for every row. A prior with no provenance is a guess
wearing a table. **MEASURED** comes from committed output or a run recorded in
this repo; **DERIVED** is arithmetic; **LITERATURE** is textbook geophysics or a
published number and is the weakest — every LITERATURE row in this file is a row
an agent should be willing to contradict with data.

`../refactor/03_DOMAIN_PRIORS.md` is **not superseded**. Its §1 (sample counts),
§3 (static variables), §5–§6 and §8 still hold. This file carries forward what
changed and adds what is new.

---

## 1. Sample counts — unchanged, still the best regression test

Carried forward from `../refactor/03_DOMAIN_PRIORS.md` §1 **without change**.
Task G1 changes the *resolution* of what is read, not the timeline logic, so
these must come out identical after G1. If they do not, G1 broke Lesson 5.

```text
timesteps = days_in_range × 4
samples   = timesteps − (k + 1)      where k = max_rollout_steps
```

| Range | Days | Timesteps | Samples (k=1) | Provenance |
| --- | ---: | ---: | ---: | --- |
| 1980–2015 (train) | 13,149 | 52,596 | 52,594 | DERIVED |
| 2016–2019 (val) | 1,461 | 5,844 | 5,842 | DERIVED |
| 2020–2023 (test) | 1,461 | 5,844 | 5,842 | DERIVED — **not read by this campaign** |
| 1980 only | 366 | 1,464 | **1,462** | **MEASURED** — B1, D1, F2 |
| 1981 only | 365 | 1,460 | **1,458** | **MEASURED** — D1, D3 |
| 1980–2016 | 13,515 | 54,060 | — | **MEASURED as the bug.** Lesson 5. |

**54,060 in a train split still means the year filter has failed.** That number
has not stopped being a tripwire.

Phase K adds a new failure mode to watch: `max_rollout_steps` rises during the K3
curriculum, so sample count **falls** as `k` rises, by exactly `k − 1` per
increment. A curriculum step that does not reduce the sample count has not taken
effect.

---

## 2. Grid and shapes — **all of these change in G1**

The old table is at `../refactor/03_DOMAIN_PRIORS.md` §2 and describes the
0.25° state. After G1:

| Quantity | Before G1 | After G1 | Provenance |
| --- | --- | --- | --- |
| Grid fed to Aurora | 720 × 1440 | **180 × 360** | D1 |
| Grid points per field | 1,036,800 | **64,800** | DERIVED — 16.0× fewer |
| `metadata.lat` | `[720]`, 0.2503° spacing | `[180]`, read from archive | D1 |
| `metadata.lon` | `[1440]` | `[360]`, read from archive | D1 |
| Patch grid `(L, Hp, Wp)` | `(4, 180, 360)` | **`(4, 45, 90)`** | **MEASURED** — live forward pass, `00_CONTEXT.md` R5 |
| Input surface var | `[1, 2, 720, 1440]` | `[1, 2, 180, 360]` | D1 |
| Input atmos var | `[1, 2, 13, 720, 1440]` | `[1, 2, 13, 180, 360]` | D1 |
| Target surface var | `[1, 720, 1440]` | `[1, 180, 360]` | D1 |
| Target atmos var | `[1, 13, 720, 1440]` | `[1, 13, 180, 360]` | D1 |
| Static var | `[720, 1440]` | `[180, 360]` | D1 |
| Backbone `window_size` | `(2, 6, 12)` | `(2, 6, 12)` — unchanged | **MEASURED** |

Two things that do **not** change: the 2-timestep input axis (Aurora's
requirement) and the absent time axis on targets (which is why `collate_fn`
exists). First sample init time for 1980 is still `1980-01-01 06:00:00`.

**45 × 90 is not a multiple of `(6, 12)`.** Aurora's Swin3D pads to `48 × 96` and
crops back (`aurora/model/swin3d.py:350`). This is expected and was confirmed by
forward pass. An agent who sees padding in a profile has not found a bug.

### 2.1 The archive grid convention — **measured in G1**

Measured from the CFS archive reference files across all eleven variables and invariants (`remap_180x360MODIS`):

- **Latitudes**: Cell-centred, 180 points from -89.5 to 89.5 with 1.0° regular spacing.
  - Raw archive file array (ascending): `[-89.5, -88.5, -87.5, -86.5, -85.5]` ... `[85.5, 86.5, 87.5, 88.5, 89.5]`.
  - Inverted for Aurora / physical North-to-South alignment (descending, `lat[1:] - lat[:-1] < 0`): `[89.5, 88.5, 87.5, 86.5, 85.5]` ... `[-85.5, -86.5, -87.5, -88.5, -89.5]`.
- **Longitudes**: Cell-centred, 360 points from 0.5 to 359.5 with 1.0° regular spacing.
  - Raw archive file array (ascending): `[0.5, 1.5, 2.5, 3.5, 4.5]` ... `[355.5, 356.5, 357.5, 358.5, 359.5]`.
- **Agreement**: All eleven variables match across all coordinates to float32 exactness.
- **Orientation**: Aurora's `Metadata` strictly enforces decreasing latitude, and Aurora expects North at row 0. `dataset.py` inverts `lat` to descending and flips data arrays along the latitude dimension (`axis=-2`).

---

## 3. Physical ranges — the G4 sanity gate

The replacement for a behavioural fingerprint. After G1 the numbers change on
purpose, so equivalence is meaningless; physical plausibility is not.

All rows **LITERATURE** unless marked. Ranges are global, all seasons, and
deliberately generous — a violation means a unit error or a mis-mapped variable,
not an unusual day.

| Variable | Units | Global mean, plausible | Hard bounds |
| --- | --- | --- | --- |
| `2t` | K | 285 – 290 | 180 – 340 |
| `10u`, `10v` | m s⁻¹ | −1 – 1 | −110 – 110 |
| `msl` (= `ps`) | Pa | 96,000 – 99,000 | 47,000 – 108,000 |
| `ttr` (= `mtnlwrf`) | W m⁻² | −250 – −215 | −400 – −50 |
| `tcwv` | kg m⁻² | 24 – 26 | 0 – 100 |
| `q` @ 1000 hPa | kg kg⁻¹ | 0.008 – 0.012 | 0 – 0.04 |
| `q` @ 50 hPa | kg kg⁻¹ | ~2.5 × 10⁻⁶ | 0 – 10⁻⁵ |
| `t` @ 500 hPa | K | 250 – 256 | 200 – 290 |
| `z` @ 500 hPa | m² s⁻² | 54,000 – 55,500 | 45,000 – 60,000 |
| `u` @ 200 hPa | m s⁻¹ | 8 – 15 | −80 – 120 |

Three that catch specific, silent errors:

- **`ttr` must be negative.** It is net top long-wave flux, downward-positive.
  A positive mean means the sign convention flipped and every RMM phase will be
  rotated by 180°. `configs/unified.yaml` records mean −226.05, consistent
  (**MEASURED**).
- **`z` is geopotential, not geopotential height.** Surface `z` mean 3709.2466
  m² s⁻² ÷ 9.80665 ≈ 378 m mean elevation (**MEASURED**, `verify_output_slt.txt`).
  If you ever see surface `z` mean ≈ 378, the units flipped and the error is
  silent.
- **`q` spans four orders of magnitude across the 13 levels.** A `q` field whose
  levels are all within one order of each other has been mis-indexed — see
  `_probe_pressure_level_indices` and the 29→13 level subset.

---

## 4. Normalisation scales — measured in G3 (1980–2015, 1° ERA5)

True training-period (1980–2015) normalisation statistics computed across all 4,752 files (3,408,220,800
samples per field) at native 1° resolution via Welford accumulation (`configs/norm_stats_1980_2015.yaml`).

| Variable | Aurora Built-in σ | G3 Measured σ | Rel Diff (%) | Provenance & Notes |
| --- | ---: | ---: | ---: | --- |
| `2t` | 21.220 | **21.2352** | +0.07% | **MEASURED** — G3 Welford |
| `10u` | 5.548 | **5.4905** | -1.04% | **MEASURED** — G3 Welford |
| `10v` | 4.765 | **4.7146** | -1.06% | **MEASURED** — G3 Welford |
| `msl` (= `ps`) | 1,332.246 | **9,504.4086** | **+613.41%** | **MEASURED** — **Lesson 1 proxy divergence** (>30% threshold). Archive supplies `ps`, not MSL. |
| `z` (13-level mean) | 3,828.21 | **3,827.7773** | -0.01% | **MEASURED** — G3 per-level mean |
| `t` (13-level mean) | 12.378 | **12.3761** | -0.02% | **MEASURED** — G3 per-level mean |
| `u` (13-level mean) | 12.748 | **12.7057** | -0.33% | **MEASURED** — G3 per-level mean |
| `v` (13-level mean) | 8.911 | **8.8704** | -0.46% | **MEASURED** — G3 per-level mean |
| `q` (13-level mean) | 0.0016364 | **0.0016362** | -0.01% | **MEASURED** — G3 per-level mean |
| `ttr` | — | **49.2100** | — | **MEASURED** (mean = -226.0305 W m⁻²) |
| `tcwv` | — | **16.3142** | — | **MEASURED** (mean = 18.2790 kg m⁻²) |

### Evaluation of Divergences:
1. **Unmodified ERA5 variables**: All standard variables (`2t`, `10u`, `10v`, `z`, `t`, `u`, `v`, `q`) reproduce Aurora's built-in scales to within **1.06%** relative deviation. The Welford accumulator is exact and fully concordant.
2. **`msl` divergence**: Exceeds the 30% threshold significantly (+613.41%), as predicted by Lesson 1. Aurora's built-in 1332.25 is sea-level-pressure calibrated; this archive provides surface pressure (`ps` mean 96,668.75 Pa, std 9,504.41 Pa), which varies dramatically across surface orography.
3. **Atmospheric `q` spread**: Confirmed spanning four orders of magnitude from **`3.59 × 10⁻⁷`** kg kg⁻¹ at 50 hPa to **`5.90 × 10⁻³`** kg kg⁻¹ at 1000 hPa (a 16,418× spread).

Placeholder constants (96,667.9822 / 9,504.6359) in `configs/unified.yaml` are **fully retired** and replaced by true values.

---

## 5. The gradient-share table — the R1 baseline, and the H1 acceptance test

Share of `L_grid` gradient per variable under the **current** physical-units
loss, assuming equal normalised error across variables. For an L1 loss this is
exact: the gradient budget per variable is proportional to its physical σ.

| Variable | σ (physical) | Share, **before** H1 | Share, **after** H1 (target) |
| --- | ---: | ---: | ---: |
| `msl` | 9,504.64 | **70.59%** | 5.9% |
| `z` | 3,828.21 | **28.43%** | 5.9% |
| `ttr` | 49.22 | 0.37% | 23.5% |
| `2t` | 21.22 | 0.16% | 5.9% |
| `tcwv` | 16.33 | 0.12% | 23.5% |
| `u` | 12.75 | 0.09% | 11.8% |
| `t` | 12.38 | 0.09% | 11.8% |
| `v` | 8.91 | 0.07% | 11.8% |
| `10u` | 5.55 | 0.04% | 5.9% |
| `10v` | 4.77 | 0.04% | 5.9% |
| `q` | 0.0016 | **0.000012%** | 23.5% |

Before-column **MEASURED** (review 2026-09-13, from `aurora/normalisation.py`).
After-column **DERIVED** — it is `w_v / Σw_v` for the weights in
`02_SCIENTIFIC_CONTRACT.md` §4.1, with the atmospheric weights split across the
three atmospheric entries. `tests/test_loss.py` asserts the after-column within
tolerance; the same test run against the pre-H1 loss must **fail**, and H1's
result file must show that it does.

`msl` : `q` before H1 is **5,808,226 : 1** (**MEASURED**). That ratio is the
single number that best explains why six months of training produced no MJO
skill.

---

## 6. Compute — what 1° costs (MEASURED in G2)

| Quantity | Value | Provenance |
| --- | --- | --- |
| `small` peak memory, 0.25°, bs 1 | 39.71 GiB | **MEASURED** — `../refactor/03_DOMAIN_PRIORS.md` §7 |
| `huge` peak memory, 0.25°, no ckpt | 78.41 GiB (**OOM**) | as above |
| `small` mean step time, 0.25° | 0.694 s | as above |
| Grid-point reduction, 0.25° → 1° | **16.0×** | DERIVED |
| `small` peak memory, 1°, ckpt off | **3.12 GiB** | **MEASURED** — G2 benchmark (Perlmutter A100-SXM4-80GB) |
| `small` peak memory, 1°, ckpt on | **2.04 GiB** | **MEASURED** — G2 benchmark |
| `small` mean step time, 1°, ckpt off | **77.2 ms** (0.077 s) | **MEASURED** — G2 benchmark |
| `small` mean step time, 1°, ckpt on | **110.1 ms** (0.110 s) | **MEASURED** — G2 benchmark |
| `full` peak memory, 1°, bs 1, ckpt off | **14.69 GiB** | **MEASURED** — G2 benchmark |
| `full` peak memory, 1°, bs 1, ckpt on | **12.52 GiB** | **MEASURED** — G2 benchmark |
| `full` mean step time, 1°, ckpt off | **221.4 ms** (0.221 s) | **MEASURED** — G2 benchmark |
| `full` mean step time, 1°, ckpt on | **301.8 ms** (0.302 s) | **MEASURED** — G2 benchmark |

### Evaluation of Prior WEAK Estimates:
1. **`full` peak memory**: Prior estimate was ~5–12 GiB. Measured is **14.69 GiB** (ckpt off, 22% higher than upper bound) and **12.52 GiB** (ckpt on, 4% above upper bound). Model parameters (1.26B floats) + optimizer state + heads account for ~5 GB static allocation, with activations scaling cleanly. At 14.69 GiB on an 80 GiB A100 card, there is **81.6% memory headroom** (65.3 GiB unused).
2. **`full` mean step time**: Prior estimate was ~1–3 s/step. Measured is **221.4 ms** (ckpt off) and **301.8 ms** (ckpt on). The prior estimate was **wrong by 4.5×–13.5×** (overly pessimistic). Windowed Swin3D attention at 1° is exceptionally fast on A100.

### Lesson 6 Verdict:
**The gradient checkpointing constraint has dissolved.**
- At 1°, `full` fits with 81.6% free VRAM (14.69 GiB peak) without checkpointing.
- Furthermore, gradient checkpointing was re-tested under 1° on `full` (12.52 GiB peak, 301.8 ms) and ran 30 steps with **zero errors** (no illegal memory access). The IMA crash observed at 0.25° did not occur under the 16× reduced patch resolution.
- However, since checkpointing adds 36% runtime overhead (301.8 ms vs 221.4 ms) and memory headroom is massive without it, `gradient_checkpointing: false` is optimal and fully unblocked.

---

## 7. RMM priors — for the J2 gate

All **LITERATURE**, all from Wheeler & Hendon (2004) and standard practice. Every
one of these is a thing J2 can check, and a mismatch localises the bug.

| Quantity | Expected | What a mismatch means |
| --- | --- | --- |
| Variance explained, EOF1 + EOF2 | ~25% combined, roughly equal split | Very unequal → the pair is not a propagating mode; likely missing the 120-day removal |
| RMM1, RMM2 variance | 1.0 each, by construction | ≠ 1 → PC normalisation (step 9) missing |
| Correlation RMM1 vs RMM2 | ≈ 0 | ≠ 0 → basis not orthonormal, or the projection is wrong |
| Lag between RMM1 and RMM2 | ~10–12 days, in quadrature | No lag → propagation is absent; the classic signature of R3 |
| Fraction of days with `A > 1` | ~50% | ≈ 0% or ≈ 100% → normalisation is off by a large factor |
| Mean period of a full 8-phase cycle | 30–60 days | Far outside → the harmonics or the 120-day removal are wrong |
| **BoM correlation, 2016–2019** | **`r > 0.95` for both RMM1 and RMM2** | **The gate.** Below 0.95, J2 fails and the campaign stops. |

If J2 fails, the diagnostic order is: (1) OLR sign, (2) EOF1/EOF2 swap or sign,
(3) missing 120-day mean, (4) coarsen to 2.5° before the EOF. In that order —
(1) and (2) are single-character fixes and account for most failures.

---

## 8. `E − P` priors — for H3

**LITERATURE.** Sanity bounds for the supervised moisture term.

| Quantity | Expected |
| --- | --- |
| Tropical evaporation `E`, warm ocean | 4 – 6 mm/day |
| Precipitation `P`, MJO active envelope | 20 – 40 mm/day |
| `E − P`, MJO active envelope | **−15 to −35 mm/day** |
| `E − P`, tropical suppressed phase | +2 to +5 mm/day |
| `E − P`, tropical-band spatial mean | small, **not zero** |
| Global, annual `E − P` | ≈ 0 |

**The middle row is R2 in one line.** The current loss drives a quantity whose
true value is −15 to −35 toward 0, in exactly the regions and phases the project
exists to predict.

**The diagnostic that decides H3's fate:** compute the current
`R = ∂⟨q⟩/∂t + ∇·⟨vq⟩` from *ERA5 fields alone* and compare it to ERA5's own
`E − P`. If they agree to within a factor of ~2 over the tropical band, the
numerics are sound and the supervised term is worth training on. If the residual
is dominated by discretisation noise, the term is retired and that is a clean
result. **Run this before writing the loss**, not after.

---

## 9. Skill priors — what a result should look like

**LITERATURE and weak.** Included so an implausible result is recognised as
implausible. Every number here is from memory and **must be verified against the
source paper before it appears in any output** — `02_SCIENTIFIC_CONTRACT.md` §1.3
and **Q-12**.

| System | Bivariate ACC = 0.5 crossing | Note |
| --- | --- | --- |
| Climatology (`RMM ≡ 0`) | 0 days | The floor |
| Persistence | ~5–8 days | |
| Damped persistence | ~8–12 days | The real floor to beat |
| Operational ensemble means | ~25–36 days | **Ensemble**. Not comparable to a single member. |
| Published ML systems | ~20–30 days | Mostly **ensemble**; check member count before comparing |
| **This campaign, target T3** | **≥ 20 days**, stretch ≥ 25 | Single member, `A(t₀) > 1`, 2016–2019 |

**An ACC above ~0.8 at day 20 from a single deterministic member is not a
success, it is a bug.** The first three things to check: 120-day mean leakage
across the initialisation (`02_SCIENTIFIC_CONTRACT.md` §6.2), test-year
contamination, and whether the "forecast" RMM is being computed from observed
rather than predicted fields.

Amplitude prior: `Â/A_obs` for a deterministic model decays roughly
exponentially. **0.5–0.7 at day 20** would be normal. Below ~0.3 the model has
stopped forecasting the MJO regardless of what ACC says
(`02_SCIENTIFIC_CONTRACT.md` §1.4).


---

## 10. Propagation priors — for the J6 gate

All **LITERATURE** and all **unverified** under **Q-12**. Standard references,
from memory: Wheeler & Kiladis (1999) for space–time spectral analysis, and the
CLIVAR MJO Working Group (2009) diagnostics package. Check them before either
appears in an output.

| Quantity | Observed | What a mismatch means |
| --- | --- | --- |
| MJO zonal wavenumbers | 1–3 | Power at wavenumber ≥ 5 in the MJO band is not the MJO |
| MJO period band | 30–60 days | |
| Eastward phase speed | **~5 m s⁻¹** (4–8) | Too slow → stalling; too fast → a Kelvin wave, not the MJO |
| **Eastward / westward power ratio, MJO band** | **~2–3** | **< 1.5 → the forecast is a standing oscillation. T5 fails.** |
| Net RMM phase advance | **~7.5° day⁻¹** (a 48-day cycle) | Near zero with sign reversals → standing |
| Fraction of cases with a phase-advance sign reversal | low in observations | High → standing, and this is the cleaner signature |
| MC crossing: phase 2–3 init reaching phase 5–6 with `A > 1` within 20 days | roughly half of active events | Near zero → the Maritime Continent barrier |

**The single number that matters is the eastward/westward ratio.** A model can
post a good bivariate ACC with a ratio near 1.0 — `02_SCIENTIFIC_CONTRACT.md`
§1.6 explains why, and why a deterministic model trained on a pointwise loss is
the kind of model that does it.

**Implementation trap:** the sign convention in the 2-D FFT determines which
half-plane is "eastward". Get it backwards and every conclusion in J6 inverts
while all the numbers still look plausible. The synthetic-wave test in J6 step 2
exists for exactly this and is not optional.

---

## 11. ENSO and regime inventory — for J7

**LITERATURE** for the thresholds, **DERIVED** for the split composition, and the
split rows are the ones worth reading.

| Quantity | Value | Provenance |
| --- | --- | --- |
| ONI El Niño threshold | ≥ +0.5 °C, 5 consecutive overlapping seasons for an *event* | LITERATURE |
| ONI La Niña threshold | ≤ −0.5 °C, same | LITERATURE |
| Neutral fraction of months, 1980–2019 | roughly 40–50% | LITERATURE, **weak** — J7 measures it |
| El Niño events, 1980–2015 | roughly 10–12 | LITERATURE, weak |
| EP vs CP split of those | roughly half each | LITERATURE, **very weak.** ~5–6 events per flavour is **underpowered for any skill significance claim** |

### 11.1 The splits are not ENSO-balanced

| Split | ENSO character | Consequence |
| --- | --- | --- |
| 1980–2015 (train) | Full range, includes 1982–83, 1997–98 and 2015–16 strong El Niño | Adequate coverage |
| 2016–2019 (val) | Decay of the very strong 2015–16 El Niño, then weak La Niña, then weak El Niño | Reasonably mixed |
| 2020–2023 (test) | **Triple-dip La Niña** — one of three on record | **Strongly atypical.** A limitation of any future test-set claim |

**This is documented now and not acted on.** The test years are quarantined
(`04_AGENT_PROTOCOL.md` §4) and the classification comes from the **ONI record**,
not from model output or ERA5 fields, so documenting it breaks no rule. It belongs
in the paper's limitations whether or not the campaign does anything about it.

### 11.2 Season

MJO behaviour is seasonally distinct. Boreal winter (NDJFMA) carries the canonical
eastward-propagating MJO. Boreal summer (MJJASO) is dominated by the **boreal
summer intraseasonal oscillation**, which propagates *northward* over the Asian
monsoon region and which RMM — an all-season index — represents poorly.

**A summer skill deficit may therefore be an index artefact rather than a model
failure.** J7 must say which it believes and why. BSISO indices exist for this and
are out of scope (`02_SCIENTIFIC_CONTRACT.md` §7).
