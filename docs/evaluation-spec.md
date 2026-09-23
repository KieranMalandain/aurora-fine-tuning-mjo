# Evaluation Specification

## Purpose

This document defines the authoritative evaluation protocol for MJO forecasting in this repository.

Agents must follow this document when implementing or modifying evaluation code.

## Core Principle

The repository's scientific target is MJO forecast skill, not just generic atmospheric reconstruction quality.

Therefore, evaluation must include explicit MJO phase-space metrics, not only gridpoint losses.

## Dataset Splits

Use chronological splits only.

- Train: 1980–2015 (36 years, 13,149 days)
- Validation: 2016–2019 (4 years, 1,461 days)
- Test: 2020–2023 (4 years, quarantined; not processed or evaluated in this campaign)

These years are confirmed as the defaults used in `compute_rmm.py`.
Customize via the `TRAIN_YEARS`, `VAL_YEARS`, `ACTIVE_YEARS` constants in that script.

## Normalization Rules

- Compute normalization statistics using training years only (1980–2015).
- Do not use validation or test years to compute train-time statistics.
- Climatological references and EOF bases are derived exclusively using training-period references.

---

## RMM Targets & Pipeline Specification

> [!IMPORTANT]
> **Historical Correction Note (Task J1, 2026-09-18):**
> Prior versions of this specification and `compute_rmm.py` specified a 3-element combined scalar vector `[olr, u850, u200]`, averaging over both latitude and longitude (`00_CONTEXT.md` R3). Because the zonal structure is what makes the MJO an eastward-propagating phenomenon, collapsing longitude destroyed all propagation information, rendering phase angles meaningless and RMM uncalibrated.
> On 2026-09-18 (Task J1), this specification and `aurora_mjo.rmm` were rebuilt to strictly implement Wheeler & Hendon (2004) per `02_SCIENTIFIC_CONTRACT.md` §6, retaining all 360 longitudes.

### The Wheeler & Hendon (2004) Procedure

1. **Input Fields**:
   - OLR: Top-of-atmosphere net longwave flux. Convention: `OLR = -mtnlwrf` (magnitude of outgoing longwave radiation, since `mtnlwrf` is net downward flux).
   - Zonal wind at 850 hPa (`u850`).
   - Zonal wind at 200 hPa (`u200`).
   - Daily means from 6-hourly reanalysis fields.

2. **Meridional Averaging**:
   - Average over the equatorial band [15°S, 15°N] using a **simple unweighted meridional mean** to match Wheeler & Hendon (2004).
   - Retain full zonal structure: output has shape `(time, longitude)` where $N_\lambda = 360$ at 1° regular resolution. No scalar collapse across longitude.

3. **Seasonal Cycle Removal**:
   - Annual cycle defined as the **mean plus the first three harmonics** of the annual cycle ($T_0 = 365.25$ days), fitted per longitude on the **training period only (1980–2015)**.
   - Raw day-of-year means are retired (noisy at 36 samples/day).

4. **Interannual (ENSO) Removal — 120-Day Mean under Convention (A)**:
   - Subtract the mean of the previous 120 days at each longitude (`02_SCIENTIFIC_CONTRACT.md` §6.2).
   - **Convention (A)**: For forecasts initialised at $t_0$, the 120-day mean is computed strictly from observed data ending at $t_0$ ($[t_0 - 120\text{d}, t_0]$) and held fixed across all forecast leads $\tau$. For verification observations, the same $t_0$-fixed observed mean is applied, keeping forecast and observation on an identical footing.
   - Using any data after the forecast valid time is strictly forbidden as data leakage.

5. **Field Normalisation**:
   - Divide each field by its **zonally averaged temporal standard deviation** ($\bar{\sigma}$) derived exclusively from the training period (one scalar per field).

6. **Combined Vector Construction**:
   - For each day, concatenate normalised anomalies into $a(t) \in \mathbb{R}^{3 \cdot N_\lambda} = \mathbb{R}^{1080}$.
   - Training matrix $X_{\text{train}}$ has shape $(T_{\text{train}}, 1080)$ where $T_{\text{train}} = 13,149$ days.

7. **EOF Computation by SVD**:
   - Decompose $X_{\text{train}}$ via Singular Value Decomposition: $X_{\text{train}} = U \Sigma V^T$.
   - EOF1 and EOF2 are the first two right singular vectors (rows of $V^T$).
   - **No $1080 \times 1080$ covariance matrix is formed.**

8. **PC Normalisation to Unit Variance**:
   - Divide raw PC projections by their training-period standard deviations:
     $\text{RMM1} = (X \cdot \text{EOF1}) / \sigma_{\text{PC1}}$, $\text{RMM2} = (X \cdot \text{EOF2}) / \sigma_{\text{PC2}}$.
   - Guarantees $\text{Var}(\text{RMM1}) = 1.0$ and $\text{Var}(\text{RMM2}) = 1.0$ on the training set, calibrating the $A > 1$ threshold.

9. **Frozen Sign and Order Transform**:
   - A $2 \times 2$ orthogonal transform matrix is stored in `data/rmm_basis.npz` (defaults to $I_2$ in J1, anchored to the Bureau of Meteorology reference in J2).

10. **Quarantine Contract**:
    - `data/rmm_targets.nc` contains daily RMM1, RMM2, amplitude, phase, and split label covering **1980–2019 only**.
    - Test years (2020–2023) are not processed.

---

## Primary Metrics

- **Bivariate Anomaly Correlation Coefficient (ACC)** vs lead time $\tau$:
  $$\text{ACC}(\tau) = \frac{\sum_t [R_1^f(t, \tau) R_1^o(t, \tau) + R_2^f(t, \tau) R_2^o(t, \tau)]}{\sqrt{\sum_t [(R_1^f)^2 + (R_2^f)^2] \sum_t [(R_1^o)^2 + (R_2^o)^2]}}$$
- **RMSE of RMM1 and RMM2** vs lead time.
- **Combined RMSE**: $\text{RMSE}_{\text{comb}} = \sqrt{\frac{1}{2}(\text{RMSE}_1^2 + \text{RMSE}_2^2)}$.
- **Mean Forecast Amplitude Ratio (Target T4, 02_SCIENTIFIC_CONTRACT.md §1.2 & §1.4)**:
  $$\frac{\mathbb{E}[A_{\text{fc}}]}{\mathbb{E}[A_{\text{ob}}]}$$
  (Primary amplitude fidelity check; target threshold $\ge 0.6$ at Day 20).
- **Amplitude Error (Bias)**: $\mathbb{E}[A_{\text{fc}} - A_{\text{ob}}]$.
- **Mean Absolute Phase Error**: $\mathbb{E}[|\theta_{\text{fc}} - \theta_{\text{ob}}|]$ in degrees.

## Secondary Metrics & Diagnostics

- Skill conditional on **Active MJO** cases ($A(t_0) > 1.0$).
- Gridded tropical RMSE for $ttr$ (OLR), $tcwv$, and $u850$ vs lead time on dual-axes with RMM ACC (`02_SCIENTIFIC_CONTRACT.md` §1.5).
- Skill stratified by **initial phase** (1–8).
- Skill stratified by **season** (boreal winter NDJFMA vs boreal summer MJJASO).
- Skill stratified by **ENSO regime** (El Niño, La Niña, Neutral).
- Eastward / westward spectral power ratio in Wheeler–Kiladis space (propagation gate).

## Initialisation Sampling Protocol

- **Validation Split**: 2016–2019 (1,461 calendar days).
- **Sampling Cadence**: Regular 5-day stride (`--init-stride-days 5`), yielding $N = 292$ total cases.
- **Justification**: 5-day cadence matches the MJO decorrelation / phase advance timescale (~$37.5^\circ$), eliminating serial pseudoreplication between consecutive days while ensuring uniform coverage across all 4 years and 16 seasons. At G2's measured step time (221.4 ms/step), $292 \times 120$ steps costs **2.15 GPU-hours** on 1 A100 GPU.
- **Active Conditioning**: Initial conditions with $A(t_0) > 1.0$ ($N \approx 175$ cases, ~60% active fraction). Both all-case and active-case curves are reported.

## Baselines Protocol (02_SCIENTIFIC_CONTRACT.md §6.4)

All four baselines evaluated on identical initial conditions and lead times:
1. **Persistence**: $\hat{RMM}(t_0 + \tau) = RMM(t_0)$.
2. **Damped Persistence**: $\hat{RMM}(t_0 + \tau) = RMM(t_0) \cdot \exp(-\tau / \tau_d)$, with $\tau_d = 10.60$ days fitted strictly on 1980–2015 training targets (no leakage).
3. **Climatology**: $\hat{RMM} \equiv 0$ (ACC = 0.0, RMSE = $\sqrt{2}$).
4. **Zero-shot Aurora**: Unmodified `AuroraPretrained` (Task J4 control).

## Output File Formats and JSON Schema

Each evaluation run produces:
- `mjo_skill_by_lead.csv` and `active_mjo_skill_by_lead.csv`: Per-lead tables of all metrics.
- `eval_results.json`: Machine-readable results and metadata.
- Plots: `mjo_acc_vs_lead.png`, `mjo_rmse_vs_lead.png`, `mjo_amp_ratio_vs_lead.png`, `mjo_phase_err_vs_lead.png`, `mjo_gridded_rmse_vs_acc.png`.

### `eval_results.json` Schema

```json
{
  "metadata": {
    "timestamp": "ISO-8601 UTC timestamp",
    "git_commit": "SHA-1 commit hash",
    "checkpoint": "path or 'untrained'",
    "config": "path to config YAML",
    "mode": "warmup | lora | rollout | physics",
    "split": "val",
    "max_lead_days": 30,
    "init_stride_days": 5,
    "tau_d_days": 10.6016,
    "n_total_evaluated": 292,
    "n_active_evaluated": 175,
    "active_threshold": 1.0
  },
  "metrics": {
    "all_cases": {
      "leads": [1, 2, "...", 30],
      "acc": ["..."],
      "rmse_rmm1": ["..."],
      "rmse_rmm2": ["..."],
      "rmse_combined": ["..."],
      "amp_err": ["..."],
      "amp_ratio": ["..."],
      "phase_err_deg": ["..."],
      "gridded_rmse_ttr": ["..."],
      "gridded_rmse_tcwv": ["..."],
      "gridded_rmse_u850": ["..."],
      "n_cases": ["..."]
    },
    "active_mjo": {
      "leads": [1, 2, "...", 30],
      "acc": ["..."],
      "rmse_rmm1": ["..."],
      "rmse_rmm2": ["..."],
      "rmse_combined": ["..."],
      "amp_err": ["..."],
      "amp_ratio": ["..."],
      "phase_err_deg": ["..."],
      "gridded_rmse_ttr": ["..."],
      "gridded_rmse_tcwv": ["..."],
      "gridded_rmse_u850": ["..."],
      "n_cases": ["..."]
    }
  },
  "baselines": {
    "persistence": { "leads": ["..."], "acc": ["..."], "rmse_combined": ["..."], "amp_ratio": ["..."], "phase_err_deg": ["..."] },
    "damped_persistence": { "leads": ["..."], "acc": ["..."], "rmse_combined": ["..."], "amp_ratio": ["..."], "phase_err_deg": ["..."] },
    "climatology": { "leads": ["..."], "acc": ["..."], "rmse_combined": ["..."], "amp_ratio": ["..."] }
  }
}
```