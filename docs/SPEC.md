# Technical Specification: Aurora MJO Fine-Tuning

**Document Status:** LIVING DESIGN DOCUMENT · Authoritative System Specification  
**Canonical Package:** `src/aurora_mjo/` (installed in editable mode as `aurora_mjo`)  
**Target Platform:** NERSC Perlmutter (1 node × 4 × NVIDIA A100 80GB SXM4)  

---

## 1. Purpose & Audience

### 1.1 Purpose
This repository adapts the **Microsoft Aurora** foundation model (a 1.3-billion-parameter 3D Swin Transformer pre-trained on diverse Earth-system reanalyses) for sub-seasonal prediction of the **Madden–Julian Oscillation (MJO)**.

Rather than predicting statistical anomaly indices (e.g. Wheeler–Hendon RMM) directly from empirical regressors, the project treats sub-seasonal MJO prediction as a **physics-consistent initial-value prognostic simulation problem**. The system ingests raw 6-hourly global atmospheric states, advances the state forward in time using Aurora's learned atmospheric dynamics, and projects or extracts MJO indices from the resulting prognostic fields.

The ultimate research target is achieving an **RMM bivariate correlation skill of $r > 0.5$ at a 30-day forecast lead time** on held-out evaluation years.

### 1.2 Audience
This specification serves:
- Scientific researchers fine-tuning foundational atmospheric models on high-performance computing clusters.
- Autonomous and human software engineers maintaining the training, evaluation, and data pipelines.
- External collaborators auditing domain priors, data lineage, and numerical integrity.

---

## 2. Domain Background & Nomenclature

| Term | Definition | Role in System |
| :--- | :--- | :--- |
| **MJO** | **Madden–Julian Oscillation**: The dominant mode of intraseasonal (30–90 day) tropical atmospheric variability, characterized by an eastward-propagating envelope of convective clouds and coupled wind anomalies over the Indian and Pacific Oceans. | Core forecasting target of the project. |
| **RMM** | **Real-time Multivariate MJO Index** (Wheeler & Hendon, 2004): A pair of standardized time series (`RMM1`, `RMM2`) defined as the principal component projections of combined equatorially averaged (15°S–15°N) OLR, 850 hPa zonal wind (`U850`), and 200 hPa zonal wind (`U200`). | Formal bivariate metric for tracking MJO amplitude and phase. |
| **OLR** | **Outgoing Longwave Radiation** (W m⁻²): Radiative cooling at the top of the atmosphere. Strongly negative anomalies denote deep convective cloud towers in the tropics. Represented in ERA5 by Top Net Long-Wave Radiation Flux (`mtnlwrf` / `ttr`). | Primary proxy for tropical convective activity. |
| **TCWV** | **Total Column Water Vapor** (kg m⁻²): Vertically integrated atmospheric water vapor content. Critical for resolving moist convective dynamics and "moisture mode" self-aggregation. | Injected surface variable absent from standard weather forecasts. |
| **2t** | 2-meter surface air temperature (K). | Prognostic surface variable. |
| **10u, 10v** | 10-meter zonal (u) and meridional (v) surface wind components (m s⁻¹). | Prognostic surface momentum variables. |
| **msl** | Mean Sea Level Pressure (Pa). In LANL ERA5, **surface pressure (`ps`) is used as a proxy**. | Prognostic mass variable; requires careful renormalisation (Lesson 1). |
| **q** | Specific humidity (kg kg⁻¹) across vertical pressure levels. | Atmospheric moisture field. |
| **t** | Air temperature (K) across vertical pressure levels. | Atmospheric thermal field. |
| **u, v** | Zonal and meridional winds (m s⁻¹) across vertical pressure levels. | Atmospheric kinetic field. |
| **z** | Geopotential ($m^2 s^{-2}$): Equivalent to geopotential height $\times$ $g_0$ ($9.80665\text{ m s}^{-2}$). | Surface topography invariant and atmospheric mass field. |
| **lsm** | Land-Sea Mask ($0.0 \le \text{fraction} \le 1.0$). | Surface static invariant. |
| **slt** | Soil Type (integer categories 0–7). | Surface static invariant from `slt_data.nc`. |

---

## 3. Data Lineage & Upstream Contract

### 3.1 Upstream Archive Details
All operational reanalysis data resides on the NERSC Community File System (CFS) at:
```text
/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results
```
> [!CAUTION]
> **Read-Only Discipline:** This path belongs to another research group's CFS allocation (`m4946`). The codebase attaches this directory **strictly read-only**. No temporary files, indices, or outputs may ever be written into this tree.

### 3.2 Variable Mapping & Naming
The native variable and directory names in the LANL remapped ERA5 archive differ from Aurora's internal naming conventions:

#### Surface Variables
| Aurora Name | Step Subdirectory | Native File Pattern | Native Variable | Notes / Physical Proxy |
| :--- | :--- | :--- | :--- | :--- |
| `2t` | `Step02/...6hrInst/T2` | `e5.oper.an.sfc.128_167_2t.*.nc` | `t2` | 2m Air Temperature (K) |
| `10u` | `Step02/...6hrInst/U10` | `e5.oper.an.sfc.128_165_10u.*.nc` | `u10` | 10m U-Wind Component (m s⁻¹) |
| `10v` | `Step02/...6hrInst/V10` | `e5.oper.an.sfc.128_166_10v.*.nc` | `v10` | 10m V-Wind Component (m s⁻¹) |
| `msl` | `Step02/...6hrInst/PS` | `e5.oper.an.sfc.128_134_sp.*.nc` | `ps` | **Surface Pressure (`ps`) used as proxy.** See §8.1. |
| `ttr` | `Step03/...6hrInst/meanTNLWFLX` | `e5.oper.an.sfc.128_179_ttr.*.nc` | `mtnlwrf` | Top Net LW Radiation Flux (W m⁻², negative). |
| `tcwv` | `Step02/...6hrInst/tcwv` | `e5.oper.an.sfc.128_137_tcwv.*.nc` | `tcwv` | Total Column Water Vapor (kg m⁻²). |

#### Atmospheric Variables (13 Pressure Levels)
| Aurora Name | Step Subdirectory | Native File Pattern | Native Variable | Units / Levels |
| :--- | :--- | :--- | :--- | :--- |
| `z` | `Step01/...6hrInst/gopt` | `e5.oper.an.pl.128_129_z.*.nc` | `z` | Geopotential (m² s⁻²) |
| `q` | `Step01/...6hrInst/sphu` | `e5.oper.an.pl.128_133_q.*.nc` | `q` | Specific Humidity (kg kg⁻¹) |
| `t` | `Step01/...6hrInst/tprt` | `e5.oper.an.pl.128_130_t.*.nc` | `t` | Temperature (K) |
| `u` | `Step01/...6hrInst/uWnd` | `e5.oper.an.pl.128_131_u.*.nc` | `u` | Zonal Wind (m s⁻¹) |
| `v` | `Step01/...6hrInst/vWnd` | `e5.oper.an.pl.128_132_v.*.nc` | `v` | Meridional Wind (m s⁻¹) |

#### Static / Invariant Fields
| Field Name | Source Path | Variable Key | Native Resolution | Ingestion Transform |
| :--- | :--- | :--- | :--- | :--- |
| `z` (surface) | `Step00/ERA5.invariant/*_z.*.nc` | `'Z'` | 1° (180 × 360) | Native 1° cell-centred (Task G1) |
| `lsm` | `Step00/ERA5.invariant/*_lsm.*.nc` | `'LSM'` | 1° (180 × 360) | Native 1° cell-centred (Task G1) |
| `slt` | `data/static/slt_1deg.nc` | `'slt'` | 1° (180 × 360) | Nearest-neighbour regridded (Task G1) |
| `sst` | `data/static/sst/sst_1deg_<year>.nc` | `'sst'` | 1° (180 × 360) | Persisted static boundary at $t_0$, zonal-mean land fill (Task G5) |

---

## 4. Grid, Vertical Levels, and Tensor Geometries

All geometries and statistical measures below are **MEASURED** from the verification suite ([`docs/verify_output_slt.txt`](verify_output_slt.txt)) and Task G1 1.0° native resolution:

```text
Native Grid Resolution:   1.0° regular lat/lon (180 latitude × 360 longitude)
Aurora Grid Resolution:   1.0° regular lat/lon (180 latitude × 360 longitude)
Vertical Pressure Levels: 13 hPa levels: (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
Temporal Spacing:         6-hourly instantaneous (00:00, 06:00, 12:00, 18:00 UTC; 4 steps/day)
```

### Tensor Dimensions
| Component | Tensor Shape | Description |
| :--- | :--- | :--- |
| **Input Surface Variables** | `[B, 2, 180, 360]` | 2 consecutive historical timesteps ($t-6\text{h}$, $t$). |
| **Input Atmos Variables** | `[B, 2, 13, 180, 360]` | 2 consecutive historical timesteps across 13 vertical levels. |
| **Target Surface Variables** | `[B, 180, 360]` | 1 prediction step ($t + \Delta t$) per rollout step. |
| **Target Atmos Variables** | `[B, 13, 180, 360]` | 1 prediction step across 13 vertical levels. |
| **Static Fields (`z`, `lsm`, `slt`, `sst`)** | `[180, 360]` | Invariant 2D surface fields injected into every forward pass. |
| **Metadata Coordinates** | `lat: [180]`, `lon: [360]` | Uniformly spaced latitudes [89.5, −89.5] and longitudes [0.5, 359.5]. |

---

## 5. Dataset Splitting & Exact Arithmetic

### 5.1 Chronological Split Policy
To prevent future-information leakage into validation and testing splits, **only strict chronological splits are permitted**:

- **Training Split:** 1980–2015 (36 years)
- **Validation Split:** 2016–2019 (4 years)
- **Test Split:** 2020–2023 (4 years)

Random splits or cross-validation shuffles across time are strictly prohibited.

### 5.2 Arithmetic Sample Counts
Because data is sampled 6-hourly instantaneous (4 timesteps/day), each sample requires consecutive timesteps ($t-6\text{h}$, $t$, and $k$ target rollout steps). For 1-step training ($k=1$):
$$\text{timesteps} = \text{days} \times 4$$
$$\text{samples} = \text{timesteps} - (k + 1)$$

| Range | Years | Leap Years | Total Days | Timesteps | Samples ($k=1$) | Derivation / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1980 only** | 1 | 1 (1980) | 366 | 1,464 | **1,462** | **MEASURED** ([`docs/verify_output_slt.txt`](verify_output_slt.txt)) |
| **1981 only** | 1 | 0 | 365 | 1,460 | **1,458** | **MEASURED** ([`tests/test_dataset_index.py`](../tests/test_dataset_index.py)) |
| **1980–2015 (Train)** | 36 | 9 | 13,149 | 52,596 | **52,594** | Arithmetic Ground Truth |
| **2016–2019 (Val)** | 4 | 1 (2016) | 1,461 | 5,844 | **5,842** | Arithmetic Ground Truth |
| **2020–2023 (Test)** | 4 | 1 (2020) | 1,461 | 5,844 | **5,842** | Arithmetic Ground Truth |
| **1980–2016 (Bug)** | 37 | 10 | 13,515 | 54,060 | 54,058 | **Observed Leakage Artifact** (Lesson 5) |

> [!WARNING]
> **The 54,060 Signature:** The pre-refactor dataset loader indexed via naive file globs, returning exactly 54,060 timesteps for the 1980–2015 split. This occurred because 2016 (the first validation year) was pulled into the training set. If any dataset verification reports 54,060 timesteps, validation leakage has occurred.

---

## 6. Normalisation & The MSL Surface Pressure Trap

### 6.1 Current Normalisation Configuration
Aurora applies channel-wise normalisation $(x - \mu) / \sigma$ at its encoder boundary. The active values in `configs/unified.yaml` are:

| Channel | Location ($\mu$) | Scale ($\sigma$) | Source / Status |
| :--- | :--- | :--- | :--- |
| `2t` | 287.85 K | 14.81 K | Aurora built-in ERA5 stats |
| `10u` | 0.44 m s⁻¹ | 6.07 m s⁻¹ | Aurora built-in ERA5 stats |
| `10v` | 0.08 m s⁻¹ | 6.09 m s⁻¹ | Aurora built-in ERA5 stats |
| `msl` | **96,668.75 Pa** | **9,504.41 Pa** | Computed 1980–2015 Welford stats (Task G3) |
| `ttr` | −225.8354 W m⁻² | 45.4344 W m⁻² | Computed 1980–2015 Welford stats (Task G3) |
| `tcwv` | 17.0782 kg m⁻² | 16.5956 kg m⁻² | Computed 1980–2015 Welford stats (Task G3) |
| `sst` | 285.5980 K | 11.7613 K | Computed 1980–2015 Welford stats (Task G5) |

### 6.2 The Arithmetic of Lesson 1
Aurora's built-in `msl` constants are calibrated to sea level ($\mu = 100,958\text{ Pa}$, $\sigma = 1,332\text{ Pa}$). Surface pressure over the Tibetan Plateau or Antarctic ice sheet reaches $\sim 52,000\text{ Pa}$.
- Under Aurora MSL stats:
  $$\frac{52,000 - 100,958}{1,332} \approx -36.8 \sigma$$
- Under the `sp` placeholder override:
  $$\frac{55,000 - 96,668}{9,505} \approx -4.4 \sigma$$

The −36.8 σ input destabilizes the attention blocks and is the confirmed trigger of 100% non-finite validation loss.

### 6.3 Requirements for Replacing Placeholder Normalisation
The historical placeholder values were replaced in Task G3 with parallel Welford population statistics over the full 1980–2015 training record at 1.0° native resolution (`configs/norm_stats_1980_2015.yaml`).

### 6.4 SST Static Boundary Condition & The 30-Day Persistence Boundary
Over a 6-hour forecast step, sea surface temperature is nearly constant. Over a **120-step (30-day) rollout**, however, SST is the dominant boundary forcing on tropical convection and the single field that encodes ENSO state (El Niño / La Niña / Neutral). Without an ocean surface boundary, atmospheric memory decorrelates in 10–15 days and forecast rollouts relax toward an ENSO-agnostic climatology.

1. **Persisted Static Semantics:** Sea surface temperature enters the model as an initial-value static variable initialised from observed daily ERA5 SST at $t_0$ and held constant across all rollout steps ($t_0 \to t_{119}$). Because static variables in Aurora possess encoder patch embeddings (`surf_token_embeds.weights['sst']`) but **no decoder output heads**, SST cannot drift or be predicted autoregressively.
2. **Scientific Horizon Limitation (~30 Days):** Persisting the initial SST anomaly through a sub-seasonal forecast matches operational S2S convention because oceanic anomalies decorrelate on monthly timescales. At a 30-day lead time this assumption is physically defensible; **beyond ~45 days simple persistence breaks down**. This is an explicit, stated boundary of every scientific claim this project makes.
3. **Absence of Ocean Dynamics:** The model contains **no ocean dynamics, no mixed-layer thermodynamics, and no ocean-atmosphere coupling**.
4. **Later-Campaign Refinement (Damped Persistence):** Relaxing SST anomalies toward climatology with lead time (damped persistence) is the logical subsequent refinement, deferred to a later campaign to establish an unconfounded baseline first.
5. **Land Fill & Normalisation Safety:** ERA5 SST is undefined over land. Missing land cells are filled with the global zonal-mean SST for each latitude band, preserving physical continuity. Under population normalisation ($\mu = 285.5980\text{ K}$, $\sigma = 11.7613\text{ K}$), the worst-case land grid cell sits at $1.7452 \sigma$ (306.12 K), completely avoiding normalisation spikes or unphysical input cliffs.

---

## 7. Prognostic Grid Loss Formulation (Task H1)

### 7.1 Loss Formulation
The composite grid loss $\mathcal{L}_{\text{grid}}$ is computed on **normalised errors** to preserve gradient balance across variables differing by orders of magnitude in physical units:

$$\mathcal{L}_{\text{grid}} = \sum_{v \in \mathcal{V}} w_v \cdot \mathcal{L}_v$$

Where $\mathcal{V}$ comprises the 6 surface and 5 atmospheric variables.

For each surface variable $v \in \{\text{2t, 10u, 10v, msl, ttr, tcwv}\}$:
$$\mathcal{L}_v = \frac{1}{\sum_\phi a(\phi) m(\phi)} \sum_{\phi, \lambda} a(\phi) m(\phi) \left| \frac{\hat{x}_{v,\phi,\lambda} - \mu_v}{\sigma_v} - \frac{y_{v,\phi,\lambda} - \mu_v}{\sigma_v} \right|$$

For each atmospheric variable $v \in \{\text{z, q, t, u, v}\}$ across the 13 pressure levels $\ell \in \{50, \dots, 1000\}\text{ hPa}$:
$$\mathcal{L}_v = \frac{1}{\sum_\phi a(\phi) m(\phi)} \sum_{\phi, \lambda} a(\phi) m(\phi) \sum_{\ell=1}^{13} c_\ell \left| \frac{\hat{x}_{v,\ell,\phi,\lambda} - \mu_{v,\ell}}{\sigma_{v,\ell}} - \frac{y_{v,\ell,\phi,\lambda} - \mu_{v,\ell}}{\sigma_{v,\ell}} \right|$$

### 7.2 Weighting Components
1. **Latitude Area Weighting $a(\phi)$:** Corrects for converging meridians on a regular 1° grid:
   $$a(\phi) = \frac{\cos \phi}{\frac{1}{N_\phi} \sum_{j=1}^{N_\phi} \cos \phi_j}, \quad \text{normalized such that } \frac{1}{N_\phi} \sum_{\phi} a(\phi) = 1$$
2. **Tropical Emphasis $m(\phi)$:** Focuses gradient updates on MJO-relevant dynamics while maintaining global coverage:
   $$m(\phi) = \begin{cases} 1.0 & \text{if } |\phi| \le 20^\circ \text{ (tropics)} \\ 0.1 & \text{if } |\phi| > 20^\circ \text{ (extratropics)} \end{cases}$$
   $a(\phi)$ and $m(\phi)$ remain **strictly separate and multiplicative** ($w(\phi) = a(\phi) m(\phi)$).
3. **Vertical Level Weighting $c_\ell$:** Defaults to pressure-thickness weighting:
   $$c_\ell = \frac{\Delta p_\ell}{\sum_{k} \Delta p_k}, \quad \sum_{\ell} c_\ell = 1.0$$
   Where $\Delta p_\ell$ represents layer thicknesses via central differences (one-sided at boundaries), naturally weighting lower tropospheric moisture dynamics. A `uniform` alternative ($c_\ell = 1/13$) is config-selectable.
4. **Variable Importance Weights $w_v$:** Configured in `configs/unified.yaml` and set *a priori* based on physical importance to MJO convection without validation tuning:
   - $w_v = 2.0$: Convective & moisture-mode variables (`ttr`, `tcwv`, `q`)
   - $w_v = 1.0$: Dynamics & thermal fields (`u`, `v`, `t`)
   - $w_v = 0.5$: Large-scale balance & boundary fields (`z`, `msl`, `2t`, `10u`, `10v`)

### 7.3 Denormalise-Then-Renormalise Path
Aurora predicts prognostic outputs in physical units. Both predictions and ground truth targets are re-normalised using the 1980–2015 Welford training population statistics (`configs/norm_stats_1980_2015.yaml` from Task G3) prior to computing absolute differences. This eliminates scientific defect R1 where raw `msl` and `z` consumed 99.03% of the gradient budget.

---

## 8. Model Architecture, Trainable Surface & Parameter Budgets (Task H4)

### 8.1 Model Scale (`model_type: full`)
The operational model uses `AuroraPretrained` (1.3B Swin3D transformer backbone), loading weights pre-trained on diverse Earth-system reanalyses. The patch size is $4 \times 4$ horizontally with 13 vertical levels and 2 historical timesteps.

### 8.2 Parameter Budgets Across Modes
The training campaign is structured into four distinct modes across four sequential training stages:
1. **`warmup` (Stage 0):** Zero LoRA adapters. The 1.3B backbone is entirely frozen. Only newly injected surface variables (`tcwv`, `ttr`) and static boundary (`sst`) patch embeddings, their corresponding decoder output heads, and the `msl` decoder head are trainable.
2. **`lora` (Stage 1):** Low-Rank Adaptation (LoRA) attached to attention projection layers in all Swin3D blocks, in addition to the warmup surface.
3. **`rollout` (Stage 2):** Autoregressive rollout multi-step fine-tuning ($k > 1$) with identical trainable surface to `lora`.
4. **`physics` (Stage 3):** Rollout fine-tuning with physics loss constraints active (`moisture_budget`).

#### Parameter Counts (`model_type: full` / `AuroraPretrained`):
| Mode | Total Parameters | Trainable Parameters | Frozen Parameters | Trainable % |
| :--- | :--- | :--- | :--- | :--- |
| **`warmup`** | 1,256,382,128 | **98,352** | 1,256,283,776 | **0.0078%** |
| **`lora`** | 1,259,232,944 | **2,949,168** | 1,256,283,776 | **0.2342%** |
| **`rollout`** | 1,259,232,944 | **2,949,168** | 1,256,283,776 | **0.2342%** |
| **`physics`** | 1,259,232,944 | **2,949,168** | 1,256,283,776 | **0.2342%** |

*(Note: If static SST was excluded, the warmup trainable count would be 81,968 parameters across 5 modules. Task G5 added `surf_token_embeds.weights.sst` adding 16,384 parameters, bringing the total to 98,352 parameters).*

#### Module Breakdown of Trainable Surface:
1. **Warmup Surface (98,352 parameters across 6 parameter tensors):**
   - Injected Encoder Patch Embeddings (49,152 parameters):
     - `backbone.encoder.surf_token_embeds.weights.sst` (shape `[1024, 16]`, 16,384 params)
     - `backbone.encoder.surf_token_embeds.weights.tcwv` (shape `[1024, 16]`, 16,384 params)
     - `backbone.encoder.surf_token_embeds.weights.ttr` (shape `[1024, 16]`, 16,384 params)
   - Injected / Recalibrated Decoder Heads (49,200 parameters):
     - `backbone.decoder.surf_heads.msl.weight` (`[16, 1024]`, 16,384) + `bias` (`[16]`, 16) = 16,400 params
     - `backbone.decoder.surf_heads.tcwv.weight` (`[16, 1024]`, 16,384) + `bias` (`[16]`, 16) = 16,400 params
     - `backbone.decoder.surf_heads.ttr.weight` (`[16, 1024]`, 16,384) + `bias` (`[16]`, 16) = 16,400 params
2. **LoRA Adapters (2,850,816 parameters across 96 parameter tensors):**
   - Attached to `WindowAttention.qkv` and `WindowAttention.proj` in Swin3D blocks throughout encoder and decoder.
   - For each adapted linear layer: rank $r=8$, scaling $\alpha=8$ (scaling factor $\alpha/r = 1.0$).
   - Total LoRA parameter count: $98,352 \text{ (warmup)} + 2,850,816 \text{ (LoRA)} = 2,949,168$ parameters.

### 8.3 LoRA Configuration & Per-Step Adapter Option
- **Current Baseline Configuration:**
  - `rank: 8`, `lora_alpha: 8`, `lora_dropout: 0.05`
  - `lora_mode: "single"`: A single shared adapter is evaluated across all autoregressive rollout steps.
  - Adapted module targets: query/key/value projections (`qkv`) and output projection (`proj`) of windowed multi-head self-attention.
- **Later-Campaign Candidate (Per-Step Adapters):**
  - Aurora 1.8.0 provides built-in support for time-step-conditioned LoRA adapters (`lora_mode: "step"` up to `lora_steps: 40`), allocating separate LoRA parameters for individual rollout lead times.
  - In this baseline campaign, `lora_mode: "single"` is strictly preserved to establish an unconfounded baseline without swelling the parameter count ($40 \times 2.85\text{M} \approx 114\text{M}$ params) or altering rollout step mechanics.

### 8.4 Autoregressive Clamping: Vendor vs. In-Tree Guards
To prevent non-physical runaway states during multi-step rollouts ($k > 1$):
1. **Aurora Native Clamping:**
   - Total Column Water Vapor: `positive_surf_vars=("tcwv",)` clamps predicted surface moisture to $\ge 0$.
   - Specific Humidity: `positive_atmos_vars=("q",)` clamps vertical humidity levels to $\ge 0$.
   - First-Step Guard: `clamp_at_first_step=False` during training. In Stage 0 (`warmup`), newly initialized heads (`tcwv`, `ttr`) output mean-zero predictions, yielding negative values across ~50% of the domain on step 1. Hard clamping on step 1 zeroes out backward gradients ($\partial \text{clamp}(x, 0)/\partial x = 0$ for $x < 0$), preventing heads from learning. Setting `clamp_at_first_step=False` allows full gradient propagation.
2. **In-Tree Physical Rollout Clamping (`_ROLLOUT_CLAMP`):**
   - Surface pressure (`msl` proxy): $[50000, 110000]\text{ Pa}$
   - 2-meter air temperature (`2t`): $[180, 340]\text{ K}$
   - Surface wind velocity (`10u`, `10v`): $[-100, 100]\text{ m s}^{-1}$
   - Outgoing longwave radiation (`ttr`): $[-450, 0]\text{ W m}^{-2}$ (downward-negative convention)
   These guards lack vendor constructor hooks in Aurora and are retained in `_advance_batch` to prevent explosive numerical instabilities during multi-week rollouts.

### 8.5 Validation Optimization (`torch.no_grad()`)
Validation loops in `Trainer.validate()` are wrapped in `torch.no_grad()`.
- **Loss Equivalence at $k=1$:** Validation loss is bitwise identical under `torch.no_grad()` vs `torch.enable_grad()` (measured: $543,162.875000$ vs $543,162.875000$).
- **Memory Scaling at $k=4$:** In multi-step autoregressive rollout validation, disabling gradient tracking prevents retaining 4 full computational graphs in VRAM. Measured peak memory on Perlmutter A100:
  - `torch.enable_grad()`: 21,395.97 MiB
  - `torch.no_grad()`: 1,240.42 MiB
  - **Reduction factor: 17.25×** (saving >20 GiB VRAM), entirely eliminating the projected Phase K validation OOM risk.

---

## 9. Promoted Durable Findings: The Eight Lessons

The following lessons were established through real failures on Perlmutter:

### Lesson 1: `msl` is Surface Pressure Wearing an MSL Costume
- **Mechanism:** Surface pressure fed through MSL normalisation creates −36 σ inputs, producing non-finite attention representations.
- **Guard:** `configs/unified.yaml` overrides `model.norm_stats.msl`.
- **Regression Test:** [`tests/test_norm_stats.py::test_msl_override_is_active_in_unified_config`](../tests/test_norm_stats.py)

### Lesson 2: Finite Loss Can Produce Non-Finite Gradients Under bf16
- **Mechanism:** Under bfloat16, PyTorch's `GradScaler` is disabled. Without an explicit per-parameter finiteness check, NaN/Inf gradients propagate into Adam moment buffers before applying learning rate, permanently corrupting the optimizer.
- **Guard:** Collective DDP finite gradient guard in `src/aurora_mjo/trainer.py` (`_collective_all_finite`).
- **Regression Test:** [`tests/test_grad_guard.py::test_collective_all_finite_detects_nan`](../tests/test_grad_guard.py)

### Lesson 3: HDF5 File Locking on Parallel Filesystems
- **Mechanism:** CFS mounts trigger `OSError: [Errno -101] NetCDF: HDF error` due to Lustre/GPFS advisory lock contention on read-only files.
- **Guard:** `HDF5_USE_FILE_LOCKING=FALSE` configured at Python process initialization in `src/aurora_mjo/env.py` and `slurm_scripts/env.sh`.
- **Regression Test:** [`tests/test_env.py::test_configure_environment_sets_hdf5_locking`](../tests/test_env.py)

### Lesson 4: Silent Fallbacks Hide Failures and Zero Out Physics
- **Mechanism:** Broad `try...except` blocks in static loading caught `[Errno -101]` and silently substituted zero tensors for `z` and `lsm`, training on a flat, ocean-only planet.
- **Guard:** Fallbacks removed; `StaticVarLoadError` raised immediately with remediation hints.
- **Regression Test:** [`tests/test_static_vars.py::test_static_var_load_failure_raises_loudly`](../tests/test_static_vars.py)

### Lesson 5: A Glob Is Not a Year Filter
- **Mechanism:** Pre-v3 dataset construction relied on filesystem globs, inadvertently leaking validation data (year 2016) into training (producing 54,060 timesteps).
- **Guard:** Multi-variable timestamp alignment maps with strict build-time year boundaries and intersection pruning.
- **Regression Test:** [`tests/test_dataset_index.py::test_year_range_enforcement_prevents_validation_leak`](../tests/test_dataset_index.py)

### Lesson 6: Gradient Checkpointing Crashes Perlmutter A100s
- **Mechanism:** Enabling PyTorch gradient checkpointing deterministically triggers an illegal memory access (IMA) crash in Triton/cuBLAS kernels on Perlmutter A100s.
- **Guard:** `gradient_checkpointing: false` strictly enforced in all config modes and rejected by boundary validation in `src/aurora_mjo/config.py`.
- **Regression Test:** [`tests/test_config_validation.py::test_rule_gradient_checkpointing_forbidden`](../tests/test_config_validation.py)

### Lesson 7: Plan Against Measured State, Not Remembered State
- **Mechanism:** Reasoning against remembered or documented state rather than inspecting real filesystem artifacts leads to false assumptions and wasted queue time.
- **Guard:** Protocol mandates measured assertions, stdlib verification gate (`scripts/check.py`), and B1/C4 behavioral fingerprinting.

### Lesson 8: Unnormalised Losses Starve Moisture Modes (Defect R1)
- **Mechanism:** Computing L1 losses in raw physical units results in large-scale geopotential ($z \sim 10^4$) and pressure ($msl \sim 10^5$) consuming >99% of backpropagated gradients, starving specific humidity ($q \sim 10^{-3}$) with an effective $5.8 \times 10^6 : 1$ gradient disparity.
- **Guard:** `TropicalWeightedL1Loss` enforces re-normalisation with 1980–2015 Welford stats, area weighting $a(\phi)$, pressure-delta vertical weighting $c_\ell$, and config-locked $w_v$.
- **Regression Test:** [`tests/test_loss.py::test_headline_per_variable_gradient_share`](../tests/test_loss.py)

---

## 10. Upstream Traps & Architectural Guards

| Upstream Trap | Architectural Failure Mode | Enforcing Code Guard | Verified In Test |
| :--- | :--- | :--- | :--- |
| **Trap 4.1: `msl` is proxy for `ps`** | −36 σ input; validation NaN forever | Config override `model.norm_stats.msl` | [`tests/test_norm_stats.py`](../tests/test_norm_stats.py) |
| **Trap 4.2: Divergent variable time axes** | Reading misaligned variables at different timesteps | Timestamp intersection in `src/aurora_mjo/dataset.py` | [`tests/test_dataset_index.py`](../tests/test_dataset_index.py) |
| **Trap 4.3: Static loading zero-fallbacks** | Training on a planet with zero elevation | Loud failure with `StaticVarLoadError` in `env.py` | [`tests/test_static_vars.py`](../tests/test_static_vars.py) |
| **Trap 4.4: `slt_data.nc` on scratch space** | File purge on `/pscratch` crashes run | Explicit error detailing non-archive path and `download_slt.py` | [`tests/test_static_vars.py`](../tests/test_static_vars.py) |
| **Trap 4.5: Advisory file locking on CFS** | Crash on `open_dataset` with `Errno -101` | `HDF5_USE_FILE_LOCKING=FALSE` in `src/aurora_mjo/env.py` | [`tests/test_env.py`](../tests/test_env.py) |

---

## 11. Non-Goals

The following areas are explicitly **out of scope** for the current architecture:
1. **Full Backpropagation Through Time (`backprop: "full"`):** Multi-step rollouts use detached pushforward (`detach=True`), detaching each step before feeding into the next step. Full BPTT is unexercised in this baseline campaign.
2. **Prognostic Ocean / SST Dynamics (Task G5 Limitation):** Sea surface temperature is treated as an initial-value persisted static boundary condition held fixed across each forecast rollout. Simulating dynamic ocean response, mixed layer thermodynamics, ocean-atmosphere coupling, or damped-persistence relaxation is explicitly out of scope for this baseline campaign.
3. **Damped-Persistence SST:** Relaxing the SST anomaly toward climatology with lead time is deferred to a later-campaign comparison; simple persistence is implemented first.
4. **Time-Step-Conditioned LoRA Adapters:** Exploring per-step LoRA adapters (`lora_mode: "step"` with `lora_steps: 40`) is deferred to a later campaign; single shared adapter (`lora_mode: "single"`) is the baseline.

---

## 12. Open Items

| Item ID | Description | Impact / Blocks | Default / Current Mitigation |
| :--- | :--- | :--- | :--- |
| **Q-01** | Package name collision with `microsoft-aurora` | Shadowing top-level import | First-party package is `aurora_mjo` under `src/aurora_mjo/`. |
| **Q-03** | Perlmutter `uv` locking and cache location | Lustre filesystem lock error 524 | Cache redirected to `/pscratch/sd/k/kam352/.cache/uv`. |
| **Q-07** | `slt_data.nc` storage permanence | Scratch purge risks data availability | Maintained at `/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc`; loud failure upon missing. |
| **OPEN-01** | Recompute true `ps` normalisation constants | Production baseline training accuracy | Uses Aurora `sp` built-in placeholder in `configs/unified.yaml`. |
