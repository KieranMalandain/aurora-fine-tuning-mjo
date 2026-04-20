# Architecture

## Overview

This repository implements a forecasting pipeline built around Microsoft Aurora for MJO-focused subseasonal prediction.

The intended system has five major parts:

1. Data ingestion and preprocessing
2. Dataset construction and normalization
3. Aurora-based model with MJO-specific extensions
4. Training and rollout optimization
5. Evaluation in MJO phase space

## Current Code Layout

- `configs/`: experiment configurations
- `scripts/`: preprocessing and utility scripts
- `slurm/`: job submission wrappers
- `src/dataset.py`: lazy-loading dataset logic
- `src/model.py`: Aurora wrapper and adaptation logic
- `src/loss.py`: custom training losses
- `src/trainer.py`: train/validation loop
- `train.py`: main training entry point

## Data Flow

Intended flow:

1. Download ERA5 and related variables.
2. Convert / harmonize fields into the format expected by the dataset.
3. Compute normalization statistics from training years only.
4. Load 6-hourly global states through `MJODataset`.
5. Feed states into Aurora with injected OLR/TTR and TCWV channels.
6. Predict next-step or rollout states.
7. Compute gridded and MJO-specific losses.
8. Evaluate forecast outputs in RMM space.

## Model Design

### Backbone

Aurora is the backbone model.

Aurora is used as a pretrained Earth-system model and adapted for MJO-specific forecasting.

### Injected Variables

The main non-default variables of interest are:
- OLR or ERA5 TTR-derived proxy
- TCWV

These variables are treated as critical for convective and moisture-envelope representation.

### MJO Head (implemented – scaffold)

An explicit MJO head is implemented in `src/model.py` as `AuroraMJO`.
The head predicts:
- RMM1
- RMM2
- amplitude (active-MJO probability is deferred to a later phase)

Implementation:
- A forward hook on `Aurora.encoder` captures the encoder latent `x`
  of shape `(B, latent_levels × H_patches × W_patches, embed_dim)`.
- The latent is reshaped to `(B, n_levels, H_patches, W_patches, embed_dim)`
  and mean-pooled over all lat patches inside the configured tropical band
  (default ±15°) and all longitudes, yielding `(B, embed_dim)`.
- A two-layer MLP (LayerNorm → Linear → GELU → Dropout → Linear) maps that
  to `(B, 3)` → `[RMM1, RMM2, Amplitude]`.
- The head is controlled by `config['mjo_head']['enabled']` and is off by
  default, making the model a transparent Aurora wrapper when disabled.
- The final linear layer is zero-initialized so the head starts neutral.
- `load_checkpoint(strict=False)` is preserved; the new MLP weights are not
  present in the pretrained checkpoint and are handled gracefully.

### LoRA

LoRA is used to specialize the model for long-horizon MJO forecasting while keeping the pretrained Aurora backbone mostly frozen.

**Confirmed insertion points** (from `aurora/model/swin3d.py`):
- `lora_qkv`: a `LoRARollout` adapter on the fused QKV projection of every Swin3D window-attention block (`dim → dim×3`).
- `lora_proj`: a `LoRARollout` adapter on the output projection of every Swin3D window-attention block (`dim → dim`).
- MLP blocks are **not** adapted by LoRA.
- Default rank `r=8`, alpha `8`; mode controlled by `config['lora_mode']` (`"single"`, `"from_second"`, or `"all"`).

**Freezing strategy** (implemented in `src/model.py::freeze_backbone()`):
- All backbone parameters start frozen (`requires_grad=False`).
- Unfrozen when `use_lora=True`: all `LoRA`/`LoRARollout` module parameters (`lora_A`, `lora_B`).
- Unfrozen always: patch-embedding weights for newly injected surface variables (`ttr`, `tcwv`) — these are randomly initialized because the pretrained checkpoint has no entry for them.
- The `MJOHead` MLP is fully trainable (it lives outside the backbone).
- To disable freezing entirely (full fine-tune), set `config['freeze_backbone'] = False`.

## Training Design

### Baseline phase

- one-step or short-horizon supervised training
- stabilize variable injection
- establish evaluation pipeline

### Rollout phase

- multi-step autoregressive training
- increasing rollout horizon curriculum
- lead-dependent loss accounting

### Physics-informed phase

- add optional moisture-budget auxiliary loss
- small weight initially
- evaluate carefully for stability vs oversmoothing tradeoff

## Evaluation Design

Evaluation should not rely only on gridded reconstruction metrics.

Primary intended metrics:
- bivariate correlation in RMM space
- amplitude and phase error
- active-event skill
- phase-conditioned skill
- seasonal skill

## Known Architectural Risks

- single-step improvement may not transfer to subseasonal MJO skill
- spectral losses may improve visual sharpness but hurt long rollouts
- poorly matched normalization can destabilize adaptation
- physics-informed losses can oversmooth if added too early
- LoRA rank or insertion choices may underfit MJO-specific dynamics

## Current Code-Specific Implementation Facts

- **Input Tensor Shapes:** Raw data from LANL is 1-degree `(180, 360)`. **Crucial:** It is upsampled on-the-fly in the DataLoader via `F.interpolate(mode='bilinear')` to Aurora's native 0.25-degree resolution `(720, 1440)` before being passed to the model.
- **Pressure Levels:** We slice exactly 13 levels required by Aurora: `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa.
- **TTR Conversion:** LANL's `mtnlwrf` is already in $W/m^2$. **Do NOT divide by 3600.**
- **TCWV Injection:** Working. Passed via the `surf_vars` dictionary mapping.
- **Missing Static Variable:** LANL data lacks `slt` (Soil Type). The DataLoader currently injects a dummy zero-tensor of shape `(720, 1440)` to prevent model crashes. Human is working on obtaining the soil-type data.
- **Checkpoint Format:** `microsoft/aurora` (aurora-0.25-pretrained.ckpt). Always loaded with `strict=False` to accommodate the randomly initialized embedding layers for `ttr` and `tcwv`.
- **Target Architecture (Pending):** A dual-head system. 
  1. *State Head:* Aurora's default decoder outputting the full grid.
  2. *MJO Head:* An MLP attached to the encoder's latent space (tropical pooled) outputting `[RMM1, RMM2, Amplitude]`.