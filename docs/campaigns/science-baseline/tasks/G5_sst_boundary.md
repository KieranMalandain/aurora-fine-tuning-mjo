# G5 — SST as a persisted static: give the model an ocean

| | |
| --- | --- |
| **Phase** | G |
| **Depends on** | G1, G2 (both `PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 5 h |
| **Compute** | Perlmutter login node for data sourcing and regridding. 1 GPU-hour for a forward-pass check. `sbatch` with approval. |
| **Touches** | `src/aurora_mjo/dataset.py` (static loading only), `src/aurora_mjo/model.py` (static variable list only), `configs/unified.yaml`, `scripts/fetch_sst.py` (new), `data/static/` , `tests/test_static_vars.py`, `docs/SPEC.md`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `_build_aligned_index` (Lesson 5). `src/aurora_mjo/loss.py`, `rmm/`, `trainer.py`. Any `results/*.md`. |

## Objective

Sea surface temperature enters the model as a **static variable held fixed
through each forecast**, initialised from the observed field at `t₀`. The model
gains the single boundary condition that distinguishes one ENSO state from
another over a sub-seasonal rollout.

## Why this task exists, and why it is a precondition for any ENSO claim

**Aurora v1.0 has no ocean.** Measured from `microsoft-aurora==1.8.0`:

```python
surf_vars   = ("2t", "10u", "10v", "msl")      # our config adds ttr, tcwv
static_vars = ("lsm", "z", "slt")
atmos_vars  = ("z", "u", "v", "t", "q")
```

No SST, no skin temperature, no mixed-layer state. The LANL archive
documentation contains **zero** mentions of SST
(`docs/nersc-dataset-information.md`, grepped 2026-09-13).

Over a 6-hour step this is immaterial; SST is nearly constant. Over **120 steps
it is the dominant boundary forcing on tropical convection**, and it is the only
field that encodes which ENSO state the forecast is in. Without it the
atmospheric initial condition decorrelates in roughly 10–15 days and the rollout
relaxes toward the model's own ENSO-agnostic climatology.

The consequence is specific: **J7's ENSO stratification cannot produce a
meaningful result on a model with no ocean.** Any skill difference it found
between El Niño and La Niña cases would be a property of the initial condition,
not of learned ENSO-conditional behaviour. G5 is what makes that experiment
interpretable.

## Why static, not a surface variable

Three reasons, and they compound:

1. **It matches S2S practice.** Persisting the SST anomaly through a sub-seasonal
   forecast is the standard convention at sub-seasonal lead, because SST
   anomalies persist on roughly monthly timescales. At 30 days this is
   defensible; beyond ~45 days it is not, and the campaign stops at 30.
2. **No decoder head.** A surface variable needs a randomly initialised output
   head, which is a new drift source over 120 steps and another channel competing
   in the loss. A static needs an encoder embedding only.
3. **No drift by construction.** A prognostic SST would be free to wander, and a
   wandering ocean under a 30-day rollout would be worse than no ocean at all.

Aurora repeats statics across the batch and history dimensions in `forward`
(`aurora/model/aurora.py:290`), so a per-forecast static is mechanically
identical to `lsm` — it is simply refreshed per sample rather than loaded once.
That is the one structural change: `_load_static_vars` currently loads three
fields at `__init__`; SST must be read per sample at `t₀`.

## What you may assume

- G1: native 1°, grid read from the archive (`results/G1_result.md`).
- G2: `model_type: full` → `AuroraPretrained` (`results/G2_result.md`).
- `lsm` is already loaded and gives the ocean mask for free — SST is undefined
  over land and must be filled, not left as a fill value.
- **Q-20 governs where SST comes from.** It is not in the LANL archive.
- The injected-variable machinery in `freeze_backbone` (`model.py:298`) handles
  *surface* variables. A new static needs the equivalent treatment for
  `encoder.static_token_embeds` (or whatever the static embedding is named) —
  check, do not assume the surface path applies.

## Steps

1. **Resolve Q-20 before writing code.** Source ERA5 `sst` (or `skt` masked to
   ocean) at 6-hourly or daily resolution for 1980–2019, regrid to the G1 1° grid,
   and record provenance exactly as G1 did for `slt`. Daily is sufficient — the
   field is persisted anyway.
2. Decide and justify the **land fill**. Options: fill with `2t`, fill with a
   zonal-mean SST, or fill with a constant. The model sees `lsm` so it can learn
   to ignore land values, but a fill of `0 K` or a NetCDF fill value would be a
   catastrophic outlier under normalisation. **State which you chose and the
   σ-value of the worst land point.**
3. Add SST to the static set. `_load_static_vars` loads three time-invariant
   fields at `__init__`; SST is time-varying between samples and fixed within a
   forecast, so it must be read in `__getitem__` at `t₀` and placed in
   `static_vars`. **Keep the read fork-safe** — open, read, close, as every other
   read in this file does.
4. Extend `freeze_backbone` so the new static's embedding is trainable (it is
   randomly initialised; the pretrained checkpoint has no entry for it). Verify
   this against the static embedding module, not by analogy with the surface path.
5. Compute normalisation statistics for SST over 1980–2015 using G3's
   `aurora_mjo.stats` and add them to `configs/norm_stats_1980_2015.yaml`.
6. **Confirm the persistence semantics in a rollout.** `_advance_batch` carries
   `in_batch.static_vars` forward unchanged, so persistence should be automatic —
   verify it rather than assume, by asserting the SST tensor is bitwise identical
   at step 0 and step 119.
7. Forward-pass check with `model_type: full` on a real sample. Confirm finite
   output and that the parameter count rose by exactly the new embedding.
8. **Record the limitation explicitly in `docs/SPEC.md`:** SST is persisted, not
   predicted; the model has no ocean dynamics; this is defensible to ~30 days and
   is a stated boundary of every claim the project makes.

## Definition of Done

- [ ] **Q-20 resolved**; SST source, resolution, retrieval date, licence and
      checksum recorded alongside the file as G1 did for `slt`
- [ ] SST regridded to the G1 1° grid; grid equality with the other statics
      asserted and pasted
- [ ] Land fill decided and justified; **worst land-point σ-value pasted**
- [ ] SST read per sample at `t₀` into `static_vars`; read is fork-safe
- [ ] Static embedding for SST confirmed trainable; the verification method
      stated (not inferred from the surface path)
- [ ] SST normalisation statistics computed over 1980–2015 and committed
- [ ] **Persistence verified**: SST tensor bitwise identical at rollout step 0
      and step 119; assertion output pasted
- [ ] `model_type: full` forward pass finite; parameter delta equals the new
      embedding exactly
- [ ] Physical range check: SST ∈ [271, 310] K over ocean, per
      `03_DOMAIN_PRIORS.md` §3
- [ ] `docs/SPEC.md` records the persisted-SST limitation and its 30-day bound
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Discontinuity section — parameter count, static variable set
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Making SST prognostic.** That is an ocean model and it is not this project.
- Coupling, mixed-layer schemes, or any SST tendency.
- Damped-persistence SST (relaxing the anomaly toward climatology with lead). It
  is the obvious next refinement and it is a **later-campaign** item — record it
  as such. Simple persistence first, so the comparison exists.
- Adding any other surface or static variable.
- Using SST to condition the loss or the sampler. It is an input, not a label.
