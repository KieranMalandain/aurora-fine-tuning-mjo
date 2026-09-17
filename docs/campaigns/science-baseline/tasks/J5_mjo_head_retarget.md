# J5 — Re-point the MJO head at corrected targets; verify; leave it disabled

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J2 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 2 h |
| **Compute** | **None.** CPU, synthetic fixtures. |
| **Touches** | `src/aurora_mjo/model.py` (only `MJOHead` and its construction), `src/aurora_mjo/dataset.py` (only `mjo_targets` assembly), `src/aurora_mjo/trainer.py` (only the `mjo_head` loss branch), `configs/unified.yaml`, `tests/test_mjo_head.py`, `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/rmm/` — frozen by J2. `src/aurora_mjo/loss.py`. Any `results/*.md`. |

## Objective

The MJO head's supervision targets come from J1's corrected RMM pipeline rather
than the 3-element collapse, the head is verified to train on them in a smoke
test, and it remains **disabled** for the whole of Phase K.

## Why this is a separate task

The head is currently supervised against targets that are not the RMM index
(`00_CONTEXT.md` R3). It is disabled, so nothing was trained wrongly — but it is
one config flag from being enabled against broken targets, and the flag is
exactly the kind of thing someone flips when a stage looks stuck.

Fixing the targets is cheap now that J1 and J2 exist. **Enabling** the head is
not cheap: it adds a loss term, a weight, and an ablation to a campaign whose job
is to establish a baseline. `02_SCIENTIFIC_CONTRACT.md` §6.5 keeps it off and
hands enablement to the follow-on campaign, where it has a baseline to be
measured against — which is precisely the post-hoc-projection-versus-latent-
prediction comparison the May research script §III-E was designed for and could
never make.

## What you may assume

- J2 is green; `data/rmm_targets.nc` holds corrected RMM1, RMM2, amplitude and
  phase for 1980–2019 on a daily axis (`results/J2_result.md`).
- The head's architecture — tropical patch pooling, LayerNorm, two-layer MLP,
  zero-initialised output projection — was reviewed and is **sound**. The problem
  was never the head; it was what it was being pointed at.
- The head pools over latitude patches within ±15°, derived from
  `metadata.lat`. **G1 changed the grid from 720 to 180 latitudes**, so
  `patch_size = lat.shape[0] // n_h` and the patch-centre computation both need
  re-checking at 45 latitude patches rather than 180.
- `02_SCIENTIFIC_CONTRACT.md` §6.5: it stays disabled. That is a decision, not an
  oversight.

## Steps

1. Add `mjo_targets` assembly to the dataset: for each sample's target timestep,
   look up `(RMM1, RMM2, A)` from `data/rmm_targets.nc` and attach it to the
   target dict. Targets are **daily**; the model steps 6-hourly. State how the
   mapping is done — nearest day, or the day containing the valid time — and why.
2. **Verify the patch-latitude arithmetic at 1°.** With 180 latitudes and
   `patch_size = 4`, `n_h = 45`; the tropical band ±15° is roughly 30 of 180
   latitudes, hence ~7 patches. Confirm `trop_mask` selects a non-empty,
   plausible set and paste the selected patch-centre latitudes. An empty mask
   produces a `NaN` mean and would be silent.
3. Confirm the loss branch reads `target_dict["mjo_targets"]` correctly and that
   the L1 is against a `(B, 3)` tensor in the right order.
4. Smoke test: enable the head, run a handful of steps on synthetic fixtures,
   confirm the loss is finite and decreasing and that gradients reach the head's
   parameters. **Then set `enabled: false` again** and confirm the model returns a
   bare `Batch` rather than a tuple.
5. Add tests: `mjo_targets` present and correctly shaped; `trop_mask` non-empty at
   1°; the head is exactly neutral at initialisation (the output projection is
   zero-initialised, so it must emit exactly `[0, 0, 0]`); the disabled path
   returns `Batch`, not a tuple.
6. Record, for the follow-on campaign, what enabling it would require: a weight
   in `configs/unified.yaml`, an ablation against post-hoc projection, and a
   decision about whether amplitude is predicted directly or derived from RMM1
   and RMM2 — **the current head predicts all three independently, so nothing
   enforces `A = √(RMM1² + RMM2²)`.** That inconsistency is worth a sentence in
   the result file; it is the sort of thing that produces a head which quietly
   disagrees with itself.

## Definition of Done

- [ ] `mjo_targets` assembled from `data/rmm_targets.nc`; daily-to-6-hourly
      mapping stated and justified
- [ ] Patch-latitude arithmetic verified at 1°; selected patch-centre latitudes
      pasted; `trop_mask` non-empty
- [ ] Loss branch verified against a `(B, 3)` target in the documented order
- [ ] Smoke test with the head enabled: finite decreasing loss, gradients reach
      head parameters; output pasted
- [ ] **`mjo_head.enabled: false` in every mode** after the smoke test;
      `grep` output pasted
- [ ] Disabled path returns `Batch`, not a tuple; tested
- [ ] Zero-initialisation neutrality tested: exactly `[0, 0, 0]` at init
- [ ] Amplitude-consistency inconsistency recorded for the follow-on campaign
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Enabling the head in any mode that Phase K runs.**
  `02_SCIENTIFIC_CONTRACT.md` §6.5.
- Changing the head's architecture, width, dropout or pooling band.
- Adding the amplitude-consistency constraint. Record it; a later campaign
  decides.
- Touching `rmm/`.
- Any training run beyond the synthetic smoke test.
