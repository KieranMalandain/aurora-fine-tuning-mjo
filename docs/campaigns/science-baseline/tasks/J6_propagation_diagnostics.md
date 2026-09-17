# J6 — **GATE.** Propagation diagnostics: does the forecast actually propagate?

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J3 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 5 h |
| **Compute** | **None** beyond reusing J3/J4 rollout output. No new `sbatch`. |
| **Touches** | `src/aurora_mjo/rmm/propagation.py` (new), `scripts/diagnose_propagation.py` (new), `tests/test_propagation.py` (new), `docs/evaluation-spec.md`, `docs/findings/2026-1x-propagation.md` (new), `docs/PROJECT_STATE.md` |
| **Must not touch** | `src/aurora_mjo/rmm/compute.py` and `data/rmm_basis.npz` — **frozen by J2**. `src/aurora_mjo/loss.py`, `dataset.py`, `model.py`, `trainer.py`. Any `results/*.md`. |

## Objective

Every evaluated checkpoint is tested for whether its tropical convection
**propagates eastward** or merely oscillates in place: Wheeler–Kiladis space–time
spectra with an eastward/westward power ratio in the MJO band, Hovmöller
diagrams, phase-speed estimates, net RMM phase-advance rate, and Maritime
Continent crossing success.

## Why this is a gate

**A model can score a high bivariate ACC while having no MJO.**

RMM is a projection onto two EOFs. A *standing* oscillation — convection
amplifying and decaying in place without moving east — excites those two EOFs in
quadrature and is indistinguishable from propagation in the 2-D phase space. This
is not hypothetical: it is the best-documented MJO failure mode in GCMs, where
models produce MJO-band variance that stalls at the Maritime Continent barrier
instead of crossing into the West Pacific.

It is also the failure mode our model is *most likely* to exhibit. A deterministic
model trained on a pointwise reconstruction loss regresses toward the conditional
mean. The conditional mean of an ensemble of propagating MJO events, averaged over
propagation-speed uncertainty, is a **standing** pattern. Amplitude damping
(`02_SCIENTIFIC_CONTRACT.md` §1.4) and propagation loss are two faces of the same
regression-to-the-mean pressure, and ACC is blind to both.

> **If the eastward/westward power ratio in the MJO band is below 1.5 for the best
> Phase K checkpoint, the campaign has not produced an MJO forecast, whatever the
> ACC says.** That is a `RED` on target **T5**, it is reportable and publishable,
> and it must not be buried under a skill curve.

Applying this to the **J4 zero-shot control** as well as to every Phase K stage is
what makes it interpretable: if zero-shot Aurora already fails to propagate, the
campaign's job is to fix propagation, and that becomes the paper's story.

## What you may assume

- J3 built the 120-step rollout harness and fixed the initialisation sampling
  (`results/J3_result.md`). J6 **reuses J3's saved rollout output** and runs no
  new inference. If J3 did not persist gridded fields, that is a `Q-nn` and a
  `BLOCKED`, not a re-run.
- J2's basis is frozen and reproduces BoM to `r > 0.95` (`results/J2_result.md`).
- `03_DOMAIN_PRIORS.md` §10 gives propagation priors: phase speed ~5 m s⁻¹,
  eastward/westward power ratio ~2–3 in observations, RMM phase advance
  ~7.5° day⁻¹, wavenumbers 1–3, periods 30–60 days.
- **Citations here are unverified and fall under Q-12.** Wheeler & Kiladis (1999)
  for the space–time spectral method and the CLIVAR MJO Working Group (2009)
  diagnostics package are the standard references, from memory. Check them before
  either appears in an output.

## Steps

1. Implement space–time spectral analysis on tropical (15°S–15°N meridional mean)
   OLR and `u850` anomalies: 2-D FFT in longitude and time, separate
   eastward- from westward-propagating power by the sign convention, and integrate
   over the MJO band (zonal wavenumbers 1–3, periods 30–60 days).
   **Report the eastward/westward power ratio** for observations, for the J4
   control, and for each Phase K checkpoint.
2. Sanity-check the implementation against a **synthetic pure eastward wave**
   before trusting it on data: it must put essentially all power in the eastward
   half-plane at the injected wavenumber and frequency. A westward wave must do
   the mirror image. This is the test that catches a sign-convention flip, which
   would invert every conclusion in the task.
3. Produce **Hovmöller diagrams** (longitude on x, lead time on y) of tropical OLR
   anomaly for a set of forecast cases initialised in phases 2–3, alongside the
   verifying observations. This is the figure that shows propagation or its
   absence to a reader in one glance.
4. Estimate **phase speed** by lag-correlation of the tropical OLR anomaly across
   longitude, or by the slope of the Hovmöller signal. Report in m s⁻¹ against the
   ~5 m s⁻¹ prior. A forecast that propagates *too fast* is as diagnostic as one
   that stalls.
5. Compute the **net RMM phase-advance rate** in degrees per day, over leads
   5–30 days, for forecasts and observations. A propagating MJO advances
   monotonically at roughly 7.5° day⁻¹; **a standing oscillation has a near-zero
   net rate with sign reversals.** Report the rate *and* the fraction of cases with
   a sign reversal — the second number is the cleaner standing-wave signature.
6. **Maritime Continent crossing.** Select forecast cases initialised in phases
   2–3 (Indian Ocean) with `A(t₀) > 1`, and score the fraction that reach phases
   5–6 (West Pacific) with `A > 1` within 20 days, for observations, the control,
   and each checkpoint. This promotes "skill across Maritime Continent crossing
   cases" from `docs/evaluation-spec.md`'s secondary list to a headline number,
   because it is the specific place MJO forecasts fail.
7. Write `docs/findings/2026-1x-propagation.md`. It must outlive the campaign
   directory — it is the evidence behind target T5 and it is the section a
   reviewer will turn to first.
8. Add the propagation metrics to `docs/evaluation-spec.md` and to J3's output
   JSON schema, so every future evaluation carries them automatically rather than
   as a one-off.

## Definition of Done

- [ ] Space–time spectral analysis implemented; **synthetic eastward and westward
      wave tests pass**, output pasted
- [ ] Eastward/westward MJO-band power ratio reported for observations, the J4
      control, and every available checkpoint
- [ ] **T5 verdict stated**: ratio ≥ 1.5 for the best checkpoint, or `RED` on T5
      with the number
- [ ] Hovmöller diagrams produced for phase 2–3 initialisations, forecast and
      observed side by side
- [ ] Phase speed estimated in m s⁻¹ against the ~5 m s⁻¹ prior
- [ ] Net RMM phase-advance rate in ° day⁻¹, plus the **fraction of cases with a
      sign reversal**
- [ ] Maritime Continent crossing fraction for observations, control and each
      checkpoint
- [ ] `docs/findings/2026-1x-propagation.md` written
- [ ] Propagation metrics added to `docs/evaluation-spec.md` and the J3 JSON schema
- [ ] `data/rmm_basis.npz` unchanged; `git diff --stat` pasted
- [ ] All citations checked against source or explicitly flagged unverified per
      **Q-12**
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing a propagation failure.** If the model does not propagate, that is a
  finding and it belongs in L1's recommendations. Changing the loss or the
  training to chase propagation inside a diagnostic task destroys the diagnostic
  and has no baseline to be measured against.
- Running new inference. Reuse J3's saved output.
- Touching `rmm/compute.py` or the frozen basis.
- Bandpass-filtering in a way that uses data after the forecast valid time. The
  20–96 day filter conventional in this analysis is **non-causal**; applied to a
  forecast it is leakage. Use only backward-looking filters, or apply the filter
  to the full observed record and to the full forecast record independently and
  say exactly what was done.
- Reading test years.
