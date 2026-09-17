# J2 — **GATE.** `r > 0.95` against the official BoM RMM series

| | |
| --- | --- |
| **Phase** | J |
| **Depends on** | J1 (`PROCEED: YES`) |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 3 h |
| **Compute** | **None.** CPU. Network access once, to fetch the reference series. |
| **Touches** | `data/reference/rmm_bom.csv` (new), `data/reference/README.md` (new), `scripts/fetch_rmm_reference.py` (new), `src/aurora_mjo/rmm/compute.py` (**only** the sign/order transform values), `data/rmm_basis.npz`, `tests/test_rmm.py`, `docs/findings/2026-1x-rmm-validation.md` (new), `docs/PROJECT_STATE.md` |
| **Must not touch** | Any other part of `rmm/`. `src/aurora_mjo/loss.py`, `dataset.py`, `trainer.py`, `model.py`. Any `results/*.md`. |

## Objective

The RMM pipeline reproduces the official Wheeler–Hendon index to `r > 0.95` on
both components over 2016–2019, the sign/order transform that achieves it is
frozen into `rmm_basis.npz`, and the validation is written up as a durable
finding.

## Why this is a gate

Everything after J2 is a number this pipeline produced. J3 measures skill with
it; J4's control baseline is defined by it; every Phase K result is scored
through it; the paper's central claim rests on it.

A pipeline that cannot reproduce a published index from the same input fields
cannot be used to claim anything. If it disagrees with BoM, either the pipeline
is wrong or the index is — and it is not the index.

> **If `r ≤ 0.95` on either component after the diagnostic sequence in Step 5,
> stop the campaign and report to the human. Do not fix forward. Do not proceed
> to J3.**

This is the same discipline as `../refactor/`'s C4: *"If C4 fails, stop the
campaign and report — do not fix forward."* It was right then and it is right
here, for a larger reason.

## What you may assume

- J1 built the pipeline to `02_SCIENTIFIC_CONTRACT.md` §6 and checked six of the
  seven properties in `03_DOMAIN_PRIORS.md` §7 (`results/J1_result.md`).
- J1 left the sign/order transform defaulting to identity. **Setting it is this
  task's job** — `eigh` and SVD fix neither sign nor order, so a mismatch in
  either is expected and is not a failure.
- **Q-15 is hard-blocking** and governs where the reference series comes from and
  lives. `04_AGENT_PROTOCOL.md` §8: a question that would change a number in the
  paper means `BLOCKED`, not a default.
- `03_DOMAIN_PRIORS.md` §7 gives the diagnostic order for a failure. Follow it;
  (1) and (2) are single-character fixes and account for most failures.

## Steps

1. **Resolve Q-15 before fetching anything.** Source, licence, retrieval date,
   durable home, commit-or-not. Write `data/reference/README.md` recording all of
   it plus the exact column semantics of the file. Write
   `scripts/fetch_rmm_reference.py` so the fetch is reproducible and record a
   checksum. **Never re-fetch silently** — a reference series that changes under
   you invalidates every comparison made against it.
2. Align the BoM series to `data/rmm_targets.nc` on a common daily time axis over
   2016–2019. Report the number of matched days and any gaps.
3. Correlate. Report `r` for RMM1 and RMM2 separately, and the bivariate
   correlation, both **before** any sign/order transform and **after**.
4. Determine the transform: which of `{identity, swap}` × `{±1, ±1}` maximises
   agreement. **Eight possibilities; test all eight and paste the table.** A
   transform chosen without seeing the alternatives is a transform nobody can
   audit.
5. **If `r ≤ 0.95` after the best transform**, work the diagnostic order from
   `03_DOMAIN_PRIORS.md` §7 and report what each step changed:
   1. OLR sign — is `OLR = −mtnlwrf` right for this archive?
   2. EOF1/EOF2 swap or sign — covered by Step 4, but re-check with the OLR sign
      corrected.
   3. Missing or misapplied 120-day mean.
   4. Coarsen to 2.5° (144 longitudes) before the EOF, matching WH's original
      resolution.
   Stop after these four. If none reaches 0.95, the task is `RED` and the
   campaign stops.
6. Freeze the winning transform into `rmm_basis.npz` and make it part of the
   stored basis, applied automatically on every projection.
7. Add a test asserting the frozen basis reproduces BoM to `r > 0.95` on a
   committed slice of the reference series, so this cannot silently regress.
   Mark it `needs_data` if the slice is too large to commit; prefer committing a
   small slice.
8. Re-check the seventh `03_DOMAIN_PRIORS.md` §7 property and, now that the basis
   is anchored, re-check the other six. Paste the full table.
9. Write `docs/findings/2026-1x-rmm-validation.md`: the correlations, the
   transform, the diagnostic path if any, and a plain statement of what is now
   trustworthy. This must outlive the campaign directory — it is the document a
   reviewer will ask for.

## Definition of Done

- [ ] **Q-15 resolved**; `data/reference/README.md` records source, licence,
      retrieval date, checksum and column semantics
- [ ] `scripts/fetch_rmm_reference.py` committed and reproducible
- [ ] Matched-day count and gaps over 2016–2019 reported
- [ ] Correlations before and after transform, for RMM1, RMM2 and bivariate,
      pasted
- [ ] **All eight sign/order combinations tested and tabulated**
- [ ] `r > 0.95` for **both** RMM1 and RMM2 — or `RED` with the full diagnostic
      path from Step 5 documented
- [ ] Winning transform frozen into `rmm_basis.npz` and applied automatically
- [ ] Regression test asserting `r > 0.95` against a committed reference slice
- [ ] All seven `03_DOMAIN_PRIORS.md` §7 properties re-checked and tabulated
- [ ] `docs/findings/2026-1x-rmm-validation.md` written
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] `docs/PROJECT_STATE.md` updated with the gate verdict
- [ ] Result file written from `results/_TEMPLATE.md`, with `PROCEED` set
      honestly

## Out of scope

- **Fixing the pipeline beyond the sign/order transform.** If Step 5 identifies a
  structural problem — a missing 120-day mean, a wrong OLR sign — **diagnose it,
  document it, and report `RED`.** J1 owns `rmm/` and a fix belongs in a
  follow-up to J1, reviewed on its own terms. Fixing forward inside the gate
  removes the gate.
- Skill metrics, rollouts, any model (J3, J4).
- Reading test years.
- Relaxing the 0.95 threshold because 0.93 "looks close". The threshold is the
  gate; a gate you move is not a gate.
