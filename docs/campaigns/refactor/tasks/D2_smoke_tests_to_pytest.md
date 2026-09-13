# D2 — Convert the six existing verify/smoke scripts to `pytest`

| | |
| --- | --- |
| **Phase** | D |
| **Depends on** | D1 (`PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 3 h |
| **Touches** | `tests/test_dataset_loader.py`, `tests/test_shapes.py`, `tests/test_freeze.py`, `tests/test_mjo_head.py`, `tests/test_rollout.py`, `tests/test_bad_values.py` (all new), `scripts/archive/` (the converted originals move here) |
| **Must not touch** | anything under `src/`. `tests/conftest.py` beyond adding a fixture if genuinely needed — if you add one, say why. |

## Objective

At the end, the checks currently living in six standalone scripts are `pytest`
tests that assert rather than print, run on the default CI path where possible,
and the originals are archived with their answers written into their docstrings.

## Why this is a separate task

`cleanup-2026-09.md` §9 Phase C got this right: **this is conversion work, not
greenfield.** Six scripts already encode what the team wanted to verify. The
value here is turning `print` into `assert` and `sys.exit(1)` into a failing
test — cheap, high-coverage, low-risk. Separating it from D3 (which writes new
tests for the paid-for lessons) keeps the mechanical work away from the
thinking work.

## What you may assume

The six sources, and what each currently does:

| Script | Lines | What it checks | Needs |
| --- | --- | --- | --- |
| `scripts/verify_dataset_loader.py` | 66 | dataset init, static stats, sample 0 shapes | data |
| `scripts/verify_shapes.py` | 98 | tensor shapes against Aurora's expectations | data |
| `scripts/smoke_test_freeze.py` | 162 | `freeze_backbone` freezes the right params; LoRA identified by module type | GPU? |
| `scripts/smoke_test_mjo_head.py` | 128 | MJO head constructs and forwards | GPU? |
| `scripts/smoke_test_rollout.py` | 243 | rollout stepping, detached backprop | GPU |
| `scripts/scan_for_bad_values.py` | 120 | non-finite scan over fields | data |
| `scripts/archive/verify_dataset_smoke.py` | 23 | (superseded by row 1; C3 archived it) | — |

D1's fixtures make rows 1, 2 and 6 runnable **offline** against synthetic data.
Rows 3 and 4 may run on CPU — **measure it, do not assume.** Row 5 almost
certainly needs a GPU.

D1's result file lists **which shape assertions are impossible on synthetic
fixtures** because the grid is reduced. Read it before writing any shape test.

Markers available: `needs_data`, `needs_gpu`, `slow`, `live`
(`01_TARGET_STATE.md` D6). The default CI path excludes the first, second and
fourth.

## Steps

1. Read all six scripts before writing anything. For each, list in your result
   file: what it verifies, whether that verification is an assertion or just a
   print, and whether it needs real data, a GPU, or neither.

   Several of them only print. **A print is not a check** — converting it means
   deciding what the correct value is. Where `03_DOMAIN_PRIORS.md` gives you a
   number, assert it. Where it does not, assert the weakest meaningful property
   (finite, right dtype, right rank, in a plausible range) and say in the test
   docstring that a tighter assertion was not available.

2. **Prefer the default path.** For each converted test, aim for no marker at
   all. Add a marker only when you have **measured** that it cannot run
   otherwise. Record the measurement.

   The reasoning is in `04_AGENT_PROTOCOL.md` §3: if a check cannot run without
   real ERA5 data, that usually means the logic under test is entangled with
   I/O. Where you find that, write the test at the boundary you *can* reach and
   record the entanglement as an Observation for a later campaign.

   Where a check genuinely needs real data, write **both**: a synthetic version
   on the default path and a `needs_data` version asserting the real numbers
   from `03_DOMAIN_PRIORS.md` (1,462 samples; `z` mean 3709.2466; `lsm` mean
   0.3357; `slt` mean 0.6708). The `needs_data` one is what a human runs on
   Perlmutter; the synthetic one is what stops a regression reaching them.

3. Convert, one test module per script:

   - **`tests/test_dataset_loader.py`** from `verify_dataset_loader.py`. Assert
     dataset length, static-var presence and finiteness, that sample 0 returns a
     `(Batch, dict, dict)` triple, that surface inputs carry a 2-timestep axis
     and targets do not, and that `metadata.atmos_levels` is the expected
     13-tuple. The level tuple is exact and grid-independent — assert it even on
     synthetic fixtures.
   - **`tests/test_shapes.py`** from `verify_shapes.py`. Assert **rank and axis
     ordering**, not absolute sizes, on synthetic fixtures. Put the absolute
     720×1440 assertions in a `needs_data` test. Note that statics *are*
     upsampled to 720×1440 by the loader even from a reduced grid, so those
     shapes **are** assertable on synthetic fixtures — verify that claim rather
     than trusting it.
   - **`tests/test_freeze.py`** from `smoke_test_freeze.py`. The most valuable
     of the six: assert that `freeze_backbone` leaves the expected small
     trainable count, that LoRA params are identified **by containing module
     type** rather than by name (the existing code is explicit about this and
     `model.py`'s comment says why), and that trainable count is a tiny fraction
     of total. `03_DOMAIN_PRIORS.md` §7 gives 41,008 trainable for `small` —
     assert it if the model can be built in this environment, otherwise assert
     the ratio and say why.
   - **`tests/test_mjo_head.py`** from `smoke_test_mjo_head.py`. Assert the head
     constructs from config, output shape matches (RMM1, RMM2, amplitude, active
     status), outputs are finite, and — importantly — that with
     `mjo_head.enabled: false` **no head is constructed at all**.
   - **`tests/test_rollout.py`** from `smoke_test_rollout.py`. Mark `needs_gpu`
     and `slow` unless measurement says otherwise. Assert the step curriculum
     advances as configured and that `backprop: "detached"` produces memory
     roughly flat in `k` — if a memory assertion is too flaky, assert the
     structural property (state is detached between steps) instead and say so.
   - **`tests/test_bad_values.py`** from `scan_for_bad_values.py`. Assert
     finite-fraction is 1.0 on synthetic fields; add a test that **injects** a
     NaN and confirms the scan detects it. A detector that has never been shown
     to fire is not known to work.

4. Archive the originals per the build guide: move to `scripts/archive/` and
   **write the answer into the docstring** — what it verified, what the answer
   turned out to be, which test module supersedes it. Do not just move them
   silently; a bare archived script is indistinguishable from an abandoned one.

   Keep `scripts/verify_dataset_loader.py` **in place, not archived**: it
   produces the human-readable report that `docs/verify_output_slt.txt` is built
   from, and that report is cited throughout `03_DOMAIN_PRIORS.md`. Reduce it to
   a thin wrapper if that is clean; otherwise leave it. Say which you did.

5. Verify all three run configurations:

   ```bash
   uv run pytest -q -m "not live and not needs_data and not needs_gpu"   # the CI path
   uv run pytest -q -m "needs_gpu" --collect-only                        # confirm they collect
   uv run pytest -q                                                       # everything, locally
   uv run python scripts/check.py
   ```

   Report **test counts** for each: how many pass on the default path, how many
   are marked `needs_data`, how many `needs_gpu`, how many `slow`. Those four
   numbers are the actual deliverable of this task and they belong in your
   `SUMMARY`.

## Definition of Done

- [ ] All six scripts read and characterised in a table: what it verifies,
      assert-or-print, and what it needs
- [ ] Six test modules created
- [ ] For every marker added, the **measurement** justifying it is recorded —
      no marker added on assumption
- [ ] Real-data numbers from `03_DOMAIN_PRIORS.md` asserted in `needs_data`
      tests where applicable
- [ ] `metadata.atmos_levels` 13-tuple asserted exactly
- [ ] `freeze_backbone` test asserts LoRA identification by module type, not by
      name
- [ ] `mjo_head.enabled: false` → no head constructed, asserted
- [ ] Bad-value detector shown to **fire** on an injected NaN
- [ ] D1's list of impossible-on-synthetic assertions was consulted; any test
      that needed it says so
- [ ] Originals archived with answers in their docstrings; disposition of
      `verify_dataset_loader.py` stated
- [ ] Test counts reported for all four categories, and in `SUMMARY`
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Any edit to `src/`. If a check cannot be written without changing source,
  that is an Observation for E1/E2 — write the closest test you can and record
  the gap.
- Writing the lesson-regression tests (D3). If you notice overlap, D3 owns it;
  do not pre-empt.
- Fixing any failure the new tests expose in existing behaviour. A test that
  documents a real bug should be written to **assert current behaviour** with a
  docstring saying it is wrong and naming the task that fixes it — or marked
  `xfail` with a reason. Do not silently assert the behaviour you wish existed.
- Deleting any script. Archive with an answer, or leave in place.
