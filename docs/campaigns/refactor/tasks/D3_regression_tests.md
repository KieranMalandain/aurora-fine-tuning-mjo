# D3 — A test for each of the seven paid-for lessons

| | |
| --- | --- |
| **Phase** | D |
| **Depends on** | D1 (`PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 3.5 h |
| **Touches** | `tests/test_dataset_index.py`, `tests/test_config_modes.py`, `tests/test_norm_stats.py`, `tests/test_grad_guard.py`, `tests/test_static_vars.py` (all new) |
| **Must not touch** | anything under `src/`. `configs/unified.yaml`. |

## Objective

At the end, each of the seven lessons in `00_CONTEXT.md` §3 has a test that
fails if the lesson is un-learned. This is the task that stops the campaign's
knowledge from living only in prose.

## Why this is a separate task

D2 converts existing checks; this writes new ones, and it requires actually
understanding the seven failures. It is the highest-value task in Phase D and
the one most damaged by being rushed alongside mechanical work.

## What you may assume

- D1's synthetic fixtures exist, including the **gapped** variant and the
  **differing per-variable chunking**. Both were built specifically for this
  task.
- D1's result file lists what cannot be asserted on synthetic fixtures.
- Exact expected numbers are in `03_DOMAIN_PRIORS.md` §1: 1980 → 1,464
  timesteps → **1,462** samples at k=1; 1980–2015 → **52,596** timesteps; the
  buggy count was **54,060**, which is exactly 1980–2016.
- **Do not fix any bug you find.** E1 and E2 are the fixing tasks. Where current
  behaviour is wrong, assert current behaviour with a docstring naming the task
  that changes it, or use `xfail(strict=True)` with a reason. A test that
  asserts the behaviour you *wish* existed will be deleted by the next agent to
  see it red.

## The seven tests

### Lesson 5 — a glob is not a year filter → `tests/test_dataset_index.py`

The single highest-value test in the repo. `02_UPSTREAM_CONTRACT.md` §4.2.

- Sample count for 1980 alone is exactly **1462**.
- Sample count for 1981 alone (non-leap) is **1458** (1,460 − 2). Derive it,
  state the derivation, and if the measured value differs, work out why before
  asserting — you may have found something the arithmetic misses.
- **Year-range enforcement**: construct over 1980 only, from a fixture archive
  that also contains 1981, and assert **no returned timestamp falls in 1981**.
  This is the test that would have caught the 54,060 bug.
- **Per-variable index independence**: assert every variable has its own
  timestamp map, and that the maps are not the same object. The chunking
  asymmetry D1 built (`t2` yearly, `q` half-yearly) means a shared index would
  produce different results — assert the counts agree despite differing file
  counts.
- **Intersection semantics**: on `synthetic_root_gapped`, assert sample count is
  **reduced by the gap** and that no returned sample spans it. Compute the
  expected reduction from the gap length and assert the exact number.
- **6-hour spacing**: for a handful of samples, assert consecutive timesteps in
  the returned batch are exactly 21,600 s apart.
- **Split disjointness**: build train (1980) and val (1981) datasets and assert
  their timestamp sets are disjoint. This is the anti-leakage invariant from
  `01_TARGET_STATE.md` §9.

### Lesson 6 — checkpointing crashes this machine → `tests/test_config_modes.py`

- All four modes resolve without error.
- **`gradient_checkpointing` is `false` in every resolved mode.** Assert it
  explicitly per mode, with a docstring pointing at the illegal-memory-access
  history. This is a config-invariant test, not a behaviour test, and it is
  cheap insurance against a well-meaning "enable checkpointing to save memory"
  commit.
- Mode chain `init_from` values are as configured: `physics_informed` and `lora`
  warm-start from `checkpoints/baseline`, `combined` from `checkpoints/lora`.
- Each mode's `checkpointing.save_dir` is distinct — two modes sharing a save
  directory would silently overwrite.
- `--override` semantics: dot-notation reaches nested keys, repeated overrides
  apply in order, an override with no `=` raises, and type coercion matches
  whatever `apply_overrides` currently does. **Fingerprint the coercion
  behaviour rather than asserting what it should be** — B1 captured it and C2
  preserved it; your job is to lock it.
- `--mode` omitted against a config with a `modes` block raises `SystemExit`
  listing available modes.
- A mode name that does not exist raises `SystemExit` naming the valid ones.

### Lesson 1 — `msl` is surface pressure wearing an MSL costume → `tests/test_norm_stats.py`

`02_UPSTREAM_CONTRACT.md` §4.1.

- The resolved config **has** a `model.norm_stats.msl` override in every mode.
  Its absence is the −36 σ failure.
- The override is applied to Aurora's `locations`/`scales` dicts by
  `load_model`. Assert the dicts actually change — an override that is read but
  not applied is the worst case, because the config looks right.
- **A guard on the placeholder.** The current values are Aurora's built-in `sp`
  stats, not computed from this data. Write a test that detects them and
  `xfail`s (or emits a warning and asserts the warning) with a message naming
  `scripts/calc_norm_stats.py`. **Do not make this a hard failure** —
  `00_CONTEXT.md` §5 puts computing the real values out of scope, and a hard
  red gate on a known-and-accepted gap trains people to ignore the gate.
- The sigma arithmetic, as an executable version of the reasoning: assert that
  under Aurora's MSL constants a 52,000 Pa input lands beyond −30 σ, and that
  under the override it lands within −6 σ. This turns `03_DOMAIN_PRIORS.md` §4
  into a test and makes the failure legible to whoever next touches
  normalisation.
- **Train-period-only normalisation.** Assert that whatever computes
  normalisation statistics is given only training years. If that is not
  currently assertable because the code path is entangled, record it as an
  Observation — it is the leakage risk flagged in `03_DOMAIN_PRIORS.md` §6.

### Lesson 2 — a finite loss can still produce non-finite gradients → `tests/test_grad_guard.py`

**This guard already exists and works** (`trainer.py` ~line 917). The task is a
test, not a reimplementation. Read the comment block above it first.

- Build a tiny `nn.Module` (not Aurora — this must run on CPU, in the default
  CI path), inject a NaN into a `.grad` after backward, and assert the guard
  detects non-finiteness and **skips** the optimizer step.
- Assert Adam's moment buffers (`exp_avg`, `exp_avg_sq`) are **unchanged** after
  a skipped step. This is the actual failure mode: buffers update before the lr
  multiply, so a poisoned step corrupts them even at lr ≈ 0 during warmup, which
  is why the observed failure was NaN-forever rather than NaN-once.
- Assert the scheduler does **not** step on a skipped batch.
- Assert `_nonfinite_grad_steps` increments.
- Assert a finite-gradient step **does** step, so the test proves the guard is
  selective rather than always-skip.
- The DDP-collective path (`_collective_all_finite`) needs multiple ranks. If it
  cannot be tested single-process, test the single-rank branch and record the
  gap explicitly — do not silently leave the collective logic uncovered.

### Lessons 3 & 4 — HDF5 locking and the zero-substitution fallback → `tests/test_static_vars.py`

`02_UPSTREAM_CONTRACT.md` §4.3–4.5. **E1 changes this behaviour; you test what
exists now.**

- Statics load from the synthetic fixture with correct shapes and finite values.
- `slt` is **truncated, not upsampled**, unlike `z` and `lsm`
  (`03_DOMAIN_PRIORS.md` §3). Assert the asymmetry so nobody "unifies" it.
- `_clean` zeroes NaN/Inf across **all three** statics (the v3 change). Inject
  NaN into a fixture copy and assert it is zeroed, not propagated.
- **The fallback, as it is today**: point `z` at a missing/unreadable file and
  assert current behaviour is a zero tensor plus a `UserWarning` containing
  `Using zeros`. Docstring must say this is **wrong**, that E1 turns it into a
  hard failure, and that this test **will be inverted by E1**. Being explicit
  about that is what stops E1's agent deleting it in confusion.
- A test that `z` all-zeros is detectably wrong: assert the real-data mean is
  far from zero (`03_DOMAIN_PRIORS.md` §3 gives 3709.2466), marked
  `needs_data`. This is the assertion that would have caught the July runs if
  B2 finds they were affected.
- `HDF5_USE_FILE_LOCKING` is set in the environment by the time the dataset
  loads. Today that comes from `conftest.py`; after E1 it comes from the
  package. Assert the variable's value, not its source, so the test survives E1.

### Lesson 7 — plan against measured state → no test

This one is procedural, not code. It is enforced by `04_AGENT_PROTOCOL.md` §2
and by C4's existence. Say so in your result file rather than inventing a test
for it.

## Definition of Done

- [ ] Five test modules created
- [ ] 1980 → 1462 and 1981 → 1458 asserted, with the 1981 derivation shown
- [ ] Year-range enforcement asserted against a fixture containing an adjacent
      year — **the 54,060-class bug is covered**
- [ ] Per-variable index independence asserted against the chunking asymmetry
- [ ] Gapped-fixture sample reduction asserted as an **exact** number, with the
      computation shown
- [ ] 6-hour spacing asserted; train/val timestamp disjointness asserted
- [ ] `gradient_checkpointing == false` asserted in all four modes
- [ ] Distinct `save_dir` per mode asserted; `init_from` chain asserted
- [ ] `--override` coercion behaviour locked against B1's fingerprint
- [ ] `msl` override presence asserted, **and** shown to reach Aurora's
      `locations`/`scales`
- [ ] Placeholder detection present as `xfail`/warning, not a hard failure, with
      the reasoning stated
- [ ] The −30 σ / −6 σ sigma arithmetic asserted
- [ ] Grad guard: skip on NaN, Adam buffers unchanged, scheduler not stepped,
      counter incremented, **and** finite step still steps
- [ ] DDP-collective coverage gap recorded explicitly if untestable
- [ ] `slt` truncate-not-upsample asymmetry asserted
- [ ] `_clean` NaN-zeroing asserted for all three statics
- [ ] Current zero-fallback behaviour asserted, with a docstring saying E1
      inverts it
- [ ] Lesson 7's no-test decision stated
- [ ] Test counts by category reported and in `SUMMARY`
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing any of the seven.** E1 fixes Lessons 3 and 4; E2 hardens config;
  the rest are already fixed or out of scope. Your job is to make un-learning
  them impossible.
- Any edit to `src/` or `configs/unified.yaml`.
- Computing real `ps` normalisation statistics (`00_CONTEXT.md` §5).
- Testing RMM or evaluation code. If C3 reported `AMBER` on the split, that
  code is still in `scripts/` and is not importable — record it as an
  Observation for a follow-on campaign.
