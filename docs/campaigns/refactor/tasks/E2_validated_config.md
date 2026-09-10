# E2 — One validated `Config` object at the boundary

| | |
| --- | --- |
| **Phase** | E |
| **Depends on** | D1, D3 (both `PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 3 h |
| **Touches** | `src/aurora_mjo/config.py` (new), `src/aurora_mjo/cli_support.py` (shrinks or goes), `run.py`, `pyproject.toml` (one dependency), `tests/test_config_modes.py`, `tests/test_config_validation.py` (new), `docs/CLI.md` |
| **Must not touch** | `configs/unified.yaml` — **its content and resolved shape must not change.** `dataset.py`, `model.py`, `loss.py`, `trainer.py`, `checkpoint.py`. |

## Objective

At the end, config is validated **once, at the boundary**, into a single object
that downstream code can trust and never re-checks. Invalid combinations fail
immediately with a message that says what to do instead.

## Why this is a separate task

It is the only design work in Phase E, it needs D1's fixtures and D3's config
tests to exist first, and it is where an invalid mode combination stops being a
runtime surprise. It is deliberately **after** C4, so a resolution change here
cannot be confused with a refactor artifact.

## Why this is worth doing at all

Today `train.py`'s resolution path is a `_deep_merge` of nested dicts followed by
string-parsed `--override` mutation, and every consumer does
`cfg.get("x", {}).get("y", default)`. That has two costs the repo has already
paid. First, a typo in an override key is silently accepted — nothing validates
that `training.optimzer.lr` is not a real path, so the run proceeds at the
default learning rate and looks fine. Second, defaults are scattered across
consumers, so the true value of a setting is not knowable from the config file.

## What you may assume

- `cli_support.py` holds the seven functions C2 moved verbatim, and its docstring
  says it is a way-station and that **E2 owns `config.py`**. C2 wrote that for
  you.
- **B1's fixtures at `tests/fixtures/baseline/config_*.json` are plain dicts,
  and C4 gated against them.** Your `Config` object must be able to reproduce
  those dicts exactly — that is how you prove you changed nothing.
- D3's `tests/test_config_modes.py` locks: four modes resolve;
  `gradient_checkpointing == false` everywhere; distinct `save_dir` per mode;
  the `init_from` chain; `--override` coercion behaviour as fingerprinted.
  **All of it must still pass.**
- C3 removed the `use_dummy` branch but **left the `data.dummy` block in
  `configs/unified.yaml`** deliberately, so C4's diff would stay clean. C3's
  note says E2 should reject it at validation time.
- `pydantic` v2 is the template's choice. It is not currently a dependency; you
  add it.

## Steps

1. Add `pydantic>=2.8` to `[project.dependencies]`, `uv sync`, and commit
   `pyproject.toml` and `uv.lock` **in the same commit**.

2. Write `src/aurora_mjo/config.py`. Model the existing structure — do not
   redesign it. Sections to model: `experiment`, `data` (with `real` and
   `dummy`), `model` (with `norm_stats` and `mjo_head`), `loss` (with `grid`,
   `spectral`, `mjo_head`, `moisture_budget`), `training` (with `optimizer`,
   `scheduler`, `rollout`), `checkpointing`, `logging`.

   **Field defaults must match today's effective defaults exactly**, including
   the ones currently living in consumers' `.get(key, default)` calls. Go and
   read them; do not guess. Every default you move out of a consumer and into
   the model is a behaviour-preserving change only if you copied it correctly,
   and a `.get()` default you miss becomes a silent change.

   Keep `--mode` overlay resolution and `--override` application **exactly as
   they are** — reuse the moved `_deep_merge` and `apply_overrides` on the raw
   dict, then validate the result. Validate at the end, not during the merge.
   That ordering is what keeps B1 reproducible.

3. **Round-trip proof.** `Config` must expose a method producing the same
   canonical dict B1 captured, so that:

   ```bash
   for m in baseline physics_informed lora combined; do
     uv run python run.py show-config --mode $m > /tmp/e2_$m.json
     diff tests/fixtures/baseline/config_$m.json /tmp/e2_$m.json && echo "$m OK"
   done
   ```

   produces four empty diffs. **This is the gate for this task.** If a diff is
   non-empty you have changed resolution — fix it, or report `RED`. Do not edit
   the fixture.

   If `pydantic` normalises something in a way that makes byte-identical output
   impossible (field ordering, float repr, `None` vs absent), say **exactly**
   what and why, and provide a semantic comparison instead — key sets equal and
   every leaf value equal — with the code you used. Do not quietly relax the
   check.

4. **The validity matrix.** Implement it as validators that fail at config-load
   time with messages saying what to do instead. At minimum:

   | Invalid combination | Message must say |
   | --- | --- |
   | `gradient_checkpointing: true` | it deterministically triggers an illegal memory access on Perlmutter; see the gameplan; the full model does not fit on 4×A100 without it, so use `model_type: small` |
   | `data.use_dummy: true` | the dummy dataset was deleted; use `--smoke-test` |
   | `model.norm_stats.msl` absent | removing it restores a −36 σ input and 100% non-finite validation |
   | `rollout.enabled: true` with `backprop: "full"` | only `"detached"` has been verified crash-free; see `probe_ima_matrix.py` |
   | `rollout.start_steps > rollout.max_steps` | nonsensical curriculum |
   | `val_years` overlapping `train_years` | chronological splits only; this is the 54,060 leakage class |
   | `test_years` overlapping either | same |
   | two modes sharing a `save_dir` | silent checkpoint overwrite |
   | `mjo_head.enabled: false` with `loss.mjo_head.enabled: true` | a loss term scoring a head that does not exist |

   The split-overlap validators are the most valuable — they enforce Lesson 5
   at config-load time rather than at index-build time.

   **`gradient_checkpointing: true` should be an error, not a warning.**
   `AGENTS.md` already prohibits it, and a hard failure at load costs seconds
   while the alternative costs a crashed job and a queue wait.

5. **Unknown-key rejection — the one that catches override typos.** Configure
   the models with `extra="forbid"` so an unrecognised key raises. Then verify
   the current `configs/unified.yaml` still validates: it contains commented
   placeholders and a `data.dummy` block, and if any real key is unmodelled you
   will find out here.

   If a legitimate key must be tolerated, model it explicitly rather than
   loosening `extra`. Loosening it discards the main benefit.

   Test that `--override training.optimzer.lr=1e-5` (note the typo) now **fails
   loudly**. That is the concrete win from this task and it belongs in your
   `SUMMARY`.

6. Reduce `cli_support.py`. The pure config functions move into `config.py`;
   `seed_everything`, `auto_scale_memory`, `_patch_config_for_smoke_test` and
   `_install_smoke_test_loader` are not config concerns. Either leave them in
   `cli_support.py` with an updated docstring saying it is now only smoke-test
   and runtime helpers, or split them further. **Pick one, say which, and update
   the docstring** so the next agent is not reading C2's way-station note about
   a file that has changed purpose.

   `auto_scale_memory` mutates the config after resolution. That is a wart —
   config should be immutable after validation. **Do not fix it here**; record
   it as an Observation with a note that it interacts with
   `gradient_checkpointing` (the config comment says auto-scaling would
   re-enable checkpointing if `backprop: "full"` were used, which the validator
   in step 4 now forbids). That interaction deserves a human's attention.

7. Write `tests/test_config_validation.py`: one test per validity-matrix row
   asserting the failure **and** that the message contains the actionable hint;
   plus the unknown-key rejection; plus the round-trip against B1's fixtures for
   all four modes. Default CI path — no data, no GPU.

8. Update `docs/CLI.md` with the validity matrix. F1 finishes the document; you
   own the matrix because you implemented it.

## Definition of Done

- [ ] `pydantic>=2.8` added; `pyproject.toml` and `uv.lock` in the same commit
- [ ] `src/aurora_mjo/config.py` models every existing section
- [ ] Every default moved out of a consumer `.get()` is listed in the result
      file with its old and new location
- [ ] Overlay and override resolution unchanged; validation happens **after**
      merge
- [ ] **Four empty diffs against B1's fixtures pasted** — or, if byte-identity
      is impossible, the exact reason plus a semantic comparison and its code
- [ ] All nine validity-matrix rows implemented, each with an actionable message
- [ ] `gradient_checkpointing: true` is a hard error
- [ ] `extra="forbid"`; current `unified.yaml` still validates; any key that
      needed explicit modelling is listed
- [ ] `--override training.optimzer.lr=1e-5` fails loudly — output pasted, and
      mentioned in `SUMMARY`
- [ ] `cli_support.py`'s new purpose stated and its docstring updated
- [ ] `auto_scale_memory` mutability wart recorded as an Observation, with the
      `gradient_checkpointing` interaction noted
- [ ] `tests/test_config_validation.py` covers every row plus round-trip; all on
      the default CI path
- [ ] D3's `tests/test_config_modes.py` **still passes unmodified** — or every
      change to it is listed and justified
- [ ] `docs/CLI.md` carries the validity matrix
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Editing `configs/unified.yaml`.** Not to remove the dead `data.dummy`
  block, not to tidy comments, not to replace the placeholder `msl` stats. Any
  edit breaks the B1 round-trip. If you believe a change is needed, that is an
  Observation for F1.
- Any edit to `dataset.py`, `model.py`, `loss.py`, `trainer.py`,
  `checkpoint.py`. Consumers keep reading a dict-like config; converting them to
  attribute access is a follow-on campaign. **Say so explicitly** — the value of
  a validated boundary is realised even if consumers have not migrated.
- Fixing `auto_scale_memory`'s post-validation mutation.
- Splitting `unified.yaml` into two files. `01_TARGET_STATE.md` §8 rejects that.
- Computing real `ps` normalisation statistics.
