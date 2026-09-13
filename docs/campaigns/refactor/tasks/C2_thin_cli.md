# C2 — `run.py` thin entry point; `train.py` deprecation shim

| | |
| --- | --- |
| **Phase** | C |
| **Depends on** | C1 (`PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 2 h |
| **Touches** | `run.py` (new), `train.py` (becomes a shim), `src/aurora_mjo/cli_support.py` (new), `docs/CLI.md` (new, stub) |
| **Must not touch** | `src/aurora_mjo/{dataset,model,loss,trainer,checkpoint}.py` — **any** edit to these is out of scope. `configs/unified.yaml`. `slurm_scripts/` (E3). |

## Objective

At the end, `run.py` is a thin `typer` app that parses flags and calls one
function per command; the logic currently in `train.py` lives in the package;
and `train.py` is a deprecation shim that still works.

## Why this is a separate task

`train.py` is 411 lines and mixes four concerns: config loading and mode
resolution, override parsing, memory auto-scaling, and a synthetic smoke-test
loader. Untangling that is delicate, and doing it in the same task as the package
move would make a C4 gate failure unattributable.

## What you may assume

- Modules are at `src/aurora_mjo/` and import as `aurora_mjo.<module>` (C1).
- `train.py`'s current surface, which must be preserved semantically:
  `--config` (required today), `--mode`, `--override KEY=VALUE` (repeatable,
  dot-notation), `--smoke-test`, `--resume auto|none|PATH`.
- Functions in `train.py` worth moving rather than rewriting: `_deep_merge`,
  `load_config`, `apply_overrides`, `seed_everything`, `auto_scale_memory`,
  `_patch_config_for_smoke_test`, `_install_smoke_test_loader`.
- `typer` is already a declared dependency (A1).
- **B1 captured `config_*.json` for all four modes plus one override case.**
  Your config resolution must reproduce them byte for byte. That is C4's gate,
  but you should check it yourself before finishing — it is a one-line diff and
  it will save the campaign a wasted gate cycle.
- The target CLI surface is in `01_TARGET_STATE.md` D8.

## Steps

1. Create `src/aurora_mjo/cli_support.py` and move the seven functions listed
   above into it **verbatim** — same names, same signatures, same bodies. Add
   type hints to the signatures only; do not touch the bodies. If a function
   body must change to work at the new location (e.g. a relative path
   assumption), say exactly which and why in your result file.

   Name rationale: the pure config-loading half of this will move into
   `config.py` in **E2**, when it becomes a validated `Config` object.
   `cli_support.py` is a deliberate way-station, not a permanent home. Say so in
   the module docstring so the next agent does not think it is settled.

2. Write `run.py` as a `typer` app. **No business logic** — each command parses
   flags, builds the config, and calls one function.

   ```text
   uv run python run.py train --mode baseline
   uv run python run.py train --mode lora --resume auto
   uv run python run.py train --mode baseline --smoke-test
   uv run python run.py show-config --mode combined
   ```

   Details that matter:

   - `--config` **defaults** to `configs/unified.yaml` rather than being
     required. There is one config file; typing its path every time is friction
     with no benefit.
   - `--override KEY=VALUE` must be repeatable and keep **exactly** the current
     dot-notation semantics, including how it coerces value types. If
     `apply_overrides` has a quirk, preserve the quirk — B1 fingerprinted it.
   - `--mode` stays required-when-modes-exist, raising the same
     `SystemExit` message listing available modes.
   - `show-config` resolves overlays and overrides and prints canonical JSON
     (sorted keys, indent 2) to stdout. This is how C4 diffs against B1, and it
     turns "what did that run actually use" from archaeology into one command.

   `evaluate` and `norm-stats` commands are listed in `01_TARGET_STATE.md` D8
   but belong to **C3**, which owns the scripts. Do not add them here — leave
   the `run.py` structure ready for them and say so.

3. Replace `train.py` with a shim of roughly six lines:

   ```python
   """DEPRECATED. Use `run.py train`. Kept so in-flight SLURM jobs do not fail.

   slurm_scripts/train_auto.slurm invokes this file under torchrun. A chained
   job submitted before this campaign merged and landing after it would fail on
   a moved entry point, costing an 11-hour slot plus another ~50-hour queue
   wait. Six lines is cheaper. Removed in a later campaign; E3 re-points the
   SLURM scripts.
   """
   ```

   followed by the notice print and an exec/delegation into `run.py`'s `train`
   command that **preserves `sys.argv`**. Verify it works under `torchrun`, not
   just under plain `python` — that is how it is actually invoked.

4. Write a stub `docs/CLI.md`: the flag table, one row per flag, with the
   validity notes you know. F1 completes it. A stub now means C3 and E2 have
   somewhere to add their flags.

5. Verify against B1. This is the important step:

   ```bash
   for m in baseline physics_informed lora combined; do
     uv run python run.py show-config --mode $m > /tmp/cfg_$m.json
     diff tests/fixtures/baseline/config_$m.json /tmp/cfg_$m.json && echo "$m OK"
   done
   uv run python run.py show-config --mode baseline \
     --override training.optimizer.lr=1e-5 > /tmp/cfg_ovr.json
   diff tests/fixtures/baseline/config_baseline_override.json /tmp/cfg_ovr.json
   ```

   **All five diffs must be empty.** If any is not, you have changed config
   resolution. Fix it or report `RED` — do not adjust the fixture.

   Also confirm both entry points still parse:

   ```bash
   uv run python run.py --help
   uv run python run.py train --help
   uv run python train.py --config configs/unified.yaml --mode baseline --help
   ```

## Definition of Done

- [ ] `src/aurora_mjo/cli_support.py` holds the seven functions, bodies
      unchanged; any necessary body change listed individually with its reason
- [ ] Module docstring says `cli_support.py` is a way-station and E2 owns
      `config.py`
- [ ] `run.py` exists as a `typer` app with `train` and `show-config`; contains
      no business logic
- [ ] `--config` defaults to `configs/unified.yaml`
- [ ] `--override` semantics preserved exactly, including type coercion
- [ ] `train.py` is a shim of ≤10 lines that prints a deprecation notice and
      delegates with `argv` preserved
- [ ] **Shim verified under `torchrun`**, not just plain `python` — command and
      output pasted
- [ ] `show-config` matches the B1 fixture for all four modes: **four empty
      diffs pasted**
- [ ] `show-config` matches the B1 override fixture: **empty diff pasted**
- [ ] `run.py --help`, `run.py train --help` and `train.py … --help` all work —
      output pasted
- [ ] `docs/CLI.md` stub exists with a flag table
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any edit to `dataset.py`, `model.py`, `loss.py`, `trainer.py` or
  `checkpoint.py`.** If moving CLI logic appears to require one, stop and raise
  a question — it means the coupling is deeper than this task assumes, and that
  is worth a human decision rather than an improvisation.
- Validating the config with `pydantic` (E2). Keep it a plain `dict` for now;
  B1's fixtures are dicts and C4 must diff against them.
- Adding `evaluate` or `norm-stats` commands (C3).
- Editing any SLURM script (E3). The shim exists precisely so you do not have
  to.
- Deleting `train.py`.
- Improving `auto_scale_memory` or the smoke-test loader. Move them; do not
  touch them.
