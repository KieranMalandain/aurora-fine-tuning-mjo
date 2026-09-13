# Target state

The exact end state, decision by decision, each with its rationale — so no
agent has to guess at intent and no settled question gets re-opened.

The reference standard is `research-repo-template`. It was written for quant
research repos, so it is applied here **with judgement**: the shape and the
discipline transfer, the specific dependencies and the Windows CI do not.
Section 8 records where this campaign deliberately diverges from the template
and why. Divergence there is a decision, not an oversight — do not "fix" it.

---

## D1. The first-party package is `aurora_mjo`, not `aurora`

**Decision.** Source moves from a flat `src/*.py` to `src/aurora_mjo/`, imported
as `aurora_mjo`, installed with hatchling in editable mode via `uv`.

**Why.** `microsoft-aurora` already owns the top-level `aurora` name and is
imported nine times across six first-party files (`00_CONTEXT.md` §4). A
first-party `aurora` package would shadow it. `aurora_mjo` also matches the
existing conda env name and the repo name, so nothing else has to be renamed.

**How it is verified.** In a clean sync, both of these succeed in the same
interpreter:

```bash
uv run python -c "import aurora, aurora_mjo; print(aurora.__file__); print(aurora_mjo.__file__)"
```

`aurora.__file__` must resolve into site-packages, `aurora_mjo.__file__` into
`src/`. A test in `tests/test_imports.py` asserts exactly this — it is cheap and
it catches the failure mode that would otherwise appear as a confusing
`ImportError: cannot import name 'Batch'`.

---

## D2. Target layout

```text
aurora-fine-tuning-mjo/
├── AGENTS.md                    # agent entry point. Rewritten in A4.
├── CLAUDE.md                    # pointer to AGENTS.md, nothing else
├── GEMINI.md                    # pointer to AGENTS.md, nothing else
├── README.md                    # rewritten for Perlmutter (F1)
├── run.py                       # THIN entry point: parse flags -> call one function
├── pyproject.toml
├── uv.lock                      # committed
├── .python-version              # 3.10
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   └── unified.yaml             # UNCHANGED in shape. Four modes stay.
├── src/aurora_mjo/
│   ├── __init__.py              # exports __version__
│   ├── config.py                # YAML + CLI flags -> ONE validated Config (E2)
│   ├── dataset.py               # from src/dataset.py
│   ├── model.py                 # from src/model.py
│   ├── loss.py                  # from src/loss.py
│   ├── trainer.py               # from src/trainer.py
│   ├── checkpoint.py            # from src/checkpoint.py
│   ├── env.py                   # process-start environment guards (E1)
│   └── rmm/                     # RMM computation + evaluation, extracted from scripts/
│       ├── __init__.py
│       ├── compute.py           # importable core of scripts/compute_rmm.py
│       └── evaluate.py          # importable core of scripts/evaluate_mjo.py
├── scripts/                     # run directly, NEVER imported
│   ├── check.py                 # THE gate
│   ├── calc_norm_stats.py
│   ├── compute_rmm.py           # thin wrapper over aurora_mjo.rmm.compute
│   ├── evaluate_mjo.py          # thin wrapper over aurora_mjo.rmm.evaluate
│   ├── diagnose_*.py            # from tools/
│   ├── probe_*.py               # from tools/
│   └── archive/                 # spent scripts; answer written into the docstring
├── slurm_scripts/
├── tests/
│   ├── conftest.py
│   ├── fixtures/                # synthetic NetCDF, COMMITTED, tiny
│   └── test_*.py
└── docs/
    ├── REPO_BUILD_GUIDE.md      # vendored (A4)
    ├── SPEC.md, SETUP.md, ARCHITECTURE.md, CLI.md, PROJECT_STATE.md
    ├── findings/
    ├── campaigns/refactor/      # this campaign
    └── archive/
```

`tools/` and `notebooks/` disappear. `test_dataset.py` and `inspect_vars.py`
leave the repo root. `environment.yml` is deleted. `.agent/` is folded into
`AGENTS.md` and `docs/` (see D9).

---

## D3. `uv` is the only package manager

**Decision.** `pyproject.toml` + committed `uv.lock`. `environment.yml` deleted.
No pip, no venv, no poetry, **no conda**.

**Why.** The lockfile is the reproducibility guarantee, and `environment.yml`
was never one — it is unpinned and it lies (it lists conda `pytorch` while the
real env has `microsoft-aurora` from pip). The dependency ground truth is
`$HOME/aurora-backup/aurora_mjo-explicit-*.txt`, not `environment.yml`.

**The hard constraint to work outward from:** `microsoft-aurora` on Python 3.10
with **torch 2.5.1+cu121**. That is not on PyPI's default index; it needs an
explicit PyTorch index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

**Why the Python pin is narrow.** `requires-python = ">=3.10,<3.11"`. The
existing env is 3.10 and `microsoft-aurora`'s own support matrix is the binding
constraint. Widening it invites a resolution that has never been tested against
this data.

**How it is verified.** `uv lock --check` passes; `uv sync --all-groups`
succeeds on Perlmutter; `uv run python -c "import torch; print(torch.__version__,
torch.cuda.is_available())"` prints `2.5.1+cu121 True` on a GPU node.

**Migration is not a cutover.** The conda env stays on disk and functional until
F2 passes. Nothing in this campaign deletes it.

---

## D4. `scripts/check.py` is the single definition of "done"

**Decision.** One command, run by humans and by CI:

```bash
uv run python scripts/check.py
```

Steps, in order: `uv lock --check` → `ruff check` → `ruff format --check` →
type check → `pytest -m "not live and not needs_data and not needs_gpu"`.

**Why.** Four commands in a README drift and the fourth gets skipped. A single
script means the human, the agent and CI cannot disagree about what green means.

**Why a Python script and not a Makefile.** Copied from the template and it
still holds: it must run before dependencies are correctly installed, since one
of the things it checks is whether they are. Therefore **stdlib only**.

**How it is verified.** It exits 0 on a clean tree and non-zero with a named
failing gate when any step fails. A6 proves both directions.

---

## D5. Type checking is a ratchet, not a cliff

**Decision.** The type checker runs over `src/aurora_mjo/` and `tests/` and must
be clean there. `scripts/` and `slurm_scripts/` are excluded initially. The
exclusion list lives in `pyproject.toml` with a comment saying it is a ratchet
and that entries come off it, never go on.

**Why.** ~6950 lines of untyped scientific Python, of which `trainer.py` alone is
1138. Demanding full type cleanliness across all of it in one campaign produces
one of two outcomes, both bad: an agent adding `Any` everywhere to get green, or
a red gate that everyone learns to ignore. A ratchet with a shrinking exclusion
list gets there and stays honest on the way.

**Note.** The template specifies `pyrefly`. Whether that is available for Python
3.10 in this environment is unverified — see `QUESTIONS.md` Q-05. Task A3 picks
the tool by *measuring*, and records what it measured.

**How it is verified.** The type step in `scripts/check.py` is green, and the
exclusion list in `pyproject.toml` is shorter at the end of the campaign than at
A3. F1 states the count in `docs/PROJECT_STATE.md`.

---

## D6. Tests are marked by what they need, and CI gets none of it

**Decision.** Four `pytest` markers:

| Marker | Meaning |
| --- | --- |
| `needs_data` | Needs the real CFS ERA5 archive. Excluded from CI. |
| `needs_gpu` | Needs a CUDA device. Excluded from CI. |
| `slow` | More than a few seconds. Excluded by `check.py --fast`. |
| `live` | Needs a live external system (CDS API, HF hub). Excluded from CI. |

`addopts = "--strict-markers"`, so a typo in a marker name is an error rather
than a silently unmarked test.

**Why.** This is the one place the template's assumptions break hardest. A CI
runner has no A100, no CFS mount, and no NERSC credentials. If the default test
path needed any of them, the suite would be permanently red and therefore
worthless. So the default path is **CPU-only, offline, synthetic-fixture**
tests, and everything else is opt-in and run by a human on Perlmutter.

**The corollary, which is the actually important part:** if a piece of logic
cannot be tested without real ERA5 data, **the logic is in the wrong layer**.
Index construction, year-range filtering, timestamp intersection, config
resolution, loss maths and normalisation-stat application are all pure and all
testable on synthetic input. Reach for a `needs_data` marker only after
establishing that the thing under test genuinely cannot be separated from I/O.

**How it is verified.** `uv run pytest -m "not live and not needs_data and not
needs_gpu"` passes on a machine with no GPU and no CFS mount. CI is that machine.

---

## D7. Repository hygiene, with one deliberate exception

**Decision.** Gitignore and untrack: `checkpoints/`, `test_outputs/`,
`tools/probe_results/`, `docs/verify_output*.txt`, `slurm_logs/`, `*.pt`,
`*.nc`, `.venv/`.

**The exception.** `checkpoints/{baseline,lora}/metrics.jsonl` are currently
tracked *on purpose* — the existing `.gitignore` has explicit `!` negations to
keep them. They are the only surviving record of the July runs' loss curves.
Task A2 **moves them** to `docs/archive/metrics/<mode>-2026-07.jsonl` and tracks
them there, then removes the negations. It does not delete them, and it does not
leave them inside an ignored directory where the next `git clean` eats them.

**Why.** The gameplan declares those runs fully non-finite, so the checkpoints
were correctly discarded. The metrics are 21 KB of evidence about *how* a run
went non-finite, which is exactly the kind of thing a later agent should not have
to re-derive. But "tracked file inside a gitignored directory with a negation
pattern" is a configuration that survives exactly until someone simplifies the
`.gitignore`.

**How it is verified.** `git status --porcelain` is empty after A2;
`git ls-files | grep -E '(checkpoints|test_outputs|probe_results)/'` returns
nothing; `docs/archive/metrics/` contains both files with their original byte
counts, and A2's result file states those counts.

---

## D8. `run.py` is thin; `train.py` is retired via a deprecation shim

**Decision.** `run.py` is a `typer` app whose commands parse flags and call one
function each. Business logic moves into `src/aurora_mjo/`. Target surface:

```bash
uv run python run.py train --mode baseline
uv run python run.py train --mode lora --resume auto
uv run python run.py train --mode baseline --smoke-test
uv run python run.py evaluate --mode baseline --checkpoint latest
uv run python run.py norm-stats --years 1980 2015
uv run python run.py show-config --mode combined
```

`--config` defaults to `configs/unified.yaml` rather than being required —
there is one config file and typing its path every time is friction with no
benefit. `--override KEY=VALUE` survives verbatim, because SLURM scripts and
muscle memory depend on it.

**`train.py` is not deleted.** It becomes a six-line shim that prints a
deprecation notice naming the `run.py` equivalent and `exec`s it, preserving
`argv`. It is removed in a later campaign, not this one.

**Why the shim.** `slurm_scripts/train_auto.slurm` invokes `train.py` under
`torchrun`, and a chained SLURM job that a human submitted before the merge and
that lands after it must not fail on an entry point that moved. The cost is six
lines; the cost of the alternative is a wasted 11-hour slot and a 50-hour queue
wait to find out.

**`show-config` is new and worth its keep.** It resolves mode overlays and
`--override` flags and prints the final config, which turns "what did that run
actually use" from archaeology into one command. It is also how C4 and E2 assert
that config resolution did not change.

**How it is verified.** `run.py --help` lists every command; `run.py train
--mode baseline --smoke-test` completes; `run.py show-config` output for all
four modes is byte-identical to the B1 baseline (see D10).

---

## D9. `.agent/` is folded in, and there is one set of rules

**Decision.** `.agent/rules/` (6 files) collapses into `AGENTS.md` plus
`docs/` where content is durable. `.agent/workflows/` (5 files) moves to
`docs/workflows/`. `AGENTS.md` is the single entry point; `CLAUDE.md` and
`GEMINI.md` become one-line pointers to it.

**Why.** There are currently three overlapping instruction sets — `AGENTS.md`,
`GEMINI.md`, and `.agent/rules/*.md` — and they already disagree. `.agent/rules/
01-project-context.md` still describes `dummy_dataset.py` as one of two live
dataloaders and instructs agents to build a `DUMMY` flag around it; that file was
deleted during consolidation. `.agent/rules/03-codebase-standards.md` mandates a
module layout (`models/`, `training/`, `evaluation/`) that this campaign is
replacing. `.agent/rules/06-agent-coordination.md` tells agents to activate a
conda env at `~/miniforge3/envs/aurora_mjo` — wrong manager and wrong path.

The day two instruction files disagree is the day neither is trusted, and that
day has already happened here.

**What survives, because it is genuinely good:** the experiment-registry
discipline from `04-experiment-protocol.md`, the anti-leakage rules from
`02-research-standards.md` (chronological splits only, train-period
normalisation only), and the remote-safety rules from `05-remote-safety.md`
(show the full command before a long job; never kill a running training process;
GPU time is valuable). These go into `AGENTS.md` as repo-specific prohibitions,
stated concretely.

**How it is verified.** `.agent/` does not exist. `grep -ri "dummy_dataset\|
miniforge3\|Bouchet" AGENTS.md CLAUDE.md GEMINI.md docs/` returns nothing outside
`docs/archive/` and this campaign's own history.

---

## D10. Two named gates, and nothing after them is trustworthy if they fail

**Decision.**

- **B1 captures a behavioural fingerprint** of the current code, before anything
  moves: resolved config for all four modes, model parameter counts (total /
  trainable / frozen) for every mode, dataset index statistics for one year,
  static-var shapes and finite-fractions, and the smoke-test loss trajectory
  under a fixed seed. It is written to `results/B1_result.md` and to
  `tests/fixtures/baseline/` as machine-comparable JSON.
- **C4 re-asserts it** after the package move and the CLI change, and must
  produce byte-identical output for the config resolution and exactly-equal
  integers for the parameter counts.

**If C4 fails, stop the campaign and report.** Do not "fix forward". A refactor
of a 1138-line trainer with no test coverage is exactly the situation where a
silent behaviour change costs a wasted 11-hour SLURM slot weeks later, and by
then nobody will connect it to this campaign.

**Why floats are handled differently.** Parameter counts and config keys are
exact. The smoke-test loss trajectory is not — non-determinism in cuDNN kernel
selection and reduction order means bitwise equality is not a fair bar. B1
records the trajectory with a stated tolerance (relative 1e-4) and C4 compares
against that tolerance, not against equality. State the tolerance you used.

**How it is verified.** C4's result file contains a diff of B1 vs post-refactor
for every fingerprint element, and the diff is empty or within the stated
tolerance.

---

## D11. Epic branch, one branch per task, no self-merges

**Decision.** Base branch `epic/refactor`. Every task branches from it as
`epic/refactor/<TASK_ID>-<short-slug>` and merges back into it. **Nothing in
this campaign touches `main`.** `epic/refactor` merges to `main` once, at the
end, by the human, after F2 is green.

**Why this supersedes `docs/git-policy.md`.** That document specifies
`agent branch -> integration/antigravity -> main` and mandates a git worktree per
agent. The worktree-per-agent model is exactly what produced the four-worktree,
three-divergent-branch, one-stash state that the September consolidation spent
nine tasks unwinding. `integration/antigravity` is superseded by
`epic/<slug>`. F1 rewrites `docs/git-policy.md` accordingly; until then, **this
document wins** for anything inside this campaign.

**How it is verified.** `git log --oneline main..epic/refactor` shows one merge
per completed task and no direct commits to `main`.

---

## 8. Where this campaign diverges from the template, deliberately

Do not "fix" these. Each is a decision.

| Template says | Here | Why |
| --- | --- | --- |
| CI on `windows-latest` | `ubuntu-latest` | The template's team develops on Windows. This project is Linux-only, on Perlmutter, and always will be. A Windows runner would test an environment nobody uses. |
| `polars` as canonical frames | `xarray` + `numpy` | The data is n-dimensional gridded NetCDF on a lat/lon/level/time mesh. Polars is a dataframe library; this is not dataframe-shaped data. |
| `src/<pkg>/schema.py`, `sources/`, `storage/`, `transforms.py` | not adopted | That skeleton describes a tabular ETL pipeline with a canonical row schema and idempotent upserts. This is a training pipeline: `dataset.py` → `model.py` → `trainer.py` is the real spine and it is already coherent. Forcing the ETL shape onto it would be cargo cult. |
| `pyrefly` for types | tool chosen by measurement in A3 | Python 3.10 support is unverified. See Q-05. |
| Two config files (`settings.yaml` + `domain.yaml`) | one `configs/unified.yaml` | The four-mode overlay system already works, is documented, and is what the SLURM scripts drive. Splitting it buys nothing and breaks `--mode`. |
| `bootstrap.py` | n/a | For creating a repo from the template. This repo exists. |
| Fixtures are "REAL samples, committed" | synthetic NetCDF, committed | A real ERA5 sample is a ~100 MB NetCDF file on a filesystem this repo must never write to and CI cannot mount. Fixtures are generated by a committed script, so they are reproducible and reviewable, and the script records which real-data properties it reproduces. |
| `data/` gitignored working dir | not created | All data is read-only on CFS at a configured absolute path. A local `data/` directory would only invite someone to copy 1.7 GB into the repo. |

---

## 9. What must NOT change

The invariants this campaign may not break, and what catches it.

| Invariant | Caught by |
| --- | --- |
| `configs/unified.yaml` resolves to identical config for all four modes | B1/C4 fingerprint; `tests/test_config_modes.py` |
| Parameter counts (total/trainable/frozen) per mode are unchanged | B1/C4 fingerprint |
| `gradient_checkpointing` stays `false` everywhere (Lesson 6) | `tests/test_config_modes.py` asserts it explicitly |
| Chronological train/val/test splits — never random (`.agent/rules/02`) | `tests/test_dataset_index.py` asserts no timestamp overlap between splits |
| Train years never include a validation year (Lesson 5) | `tests/test_dataset_index.py` asserts exact sample counts |
| The DDP-collective non-finite gradient guard keeps working (Lesson 2) | `tests/test_grad_guard.py` |
| The SLURM signal contract: USR1 → checkpoint → exit 99; `DONE` marker; `STOP_TRAINING` | F2 acceptance; do not touch `train_auto.slurm`'s trap logic in this campaign |
| `--override KEY=VALUE` semantics | `tests/test_config_modes.py` |
| Nothing writes to the CFS ERA5 archive, ever | `02_UPSTREAM_CONTRACT.md` |