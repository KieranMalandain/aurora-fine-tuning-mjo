# Campaign: refactor — repository modernisation

**Purpose.** Bring `aurora-fine-tuning-mjo` onto the `research-repo-template`
standard: `uv` instead of conda, a real `pytest` suite, source organised as an
installable package under `src/`, one verification gate, CI, and one coherent set
of agent instructions. It also lands the six already-diagnosed correctness fixes
listed in `00_CONTEXT.md` §2.4, because a test suite built around known-broken
behaviour would encode the bugs.

**What it deliberately does not do.** No science. Findings 3, 5 and 6 from
`AURORA_MJO_GAMEPLAN.md`, the small-vs-full Aurora decision, and computing the
real `ps` normalisation statistics are all **out of scope** and are called out in
`00_CONTEXT.md` §5. No history rewriting. No training run whose purpose is a
result — the only run this campaign performs is the acceptance smoke test in F2.

**Audience.** Autonomous coding agents working one task at a time, plus the human
owner reviewing result files.

**Base branch.** `epic/refactor`. Every task branches off it and merges back into
it. **Nothing in this campaign touches `main`.** The human merges
`epic/refactor` → `main` once, at the end, after F2 is green.

---

## Read these in order, once, before your first task

| # | Doc | Why |
| --- | --- | --- |
| 1 | [`00_CONTEXT.md`](00_CONTEXT.md) | What is already true, and the seven lessons this repo has already paid for. Non-negotiable background. |
| 2 | [`01_TARGET_STATE.md`](01_TARGET_STATE.md) | The exact end state, decision by decision, including where this campaign deliberately diverges from the template. |
| 3 | [`02_UPSTREAM_CONTRACT.md`](02_UPSTREAM_CONTRACT.md) | What the ERA5 archive guarantees and the five alignments that silently return wrong answers. **Binding.** |
| 4 | [`03_DOMAIN_PRIORS.md`](03_DOMAIN_PRIORS.md) | Measured and expected values to sanity-check against. Priors, not truth. |
| 5 | [`04_AGENT_PROTOCOL.md`](04_AGENT_PROTOCOL.md) | How to run a task, result format, git hygiene, Perlmutter rules. |

Also read [`QUESTIONS.md`](QUESTIONS.md) — Q-01 to Q-08 are pre-seeded and each
carries a default the task files already assume.

[`cleanup-2026-09.md`](cleanup-2026-09.md) is the inherited handoff note that
preceded this campaign. `00_CONTEXT.md` summarises what you need from it; read
the original if you need the git-consolidation history.

Then read **the single task file you have been assigned**. Do not read ahead —
look-ahead produces work that belongs to another task and merge conflicts that
belong to nobody.

---

## Task map

Phases are ordered. A task may start only when every task it depends on has a
result file in `results/` with `PROCEED: YES` (or `YES-WITH-CAVEATS` where that
task's preconditions say caveats are acceptable).

### Phase A — Foundations (offline; no data, no GPU, no cluster)

| Task | Title | Depends on |
| --- | --- | --- |
| [`A1`](tasks/A1_uv_migration.md) | conda → `uv`: `pyproject.toml`, lockfile, retire `environment.yml` | — |
| [`A2`](tasks/A2_repo_hygiene.md) | `.gitignore`, untrack artifacts, archive the July metrics | — |
| [`A3`](tasks/A3_verification_gate.md) | `scripts/check.py`, ruff, type checker, CI | A1 |
| [`A4`](tasks/A4_agent_files.md) | One set of agent instructions; fold `.agent/` in | — |

### Phase B — Baseline capture (read-only, needs the cluster)

| Task | Title | Depends on |
| --- | --- | --- |
| [`B1`](tasks/B1_behavioural_baseline.md) | **THE GATE.** Fingerprint current behaviour before anything moves | — (runs under the existing conda env; must precede C1) |
| [`B2`](tasks/B2_july_log_forensics.md) | Grep July SLURM logs for `Using zeros` — answers open question 1 | — |

> **B1 is the campaign's primary gate.** It is the only record of what this code
> did *before* the refactor. **If B1 fails or is skipped, nothing in Phase C is
> trustworthy** — a silent behaviour change in an 1138-line trainer would surface
> weeks later as a wasted 11-hour SLURM slot that nobody connects back to this
> campaign.

### Phase C — Package migration (offline)

| Task | Title | Depends on |
| --- | --- | --- |
| [`C1`](tasks/C1_package_move.md) | `src/*.py` → `src/aurora_mjo/`; re-point 27 imports | A1, A3, B1 |
| [`C2`](tasks/C2_thin_cli.md) | `run.py` thin entry point; `train.py` deprecation shim | C1 |
| [`C3`](tasks/C3_scripts_consolidation.md) | Fold `tools/` into `scripts/`; dead code; RMM extraction | C1 |
| [`C4`](tasks/C4_fingerprint_reassert.md) | **THE GATE.** Re-assert the B1 fingerprint | C1, C2, C3 |

> **C4 is the second gate.** It exists to prove the refactor changed nothing.
> **If C4 fails, stop the campaign and report — do not fix forward.** Nothing in
> Phase D or later is trustworthy on a failed C4.

### Phase D — Test suite (offline)

| Task | Title | Depends on |
| --- | --- | --- |
| [`D1`](tasks/D1_pytest_scaffold.md) | `tests/`, `conftest.py`, markers, synthetic NetCDF fixtures | C4 |
| [`D2`](tasks/D2_smoke_tests_to_pytest.md) | Convert the six existing verify/smoke scripts to `pytest` | D1 |
| [`D3`](tasks/D3_regression_tests.md) | A test for each of the seven paid-for lessons | D1 |

### Phase E — The diagnosed fixes (offline; tests exist by now)

| Task | Title | Depends on |
| --- | --- | --- |
| [`E1`](tasks/E1_env_guards_hard_fail.md) | `HDF5_USE_FILE_LOCKING` at process start; statics fail loudly | D3 |
| [`E2`](tasks/E2_validated_config.md) | One validated `Config` object at the boundary | D1, D3 |
| [`E3`](tasks/E3_slurm_rewrite.md) | Rewrite `eval.slurm`; make every SLURM script consistent | E1, C2 |

### Phase F — Documentation and acceptance

| Task | Title | Depends on |
| --- | --- | --- |
| [`F1`](tasks/F1_documentation.md) | `SPEC`, `SETUP`, `ARCHITECTURE`, `CLI`, `PROJECT_STATE`; README; stale refs | E1, E2, E3 |
| [`F2`](tasks/F2_acceptance.md) | Acceptance on Perlmutter: smoke test end-to-end under the new CLI | F1 |

### Parallelism

A1, A2, A4, B1 and B2 are all independent and can run concurrently. B1
deliberately runs under the **existing conda env**, not `uv` — fingerprinting the
code in a freshly resolved environment would measure two changes at once and
make a C4 mismatch unattributable. C1 must land before C2 and C3, which touch
disjoint file sets and can then run concurrently. D2 and D3 likewise parallelise
after D1. Everything else is sequential.

---

## How the human deploys an agent

```text
You are an agent working on the `aurora-fine-tuning-mjo` repo.

Read AGENTS.md first. Then read docs/campaigns/refactor/00_CONTEXT.md through
04_AGENT_PROTOCOL.md, QUESTIONS.md, and the existing result files in
docs/campaigns/refactor/results/. Do NOT read the other task files in tasks/ —
we do not want look-ahead errors.

Your task is to implement <ID> in docs/campaigns/refactor/tasks/<FILE>.md
exactly as specified. Respect its Touches / Must not touch lists. Follow the git
hygiene in 04_AGENT_PROTOCOL.md: branch off epic/refactor, commit at internal
milestones, do not merge.

Paste the summary table from `uv run python scripts/check.py` into your result
file. A task is not done with a red gate. If a Definition-of-Done item is not
literally true, the status is not GREEN.
```

---

## Finishing

When `epic/refactor` merges to `main`, move this directory to
`docs/archive/campaigns/refactor/`. The `results/` files are the permanent record
of what was measured and what was ruled out — they are why a later agent does not
re-run a dead end.

Anything that turned out to be **durably** true gets promoted into `docs/SPEC.md`
or a document under `docs/findings/` rather than left buried in a campaign
result. In particular: the answer to B2 (were the July runs affected by the
zeroed-statics bug) belongs in `docs/findings/`, because it determines whether any
prior result on this project means anything.