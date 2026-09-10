# Repo Build Guide

**Version 2.0 — 2026-09-02.** Canonical copy lives in the `research-standards`
repository. This file is a **vendored copy**: it is committed into every repo so an
agent can read it locally without network access. CI checks it against the canonical
version and fails if it has drifted, so edit the canonical copy and re-vendor — never
edit a repo's copy in place.

**Purpose.** This document is the standard for setting up and building a new Python
data/analytics repository. It is written to be handed to an execution agent in a
**fresh context window** with no other background: everything needed is here.

**Status of this document.** It is a *directive*, not a menu. Where a decision has
already been made it is stated as MUST or DO. Where genuine discovery is required it
is flagged **[PROBE]** — those are the only things the agent should come back and ask
about, and they are deliberately isolated so the rest of the build is unblocked.

**Section numbers are load-bearing.** Code comments, docstrings and result files cite
this document by section (`§4.1`, `§7.2`). Numbering below §2 is unchanged from v1.0;
do not renumber, only append.

### What changed in v2.0

| Change | Sections |
| --- | --- |
| Repos are created from the **template repository**, not scaffolded by hand. The pre-scaffold spec gate is replaced by campaign `000-initial-build`, which produces `docs/SPEC.md` as a deliverable. | §2 rewritten, §2.5 replaced |
| **`pyrefly` replaces `mypy`** as the type-check gate for new repos. | §3.1, §12.2, §15 |
| **One verification command**, `scripts/check.py`, used identically by humans and CI. | §15 |
| **CI is mandatory** and is the required status check on `main`. | §16 (new) |
| **Campaigns** are the standard shape for large work: `docs/campaigns/<YYYY-MM-slug>/`. | §17 (new) |
| `AGENTS.md` and `CLAUDE.md` are required at the repo root. | §18 (new) |
| `.gitattributes` and `.python-version` added to the required file set. | §12.1, §12.5 |

---

## 0. How to use this guide

If you are the execution agent reading this at the start of a new project:

1. Read §1 (principles) and §2 (the planning contract) before writing any code.
2. The repo is **already scaffolded** — it was created from the template repository
   and `bootstrap.py` has renamed the package. Do not re-create the tree by hand.
   Confirm the scaffold is green first: `uv run python scripts/check.py` (§15).
3. Read §3–§6 (`uv`, layout, config, CLI) to understand what is already there and
   why. These sections are now mostly *description of the template* rather than
   instructions to build it.
4. Produce `docs/SPEC.md` (§2.1) as the first deliverable of campaign
   `000-initial-build`. **No feature code before the canonical schema table is
   complete.**
5. Write the **README** and `docs/SETUP.md` (§9) *before* implementing features, so a
   newcomer can run the thing the moment there is anything to run.
6. Implement in phases (§2.4), earliest phase first, shipping working software.
7. Follow §7 (testing), §8 (external systems), §10 (findings docs) throughout.

If you are the human: §2.5 is the checklist of what the agent should hand back before
it starts on feature code, and `NEW_REPO_CHECKLIST.md` in the standards repo is your
own day-one list.

---

## 1. Principles

These are ordered. When two conflict, the earlier wins.

1. **One source of truth for the schema.** Every input normalises *into* one canonical
   shape defined in exactly one file. Downstream code never branches on where data
   came from. Missing fields are explicit typed nulls, never absent columns.
2. **One thin entry point, many tested modules.** The CLI parses flags and calls one
   function. All logic lives in an importable, testable package under `src/`. No
   business logic in the entry point.
3. **Boring, explicit, tested code over cleverness.** If it feeds a model, a report, or
   a trade, correctness and reproducibility beat elegance every time.
4. **Fail loudly, at the boundary, with the data in the message.** An error that says
   *"no rows matched 2026-07-30; this file contains 2026-07-29"* is worth ten that say
   *"no data"*. Never let a wrong-but-plausible result through silently.
5. **Swappable backends and sources.** Local dev store today, warehouse later, via a
   thin abstraction. The migration must be a config change, not a code change.
6. **Modular expansion.** Adding the second product/region/client is a config block,
   never a schema or pipeline change. Design for the second one while building the
   first; do not build the second.
7. **Measure before you optimise, and write down what you measured.** See §11. Most
   wasted engineering time is confident reasoning about an unmeasured system.

---

## 2. The planning contract (do this before any FEATURE code)

> **Changed in v2.0.** In v1.0 this was a gate before scaffolding. Scaffolding is now
> a one-second operation — clone the template, run `bootstrap.py` — so gating it
> bought nothing and delayed the moment there was something runnable to point at.
>
> The gate now sits before **feature code**, and the spec is the first deliverable of
> campaign `000-initial-build` (§17). Everything else in this section is unchanged,
> because the spec's *content* was never the problem.

### 2.1 The spec document

The first artifact is `docs/SPEC.md`. It is the reference every later decision points
back to, and unlike a campaign document it is **durable**: campaigns get archived when
they end, the spec is maintained for the life of the repo, because it is the contract
other repos read. It MUST contain:

| Section | Contents |
| --- | --- |
| Purpose & audience | What this exists to do, who consumes the output |
| Domain background | Enough context that someone new can follow the vocabulary |
| Sources | Every input, its verified schema, its quirks, what it uniquely provides |
| Canonical schema | The full column table: name, type, source availability, derivation |
| Repository layout | The literal directory tree (§4) |
| Transforms | Each derived column, its rule, its edge cases, its tie-breaks |
| Storage | Backend choice, write semantics, natural key, idempotency strategy |
| Config & CLI | The flag table, the validity matrix, example invocations |
| Non-goals | What is explicitly **not** being built, and why |
| Phased delivery | Ordered phases, each ending in something that runs (§2.4) |
| Testing bar | What must be tested and to what standard |
| Open items | **[PROBE]** items and questions for the human |

### 2.2 Verify schemas against real data, not documentation

Never write a source parser from a description. Obtain a real sample, inspect it, and
record in the spec:

- exact column names, **as they actually appear**, including case and spacing
- row counts, and counts of each significant subtype
- the **candidate key**, and proof it is unique on the real sample
- which columns are attributes vs part of the key
- units (percent vs fraction; the single most common silent corruption)
- which fields are *absent* — and say so explicitly, so nobody assumes they exist

Ship the real sample as a test fixture and assert those exact numbers (§7).

### 2.3 Mark unknowns, don't guess past them

Anything that cannot be known without touching the live external system gets a
**[PROBE]** tag and a Phase 0 (§2.4). Do not invent a plausible value and proceed;
do not block the whole build on it either. Isolate it.

### 2.4 Phases

Phases are ordered so that **understood work ships first** and unknowns come second.
Every phase ends in something a user can run.

```
Phase 0   Discovery of [PROBE] items. Throwaway scripts, no production code.
          Output: filled-in config + a findings doc. No exceptions.
Phase 1   The best-understood path, end to end, into local storage, with tests.
Phase 2   The hard/unknown path (usually the external API), now unblocked by Phase 0.
Phase 3   Remaining sources; parity tests against any legacy system.
Phase 4   Backend migration (local -> production warehouse). No pipeline changes.
Phase 5   Expansion to further products/regions. Config only.
```

**Rule:** do not begin Phase N+1 until Phase N's deliverable runs and its tests pass.

### 2.5 What the agent hands back before writing feature code

- [ ] `uv run python scripts/check.py` green on the freshly bootstrapped scaffold,
      with the summary table pasted (§15)
- [ ] `docs/SPEC.md` (§2.1)
- [ ] The canonical schema table, complete, verified against a real sample (§2.2)
- [ ] The **[PROBE]** list, and what the discovery phase will do about each
- [ ] The phase plan with a concrete deliverable per phase
- [ ] Open questions for the human, each with the agent's proposed default so work is
      not blocked while waiting

---

## 3. Environment: `uv`, always

All dependency and environment management goes through [`uv`](https://docs.astral.sh/uv/).
No `pip`, no `venv`, no `poetry`, no `conda` — mixing them is how lockfiles drift.

### 3.1 `pyproject.toml`

```toml
[project]
name = "your-project"
version = "0.1.0"
description = "One clear sentence."
requires-python = ">=3.11,<3.12"     # pin a NARROW range; see note below
dependencies = [
    # Runtime only. Everything imported by src/ or scripts/ at runtime.
]

[dependency-groups]
dev = ["ruff>=0.6", "pyrefly>=1.0", "pytest>=8.3", "pre-commit>=3.8"]
notebook = ["jupyter>=1.0"]

[project.optional-dependencies]
# Platform- or capability-specific extras the average contributor does NOT need.
# e.g. windows-only vendor bridges, heavy optional readers.

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers"
markers = [
    "live: needs a live external system. Excluded from CI.",
    "needs_data: needs a real local store that is not committed. Excluded from CI.",
    "slow: takes more than a few seconds.",
]

[tool.pyrefly]
project-includes = ["src", "scripts", "tests", "run.py"]
search-path = ["src"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/your_package"]
```

**Narrow Python pin.** `>=3.11,<3.12` not `>=3.11`. A vendor wheel or a compiled
dependency that only exists for one minor version will otherwise fail on a colleague's
machine months later, far from the decision that caused it.

**`.python-version` MUST exist too**, holding the same minor version. `requires-python`
tells a *resolver* what is acceptable; `.python-version` tells `uv` which interpreter to
actually fetch and use. Without it, `uv` picks whatever it finds first, which on a
Windows machine with two Pythons installed is a coin toss — and the resulting failure
appears as an unrelated import error weeks later.

**Type checking is `pyrefly`, not `mypy`, in new repos.** Changed in v2.0. Pyrefly
reached stable 1.0 in May 2026, is 10–50x faster, scores higher on the typing-spec
conformance suite, and checks unannotated code that mypy skips by default. That last
property is the reason for the switch and also the reason it is **new repos only**:
turning it on over an existing codebase surfaces a wave of findings in code that mypy
passes today. Existing repos (`ice-options-db-build`, `dumbo`, `timothyq`) keep `mypy`
as their gate until deliberately migrated. Do not run both as gates in one repo — the
day they disagree about a line, neither is trusted.

### 3.2 The dependency rule that catches everyone

> **Every module imported at runtime MUST be a direct entry in `[project.dependencies]`
> — even if it currently arrives via another package's extra or via the dev group.**

This is not pedantry. The classic failure: config validation uses `pydantic`, but
`pydantic` is only reaching the runtime through a dev-group testing library. Everything
works locally. Then someone runs `uv sync --no-dev` in CI or on a server and the entry
point crashes on import.

**Audit command — run it before every release:**

```bash
uv sync --no-dev --reinstall && uv run python -c "import your_package"
uv run --no-dev pytest tests/ -q          # if tests are runnable without dev extras
```

When you add a dependency because of this rule, **comment why** in `pyproject.toml`:

```toml
"pydantic>=2.8",     # ADDED. config.py validates with pydantic, but pydantic was
                     #   reaching the runtime ONLY through the DEV-group test lib.
                     #   `uv sync --no-dev` broke the entry point before this line.
```

### 3.3 Commands

```bash
# First-time setup (contributors)
uv sync --all-groups

# Runtime only, as production/CI installs it
uv sync --no-dev

# With a platform-specific extra
uv sync --extra <extra-name>

# Run anything (never activate the venv manually)
uv run python run.py --help
uv run pytest -q
uv run ruff check . && uv run ruff format .

# Add / remove
uv add polars
uv add --dev pytest-cov
uv remove somelib

# Upgrade one package, or everything, then TEST before committing the lock
uv lock --upgrade-package polars
uv lock --upgrade
```

**`uv.lock` is committed.** It is the reproducibility guarantee. Never hand-edit it.
A lockfile change is a reviewable event: it goes in its own commit with a note on why.

---

## 4. Repository layout

```text
your-project/
├── AGENTS.md                   # agent entry point (§18). READ FIRST.
├── CLAUDE.md                   # one-line pointer to AGENTS.md (§18)
├── README.md                   # what/why + quickstart + pointers (§9)
├── run.py                      # THIN entry point: parse flags -> pipeline.run(cfg)
├── bootstrap.py                # template rename; DELETES ITSELF on first run
├── pyproject.toml
├── uv.lock                     # committed
├── .python-version             # the interpreter uv fetches (§3.1)
├── .gitignore
├── .gitattributes              # line-ending normalisation (§12.1)
├── .env.example                # documents required env var NAMES; NEVER commit .env
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/ci.yml        # THE required status check (§16)
│   ├── pull_request_template.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── .vscode/                    # settings.json + extensions.json, both committed
├── config/
│   ├── settings.yaml           # environments/targets, paths, tunables
│   └── domain.yaml             # per-product/region/client config blocks
├── src/
│   └── your_package/
│       ├── __init__.py
│       ├── config.py           # CLI flags + YAML -> ONE validated Config object
│       ├── schema.py           # CANONICAL / DERIVED / PUBLISHED. Source of truth.
│       ├── sources/
│       │   ├── base.py         # abstract Source -> canonical frame
│       │   └── <source>.py     # one per input; parse/rename/type ONLY
│       ├── transforms.py       # pure frame -> frame derived-column functions
│       ├── storage/
│       │   ├── backend.py      # backend factory (local | production)
│       │   └── writer.py       # create-or-upsert, idempotent, backend-aware
│       └── pipeline.py         # orchestrate: fetch -> transform -> write
├── scripts/                    # run directly, NEVER imported
│   ├── check.py                # THE verification gate (§15)
│   ├── probe_*.py              # discovery
│   ├── diagnose_*.py           # targeted fault characterisation
│   ├── time_*.py               # performance measurement
│   └── archive/                # spent scripts, answers written into their docstrings
├── data/                       # gitignored; local stores, drop zones, caches
├── docs/
│   ├── REPO_BUILD_GUIDE.md     # THIS FILE, vendored (§18)
│   ├── SPEC.md                 # the design document (§2.1)
│   ├── SETUP.md                # clone -> green tests (§9.1)
│   ├── ARCHITECTURE.md
│   ├── CLI.md
│   ├── PROJECT_STATE.md        # living status + the next action (§9.3)
│   ├── campaigns/              # one directory per large piece of work (§17)
│   │   └── _TEMPLATE/
│   ├── findings/               # one document per investigation (§10)
│   └── archive/                # superseded state docs and finished campaigns
└── tests/
    ├── conftest.py
    ├── fixtures/               # REAL samples, committed
    └── test_*.py
```

**Do not build this by hand.** Create the repo from the template repository and run
`bootstrap.py`. The v1.0 instructions for hand-scaffolding (and the PowerShell
walkthrough that went with them) are superseded: a hand-built tree is a tree that
drifts, and the template is continuously verified by its own CI.

### 4.1 The `src/` vs `scripts/` discipline

- **`src/` is imported and tested. Never run directly.**
- **`scripts/` is run directly. Never imported.**

Enforce it. Probes, one-offs, diagnostics, and measurement tools live in `scripts/`;
they may import from `src/`, never the reverse. This keeps the tested surface small and
stops throwaway code accreting into the pipeline.

### 4.2 Sources stay dumb

A source does **parse, rename, type** and nothing else. No derived columns, no business
rules, no cleverness. All derivation lives in `transforms.py`, applied in a fixed order
by `pipeline.py`.

This is what stops two input paths silently drifting apart. If the email path and the
API path each computed their own `is_atm`, they *will* eventually disagree, and you
will find out months later via a wrong number in a model.

Every source fills unavailable canonical columns with typed nulls and sets a
`provenance` column identifying itself.

---

## 5. Configuration

### 5.1 Two YAML files

- `config/settings.yaml` — environments/targets, paths, throttles, tunables.
- `config/domain.yaml` — one block per product/region/client. Adding the second one
  must be a new block here and nothing else.

### 5.2 One validated config object

`config.py` merges CLI flags and YAML into a single validated object (`pydantic` v2).
Validation happens at that boundary and nowhere else. Downstream code receives a
`Config` it can trust and never re-checks.

**Invalid combinations must fail immediately, with a message that says what to do
instead.** Build a validity matrix in the spec and implement it:

| mode \ source | file | api |
| --- | --- | --- |
| `historical` | ✗ (only one day exists per file) | ✓ primary |
| `daily` | ✓ cheap steady state | ✓ when the file lacks a field |

### 5.3 Defaults must be safe, not convenient

A missing value must never be silently replaced with a permissive default that changes
the meaning of the request.

> **Worked example, from a real incident.** A missing `--end` was replaced with
> `date(2100,1,1)`. Harmless for a file source (an unbounded filter over a file holding
> one day is a no-op). For an API it meant a *daily* run requested the entire history of
> every symbol. Nobody noticed for weeks, because it worked — it was just slow.

Rules:

- If a default changes meaning per source, it is not a default. Resolve it per source
  explicitly, in one named function, unit-tested.
- Guard the invariant in **two** places: the config validator and the consumer. Cheap,
  and it survives someone constructing the object directly later.
- Echo the resolved values back to the user at the start of the run
  (`Window: 2026-07-29 (source=api, mode=daily)`). Resolution the user cannot see is
  resolution the user cannot check.

---

## 6. The CLI

`run.py` parses flags, builds `Config`, calls `pipeline.run(cfg)`. Nothing else.

Use `typer`. Design flags as **orthogonal dimensions**, not a pile of switches:

| Flag | Values | Default | Meaning |
| --- | --- | --- | --- |
| `--mode` | `historical`, `daily` | required | write behaviour: create-fresh vs upsert |
| `--source` | per project | required | which input |
| `--target` | name from `settings.yaml` | `local` | resolves the backend |
| `--date` | ISO date | see §5.3 | shorthand for `--start X --end X` |
| `--force` | flag | `false` | allow destructive overwrite |

**Named targets, never raw paths.** `--target prod` resolves through config. A raw
connection string or path on the command line is how production gets clobbered.

Document every flag in `docs/CLI.md` with a worked example per realistic use case.

---

## 7. Testing

### 7.1 The bar

| Requirement | Why |
| --- | --- |
| Real sample as a committed fixture, with exact assertions (row counts, subtype counts, key uniqueness) | Synthetic fixtures encode your assumptions, then confirm them |
| **Format-variance tests** on every parser | See §7.2 — this is the one people skip |
| Idempotency: load twice → identical row count and content | The single most valuable pipeline test |
| Transform unit tests in isolation, including tie-breaks and null handling | Derived columns are where silent wrongness lives |
| Config validation: bad flag combos raise clear errors | Cheap, and it documents the matrix |
| Determinism: same input → same output, except audit timestamps | |
| Parity tests against any legacy system, with **every** difference explained | "Mysterious but small" differences are unexplained bugs |

### 7.2 Test the container, not just the content

> **Worked example, from a real incident.** A parser read dates with
> `str.strptime(col, "%m/%d/%Y")`. The test fixture stored dates as *text*, so it
> passed. Real files from the vendor stored them as Excel *date cells*, so the column
> arrived as a date type and `.str.strptime` raised. Same data, same columns, same
> headers — different cell formatting, total failure.

For every parser, test that these all produce **an identical canonical frame**:

- the native type *and* the string form of every date/number column
- header case, underscore-vs-space, and stray whitespace variants
- a renamed sheet/tab, or reordered columns
- trailing blank rows

**The load-bearing assertion is equivalence:** parse the same data in two encodings and
assert the resulting frames are equal. Formatting must not be semantic.

Be **tolerant about containers, strict about values**: accept any reasonable encoding,
but a value that is present and unparseable must raise — naming the offending cells and
the formats tried — never become a null. A null in a key column silently corrupts
identity and produces rows that can never be upserted over.

### 7.3 Test what production actually runs

> **Worked example, from a real incident.** A retry mechanism shrank a batch size
> mid-run. Its unit test iterated the chunk generator lazily and passed. Production
> wrapped that generator in `list()` for a progress bar, materialising every chunk
> before the first call — so the shrink never applied. The test asserted on a code path
> production did not take, and the mechanism was inert for weeks while looking tested.

When testing a mechanism that depends on evaluation order, laziness, or mutation
timing, exercise it **through the same call path production uses**.

---

## 8. Working with external systems

Assume any external API is undocumented, slow in unpredictable ways, and occasionally
just wrong. The following are not optional extras; they are the difference between a
five-minute diagnosis and a lost day.

### 8.1 Deadline every call

Wrap external calls with a client-side deadline (daemon thread + `join(timeout)`).
A hung call that never returns will otherwise consume its full vendor timeout, and a
job full of them consumes hours. You usually cannot cancel the remote work — but you
can stop waiting, and that is what lets recovery logic actually run.

### 8.2 Distinguish "no data" from "no answer"

This is the highest-value defensive measure in this entire guide.

Many APIs signal a timeout as an *empty but well-formed* response — a header row with
no data. That is indistinguishable from a legitimate "nothing matched". Conflating them
causes **silent data loss**, and in a discovery/probe path it is worse: if an empty
result means "this item does not exist", a timed-out batch reads as a definitive
negative and real items are permanently excluded, with no error anywhere.

Discriminate on **elapsed time**: a genuinely empty result returns in milliseconds; a
timeout takes seconds to minutes. Then:

- no data + slow → treat as failure: retry, split, escalate
- no data + fast → a real empty result, pass through unchanged
- an item that fails alone → **name it in the run summary**, never drop it silently

### 8.3 Recovery ladder

Order recovery steps by cost, cheapest first:

1. **Retry with a different request shape** if one is known to work (one call).
2. **Split the batch** and retry the halves (log₂ n calls) to isolate the culprit.
3. **Quarantine** the culprit for the rest of the run.
4. **Report** everything that failed, with enough context to judge severity.

### 8.4 Quarantine is a circuit breaker, not a blocklist

> **Worked example, from a real incident.** A symbol that hung the API was quarantined
> for 30 days. It turned out to be the front-month at-the-money strike — the single most
> important contract for the model being fed. Worse, the "which strike is at the money"
> calculation picks the nearest strike *present in the data*, so excluding the true ATM
> strike silently re-anchored the entire volatility surface to its neighbour. No error,
> no null, just quietly wrong numbers downstream.

Therefore:

- Quarantine entries **age out fast** (default: one day). They exist to stop one run
  re-isolating the same item repeatedly, not to exclude it permanently.
- **Score every exclusion by domain importance** and warn loudly when a *significant*
  item is missing. "3 far-out-of-range items missing" is a footnote; "the central item
  is missing" is a correctness problem, and the output must say which you have.
- Prefer **characterising and recovering** the fault over routing around it (§11.2).

### 8.5 Fail open on optional optimisations

Caches, screens, and pre-filters must never be able to fail a load. On error, or on a
suspicious result (e.g. a filter that would drop *everything*, far more likely a wrong
field name than an empty universe), fall back to the unoptimised path and print why.

---

## 9. Documentation

### 9.1 The set

| File | Contents |
| --- | --- |
| `README.md` | What this is, why it exists, quickstart, layout, pointers to the rest. Written for someone who has never seen the project **and** for someone maintaining it. |
| `docs/SETUP.md` | Clone → `uv` → green test suite. Every prerequisite. Maintenance commands. Troubleshooting for the errors people actually hit. |
| `docs/ARCHITECTURE.md` | How data moves. A diagram. The core invariants and why the boundaries are where they are. |
| `docs/CLI.md` | Every flag, the validity matrix, worked examples per use case. |
| `docs/SPEC.md` | The design document from §2. Durable: campaigns get archived, this is maintained. |
| `docs/PROJECT_STATE.md` | Living status: what works, what is broken, what is open, **what the next action is**. |
| `docs/findings/*.md` | One per investigation. See §10. |
| `docs/REPO_BUILD_GUIDE.md` | This document, vendored. See §18. |
| `docs/campaigns/<YYYY-MM-slug>/` | One directory per large piece of work. See §17. |
| `AGENTS.md`, `CLAUDE.md` | Agent entry point and its pointer. See §18. |

### 9.2 Write the README before the features

A README written at the end documents what got built. A README written first is a
design tool: if the quickstart is embarrassing to write, the CLI is wrong.

### 9.3 `PROJECT_STATE.md`

The handover document. Keep a status-at-a-glance table (item / state / reference), a
section per issue with root cause and fix, an explicit **OPEN** list, and a single
stated **next action**. When it grows unwieldy, archive it with a date suffix and start
a fresh one that says what it supersedes.

### 9.4 Comments explain *why*, and record what was ruled out

Code says what it does. Comments say why it does it that way, what the alternative was,
and what evidence settled it. Especially: record **rejected hypotheses**, so the next
person (or the next context window) does not re-run the same dead end.

```python
# Batching is capped by symbol COUNT only.
#
# A previous version also capped total request characters, on the theory that the
# abrupt 50 -> 100 cliff looked like a fixed-size transport buffer.
#
# THAT HYPOTHESIS IS REFUTED: a different endpoint moved 500 symbols (~6,000
# characters) through the identical bridge in 1.45s. There is no ~1KB buffer.
# The limit is server-side and endpoint-specific.
```

Record measurements inline where they justify a constant, with the date and the command
that produced them, so nobody "tidies up" a value that is load-bearing.

---

## 10. Findings documents

Every non-trivial investigation produces `docs/findings/<topic>_<date>.md`:

```markdown
# <Title>

**Date:**
**Scope:** files touched
**Status:** FIXED / characterised / open

## Symptom
What was observed, verbatim where possible.

## Root cause
The mechanism. Show the code or the data that proves it.

## Evidence
Measurements, in a table. Include the command that produced them.

## What this RULES OUT
| Hypothesis | Evidence against |

## The fix
What changed and why that specific change.

## Open
What is still unknown, and the cheapest next experiment.
```

**The "rules out" table is the most valuable section.** It is what stops the next
investigation — human or agent — repeating the last one.

**Correct the record when you are wrong.** If a finding is superseded, edit the doc to
say so explicitly and explain what the evidence actually showed. A findings doc that
quietly still asserts a refuted theory is worse than no doc.

---

## 11. Measurement and debugging

### 11.1 Instrument before theorising

When something is slow or flaky, do not reason about it — measure it. Add per-phase
wall-clock accounting to the run itself (one clock read per external call is free) and
print a summary:

```text
ICE call timing --- 21.3s across 52 calls
  autolist:chain                  7.2s  n=1     mean=  7.20s  max=  7.20s
  timeseries:options              6.1s  n=40    mean=  0.15s  max=  0.31s
```

Flag any single call that is an order of magnitude above the norm *as it happens*. A
total tells you nothing about which phase owns it.

### 11.2 Diagnostic scripts: three rules

Learned expensively. A diagnostic that wastes the operator's time is worse than none.

1. **Never re-measure a settled result.** Once something is established, record it in
   the script's docstring as fact and put re-measurement behind an explicit flag.
2. **Deadline every call.** Waiting out a known 300-second timeout teaches nothing.
3. **Print the worst-case runtime before the first call**, computed from the number of
   experiments × the deadline. A tool that might take 30 minutes without saying so will
   be killed halfway and re-run.

### 11.3 Vary one thing at a time, against a control

To characterise a fault, hold everything constant and vary one property per experiment —
with a known-good control to prove the harness itself is sound.

> **Worked example.** One symbol hung an API. Experiments varied: each field
> individually, four different dates, three window shapes, three granularities, and two
> alternative endpoints — each against neighbouring symbols as controls. Result: every
> field hung, every date hung, but weekly granularity worked, the alternative endpoint
> worked, and a *future* end date worked and returned the wanted data in 0.44s. One rule
> covered all 25 observations, and it pointed straight at a one-call fix that recovered
> the data rather than excluding it.

The output should end with a **summary of what worked** — every successful row is a
candidate route to the data.

### 11.4 Beware measuring the wrong variable

> **Worked example, from a real incident.** A diagnostic tested batch sizes 25, 50, 100,
> 200 by taking *prefixes of one symbol list*. 25 and 50 were fast; 100 and 200 hung.
> Reproducibly. The conclusion — "a batch-size ceiling between 50 and 100" — was wrong.
> The list was ordered, and the first bad item sat at position ~60. The experiment had
> measured *where the bad item was*, not *how many items fit*. A real run then fetched
> 50 items instantly and hung on the next 50, refuting it in one line.

Before believing a threshold, ask: **is the thing I varied the only thing that changed?**
Prefixes of an ordered list confound position with size. Randomise, or vary the
composition while holding size fixed.

---

## 12. Hygiene

### 12.1 `.gitignore`

```gitignore
__pycache__/
*.pyc
.venv/
.env
.env.*
!.env.example
secrets/
*.key
*.pem
data/*
*.duckdb
*.parquet
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.DS_Store
```

`uv.lock` **is** committed. Data is not (except deliberate test fixtures).

**`.gitattributes` is required too**, and it is not optional politeness. Without
`* text=auto eol=lf`, a repo edited on both Windows and macOS produces diffs that are
entirely CRLF noise, and — worse — `ruff format --check` can pass locally and fail in
CI on the identical file. If you hit that, run `git add --renormalize .` once.

```gitattributes
* text=auto eol=lf
*.py   text eol=lf
*.md   text eol=lf
*.ps1  text eol=crlf
*.xlsx    binary
*.parquet binary
*.duckdb  binary
uv.lock linguist-generated=true
```

### 12.2 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout      # if notebooks are used
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.19.2
    hooks:
      - id: gitleaks
        exclude: ^uv\.lock$   # lockfile hashes are high-entropy and read as secrets

  # pyrefly runs from the project venv rather than a pinned remote rev, so the hook
  # and scripts/check.py can never disagree about which version ran.
  - repo: local
    hooks:
      - id: pyrefly
        name: pyrefly
        entry: uv run pyrefly check
        language: system
        types: [python]
        pass_filenames: false
```

Install once: `uv run pre-commit install`.

**Pre-commit is a fast subset of the gate, not a substitute for it.** It sees only
staged files, on your machine, with your accumulated state. CI sees the whole repo on a
clean one. Never `--no-verify`.

### 12.3 Secrets

Never in the repo. `.env.example` documents the required variable *names*, never values.
Credentials come from environment variables or a secret manager. `python-dotenv` for
local convenience only.

### 12.4 Changes are delivered as a manifest

When handing back work, state explicitly **which files are new**, **which are modified**,
and note that everything else is unchanged. Never hand over a wholesale replacement of a
repository — it destroys history and makes review impossible. If several rounds occur,
say what changed *since the previous round* as well.

---

### 12.5 The required file set

A repo is not conformant until all of these exist. `NEW_REPO_CHECKLIST.md` in the
standards repo is the tick-list version.

| File | §  |
| --- | --- |
| `AGENTS.md`, `CLAUDE.md` | §18 |
| `README.md`, `docs/SETUP.md`, `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `docs/CLI.md`, `docs/PROJECT_STATE.md` | §9.1 |
| `docs/REPO_BUILD_GUIDE.md` (vendored) | §18 |
| `docs/campaigns/_TEMPLATE/` | §17 |
| `pyproject.toml`, `uv.lock`, `.python-version` | §3 |
| `.gitignore`, `.gitattributes`, `.env.example`, `.pre-commit-config.yaml` | §12 |
| `.github/workflows/ci.yml`, `.github/pull_request_template.md`, `.github/CODEOWNERS`, `.github/dependabot.yml` | §16 |
| `scripts/check.py` | §15 |

---

## 13. Agent working agreement

For the execution agent, in priority order:

1. **Read before writing.** Read the spec, the existing code, and any provided vendor
   documentation before proposing changes. Verify claims against the files.
2. **Verify every edit landed.** After a find-and-replace or patch, re-read the file or
   assert on the result. A silently no-op edit is worse than a failed one: it produces
   confident reports of work that does not exist.
3. **Run the tests after every change.** Report the count.
4. **When measurement contradicts your hypothesis, say so plainly and correct the
   record** — in the response, the code comments, and the findings doc. Do not preserve
   a refuted theory anywhere.
5. **Do not repeat a failed approach.** If something has not worked, or the human has
   said it does not work, change the approach.
6. **Respect the human's time above your own thoroughness.** Do not ask for a
   long-running run to re-derive a known result. Predict and state runtimes.
7. **Flag unverified assumptions explicitly**, especially where a workaround is adopted
   without proof it returns identical results to the correct path.
8. **Distinguish evidence from inference** in every report. "Measured: X" and
   "This suggests: Y" are different claims and must be labelled differently.

---

## 14. Definition of done, per phase

- [ ] **`uv run python scripts/check.py` is green, and its summary table is pasted**
      into the PR and the result file (§15)
- [ ] **CI is green on the branch** (§16). This is the item that is not self-reported.
- [ ] Deliverable runs from a clean clone following `docs/SETUP.md` alone
- [ ] `uv sync --no-dev` then import the package — succeeds (§3.2; CI checks this)
- [ ] New behaviour has tests, including format-variance where a parser was touched
- [ ] Docs updated: README/CLI/ARCHITECTURE as applicable
- [ ] `PROJECT_STATE.md` updated with state and the next action
- [ ] A findings doc exists for any non-trivial investigation, including its
      "rules out" table
- [ ] Change manifest handed over: new / modified / unchanged
- [ ] Open items and unverified assumptions listed explicitly

---

---

## 15. The verification gate

> **New in v2.0.** Replaces "run these four commands and remember the order".

There is exactly **one** command that defines whether the repo is green:

```bash
uv run python scripts/check.py
```

It runs, in order, and reports a pass/fail table:

| Step | Catches |
| --- | --- |
| `uv lock --check` | a dependency added to `pyproject.toml` without re-syncing |
| `ruff check .` | lint |
| `ruff format --check .` | formatting drift |
| `pyrefly check` | type errors |
| `pytest -q -m "not live and not needs_data"` | behaviour |

Options: `--fix` runs ruff's autofix first; `--fast` also skips tests marked `slow`.

**Why a script and not a Makefile.** `make` is not present on Windows by default and
this team develops on Windows. Handing people a tool they must install before they can
verify anything guarantees some of them will not verify anything.

**Why a script and not four commands in a document.** Four commands in a document
drift, and the fourth one gets skipped. One command has one definition, and CI runs
literally it — so a green result locally means a green result in CI, which is the
property that makes the gate worth trusting.

The script keeps going after a failing step rather than exiting on the first one:
seeing every failure in a single run beats discovering them one at a time, six CI runs
deep.

**Test markers.** `--strict-markers` is on, so an unregistered marker is an error
rather than a silent no-op. Three markers are standard:

| Marker | Meaning |
| --- | --- |
| `live` | needs a live external system (vendor API, COM bridge, network) |
| `needs_data` | needs a real local store that is not, and should not be, committed |
| `slow` | more than a few seconds |

The default path must be green with none of those available. If new code cannot be
tested without real upstream data, the logic is in the wrong layer (§7.1).

---

## 16. Continuous integration

> **New in v2.0.** CI is not optional, and it is the required status check on `main`.

### 16.1 What it is, and why this team needs it

CI is `scripts/check.py`, run by GitHub, on a throwaway clean machine, automatically,
on every pull request and every push to `main`. The result is a green check or a red X
on the PR. Nothing is deployed. That is the whole mechanism.

It matters here for one reason above all others: **it is the only claim in this repo
that an agent cannot misreport.** §13.2 already warns that a silently no-op edit
"produces confident reports of work that does not exist", and §14 already lists
Definition-of-Done items whose only evidence is pasted terminal output. A GitHub check
run cannot be pasted. It is the independent second opinion that makes every other
report trustworthy.

Three more things it catches that nothing else does:

- **The clean-clone promise.** §14's first item is "runs from a clean clone following
  `docs/SETUP.md` alone". Nobody has a clean machine, so that item has historically
  been asserted rather than tested. CI is a clean clone, every run.
- **§3.2, mechanically.** A separate job runs `uv sync --no-dev` and imports the
  package. That is the bug that has already broken an entry point on this team once,
  turned into a check that cannot regress.
- **Lockfile drift**, via `uv lock --check`.

### 16.2 Shape

Two jobs, `windows-latest`, plus a cheap drift check on Linux. See
`.github/workflows/ci.yml` in the template.

**Windows, not Linux, deliberately.** The whole team develops on Windows, so a red
result is always a real defect rather than an environment difference. The alternative —
Linux CI against Windows development — produces "passes on my machine, fails in the
cloud", which is exactly the failure a colleague new to this tooling cannot debug and
will therefore learn to ignore. An ignored gate is worse than no gate.

**Concurrency.** `cancel-in-progress: true`, grouped on the ref, so pushing three
commits in a row does not run three full suites.

### 16.3 Cost

Private repos on the Team plan include 3,000 Actions minutes a month. **Included
minutes drain at an OS multiplier: Linux 1x, Windows 2x, macOS 10x.** Overage rates
after the January 2026 repricing are $0.006/min Linux, $0.010/min Windows.

A run here is 3–5 minutes wall clock, so ~8 minutes against the allowance, so ~375 runs
a month before anything is billed. Realistic usage across a handful of repos is well
under that. If it were ever exceeded, 500 runs a month costs about $20.

**Set an org spending limit deliberately.** At the $0 default, exhausting the allowance
*blocks* runs rather than billing them — and a required status check that never reports
blocks every merge in every repo. Silent CI is a worse outcome than a small invoice.

### 16.4 Enforcement

The gate is enforced by an **organisation ruleset** targeting repos by a custom
repository property (e.g. `repo_class = trading-model`), not by per-repo settings.
Rules: pull request required with one approval, `check` must pass, no force push,
linear history.

Use the ruleset's *require workflows to pass* rule, which points at a workflow file in
a central repo. A repo cannot then dodge the gate by deleting its own `ci.yml`, and a
new repo without CI does not brick itself with a required check that never reports.

Keep an admin bypass, and write down in `docs/PROJECT_STATE.md` when it was used and
why. A gate with no documented escape hatch gets worked around informally instead.

---

## 17. Campaigns

> **New in v2.0.** Standardises the shape that `dumbo`'s `docs/expansion/` and
> `timothyq`'s `docs/campaigns/` arrived at independently.

A **campaign** is one large, named piece of work: an expansion, a migration, a research
push, or the initial build. It gets a directory, a base branch, and a fixed document
set, so an agent can be dropped into a fresh context window and be productive with no
other briefing.

```text
docs/campaigns/<YYYY-MM-slug>/
├── README.md                 # purpose, base branch, reading order, task map
├── 00_CONTEXT.md             # what is already true; lessons already paid for
├── 01_TARGET_STATE.md        # the exact end state, decision by decision
├── 02_UPSTREAM_CONTRACT.md   # what upstream guarantees; the joins that lie. Binding.
├── 03_DOMAIN_PRIORS.md       # expectations to sanity-check against. Priors, not truth.
├── 04_AGENT_PROTOCOL.md      # how to run a task; result format; git hygiene
├── QUESTIONS.md              # agents append; the human answers in place
├── tasks/                    # one file per task. READ-ONLY to agents.
└── results/                  # one file per task. Agents write exactly one.
```

Rules that carry most of the value:

- **Agents read `00`–`04` and their own task file only.** Reading ahead produces work
  belonging to another task and merge conflicts belonging to nobody.
- **A task may start only when every task it depends on has a result file with
  `PROCEED: YES`** (or `YES-WITH-CAVEATS` where its own preconditions allow).
- **The first four lines of a result file are machine-read** by the next agent:
  `STATUS`, `PROCEED`, `BLOCKED-ON`, `SUMMARY`. Exact key names, one per line.
- **Write results as you go.** A task that dies at minute 50 of a 60-minute budget with
  nothing written has destroyed all of its own value.
- **Name your gates.** If a task exists to prove a refactor changed nothing, say in the
  task map that nothing after it is trustworthy if it fails.
- **Questions carry a proposed default**, so the campaign does not block on an inbox.

**Phases inside a campaign** follow §2.4: offline foundations, then read-only
discovery, then the offline build, then live execution and acceptance.

**Finishing.** Move the directory to `docs/archive/campaigns/`. Anything that turned
out to be *durably* true is promoted into `docs/SPEC.md` or a findings document rather
than left buried in a result file.

---

## 18. Agent-facing files

> **New in v2.0.** The primary audience for these repos is coding agents, and in v1.0
> nothing in the repo told an agent that this standard existed.

Three files, at the repo root or in `docs/`, all required:

**`AGENTS.md`** — the entry point. Roughly forty lines: read order, the one
verification command, the layout rules that are not negotiable, the never-acceptables,
git conventions, and what to do when stuck. It points at the build guide rather than
restating it. Repo-specific prohibitions go here and should be **concrete**: "never
default a new entity's roll schedule to KC's values — refuse and ask" is enforceable;
"be careful with defaults" is not.

**`CLAUDE.md`** — a pointer to `AGENTS.md`, nothing more. It exists because Claude Code
loads that filename specifically. Do not let two sets of rules grow in the two files:
the day they disagree, neither is trusted.

**`docs/REPO_BUILD_GUIDE.md`** — this document, **vendored** into every repo. An agent
needs to *read* the standard, and a link to another repository is not readable from
inside a sandbox. The canonical copy lives in the standards repo; CI compares the
vendored copy against it and fails on drift. Edit the canonical copy and re-vendor;
never edit a repo's copy in place.

That drift check is what makes vendoring safe. Without it, eleven repos slowly acquire
eleven different standards, all called the same thing.

## Appendix A — Quickstart for the agent

```
1.  Read AGENTS.md, then this guide. Read any provided domain docs and real samples.
2.  Confirm the scaffold is green: uv run python scripts/check.py  (§15)
3.  Write docs/SPEC.md (§2.1): canonical schema table + [PROBE] list.
4.  Hand back the §2.5 checklist. WAIT for approval before feature code.
5.  Write README.md and docs/SETUP.md (§9.2).
6.  Discovery phase: probe scripts for [PROBE] items -> findings doc + filled config.
7.  Build phase: schema -> config -> storage -> best-understood source -> transforms
    -> pipeline -> thin CLI. Tests throughout.
8.  Verify against §14, including green CI. Hand over with a change manifest (§12.4).
9.  Later phases in order. Do not skip ahead.
```

The scaffold in step 2 came from the template repository; you did not build it and you
should not rebuild it. If it is not green on a fresh clone, that is a defect in the
template — report it rather than working around it locally.

## Appendix B — Failure modes this guide exists to prevent

| Failure | Prevented by |
| --- | --- |
| Parser works on the fixture, dies on real files | §7.2 format-variance tests |
| A permissive default silently changes what was requested | §5.3 safe defaults, echo resolved values |
| Timeout indistinguishable from "no data" → silent loss | §8.2 elapsed-time discrimination |
| Excluding a broken item quietly corrupts derived output | §8.4 age-out + importance scoring |
| Confident wrong diagnosis from a confounded experiment | §11.4 vary one thing, §11.3 controls |
| Days lost re-running a diagnostic that re-derives known results | §11.2 the three rules |
| A mechanism looks tested but is inert in production | §7.3 test the real call path |
| `uv sync --no-dev` breaks the entry point | §3.2 direct-dependency rule |
| Two input paths silently diverge on a derived column | §4.2 dumb sources, shared transforms |
| An agent reports GREEN when the suite is red | §16 CI — the one claim that cannot be self-reported |
| "Runs from a clean clone" asserted but never tested | §16.1 CI is a clean clone every run |
| Four verification commands drift; the fourth gets skipped | §15 one gate, one definition |
| `ruff format` passes locally and fails in CI on the same file | §12.1 `.gitattributes` |
| `uv` picks the wrong interpreter on a machine with two Pythons | §3.1 `.python-version` |
| An agent never learns the standard exists | §18 vendored guide + `AGENTS.md` |
| Eleven repos acquire eleven different "standards" | §18 CI drift check on the vendored copy |
| A repo dodges the gate by deleting its own ci.yml | §16.4 org ruleset, required workflow |
| An agent does work belonging to another task | §17 read your task file only |
| The next investigation repeats the last one | §10 findings docs with "rules out" |
| Review impossible because a whole repo was replaced | §12.4 change manifests |
