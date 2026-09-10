# A4 — One set of agent instructions; fold `.agent/` in

| | |
| --- | --- |
| **Phase** | A |
| **Depends on** | none |
| **Base branch** | `epic/refactor` |
| **Budget** | 60 min |
| **Touches** | `AGENTS.md`, `CLAUDE.md` (new), `GEMINI.md`, `docs/REPO_BUILD_GUIDE.md` (new, vendored), `docs/workflows/` (new), `.agent/` (delete) |
| **Must not touch** | any `.py`, `.yaml` or `.slurm` file. `docs/git-policy.md` — F1 rewrites it. Anything in `docs/campaigns/`. |

## Objective

At the end there is exactly **one** set of agent instructions, `AGENTS.md`, with
`CLAUDE.md` and `GEMINI.md` as one-line pointers to it, the build-guide standard
vendored into `docs/`, and `.agent/` gone with its durable content relocated.

## Why this is a separate task

There are currently three overlapping instruction sets that already disagree, and
every agent in this campaign reads them. Fixing that first means the remaining
fifteen tasks get consistent instructions. It is also pure documentation, so it
carries no behavioural risk and can run concurrently with A1 and A2.

## What you may assume

The three existing sets and their specific contradictions
(`01_TARGET_STATE.md` D9):

- `AGENTS.md` (root, ~20 lines) points at `GEMINI.md`, `docs/project-brief.md`
  and `.agent/rules/`.
- `GEMINI.md` (root, ~40 lines) holds the scientific direction and a git
  workflow ending `agent branch -> integration/antigravity -> main`.
- `.agent/rules/` (6 files) and `.agent/workflows/` (5 files).

Known-wrong content that must not survive:

| File | What is wrong |
| --- | --- |
| `.agent/rules/01-project-context.md` | describes `dummy_dataset.py` as one of two live dataloaders and tells agents to build a `DUMMY` flag around it. That file was **deleted** during consolidation. Also says data is on Yale Bouchet. |
| `.agent/rules/03-codebase-standards.md` | mandates a `models/` `training/` `evaluation/` layout that this campaign is replacing |
| `.agent/rules/06-agent-coordination.md` | tells agents to activate conda at `~/miniforge3/envs/aurora_mjo` — wrong manager, wrong path |
| `GEMINI.md` | `integration/antigravity` is superseded by `epic/<slug>` (`01_TARGET_STATE.md` D11) |

Content that **is** genuinely good and must survive, stated concretely:

- **`.agent/rules/02-research-standards.md`** — chronological splits only;
  train-period normalisation only; do not describe ideas as implemented unless
  code and configs exist; when adding a loss, state what failure mode it
  addresses.
- **`.agent/rules/04-experiment-protocol.md`** — every meaningful experiment
  change updates code, config **and** `docs/experiment-registry.md`, recording
  name, date, objective, changed files, data split, variables, losses, rollout
  horizon, LoRA settings, expected outcome, actual result.
- **`.agent/rules/05-remote-safety.md`** — confirm which HPC you are on; show
  the full command and get approval before a long job; never kill a running
  training process; never move or delete checkpoint directories without
  approval; GPU time is valuable.

## Steps

1. Vendor the standard: copy `docs/REPO_BUILD_GUIDE.md` from
   `research-repo-template` into `docs/`. Copy it **verbatim** — do not edit it
   to fit this repo. Where this repo diverges, the divergence is recorded in
   `01_TARGET_STATE.md` §8, which is the right place for it. Add a short header
   note in `AGENTS.md` saying the guide is vendored, is written for quant
   research repos, and that `01_TARGET_STATE.md` §8 lists the deliberate
   divergences.

2. Rewrite `AGENTS.md`. Roughly 60–80 lines. Sections:

   - **Read order**: this file → `docs/REPO_BUILD_GUIDE.md` →
     `docs/PROJECT_STATE.md` → your campaign task file only.
   - **The one command that defines done**: `uv run python scripts/check.py`,
     and that reporting green when it is red is the single most damaging thing
     an agent can do here.
   - **What this project is**, in three sentences, including that it runs on
     **Perlmutter with 4 × A100 80G** — not Yale Bouchet, not H200s.
   - **Layout rules that are not negotiable**: `src/aurora_mjo/` is imported and
     tested, never run; `scripts/` is run, never imported; `run.py` parses flags
     and calls one function; `uv` only, no conda; every runtime import is a
     direct entry in `[project.dependencies]`.
   - **Repo-specific prohibitions.** These must be **concrete and named**, not
     "be careful". At minimum:
     - Never write to `/global/cfs/cdirs/m4946/` — it is another group's
       allocation and this repo reads it only.
     - Never flip `gradient_checkpointing` to `true`. It deterministically
       triggers an illegal memory access on Perlmutter.
     - Never remove the `model.norm_stats.msl` override in
       `configs/unified.yaml`. Removing it restores a −36 σ input and 100%
       non-finite validation.
     - Never replace a probe with a constant in `dataset.py` — the pressure-level
       indices and per-variable time axes are measured on purpose.
     - Never use a random train/val split. Chronological only, and normalisation
       statistics come from the training period only.
     - Never kill a running training process; never move or delete a checkpoint
       directory without approval.
     - Never create a git worktree. One clone, one branch per task.
   - **The experiment-registry rule**, carried over from `.agent/rules/04`.
   - **Git**: branch from the campaign's epic branch, never from `main`; commit
     at internal milestones; messages say why; push and stop.
   - **When stuck**: append to the campaign's `QUESTIONS.md` with an ID, state
     what it blocks, propose a default, then carry on with anything that does
     not depend on the answer.

3. Replace `CLAUDE.md` (new) and `GEMINI.md` (overwrite) with pointers of about
   five lines each: the instructions live in `AGENTS.md`, read that then
   `docs/REPO_BUILD_GUIDE.md`. Add the reason they are pointers and not copies —
   the day two instruction files disagree is the day neither is trusted, and
   that day has already happened in this repo.

   **The scientific direction currently in `GEMINI.md` must not be lost.** Move
   it to `docs/project-brief.md`, appending rather than overwriting, under a
   heading saying where it came from. Check for and remove duplication against
   what is already there.

4. Move `.agent/workflows/*.md` (5 files: `add-mjo-head`, `build-rmm-eval`,
   `onboard-repo`, `physics-loss`, `rollout-training`) to `docs/workflows/`.
   Move them as-is; **do not rewrite them**. Add `docs/workflows/README.md`
   saying these are pre-campaign task sketches, that they have **not** been
   checked against the current codebase, and that `onboard-repo.md` in
   particular is superseded by `AGENTS.md`.

5. Delete `.agent/` entirely, having confirmed everything from the tables above
   is either relocated or deliberately dropped. **List in your result file, file
   by file, what happened to each of the 11 files** — relocated to where, or
   dropped and why.

6. Verify nothing wrong survived:

   ```bash
   grep -rin "dummy_dataset\|miniforge3\|Bouchet\|H200\|integration/antigravity" \
     AGENTS.md CLAUDE.md GEMINI.md docs/ --exclude-dir=campaigns --exclude-dir=archive
   ```

   This should return nothing. Hits inside `docs/campaigns/` and `docs/archive/`
   are historical record and are fine. Note that `README.md`,
   `docs/documentation.md` and `docs/experiment-registry.md` also contain stale
   references — **those are F1's**, not yours; do not touch them, but do list
   what you saw as Observations.

## Definition of Done

- [ ] `docs/REPO_BUILD_GUIDE.md` vendored verbatim
- [ ] `AGENTS.md` rewritten; every prohibition in step 2 present and phrased
      concretely; Perlmutter/4×A100 stated
- [ ] `CLAUDE.md` and `GEMINI.md` are pointers only
- [ ] `GEMINI.md`'s scientific direction preserved in `docs/project-brief.md`,
      de-duplicated, with its provenance noted
- [ ] `docs/workflows/` holds all 5 workflow files plus a README with the
      not-verified warning
- [ ] `.agent/` does not exist; per-file disposition of all 11 files listed in
      the result file
- [ ] The `grep` in step 6 returns nothing — command and empty output pasted
- [ ] `uv run python scripts/check.py` green — summary table pasted (skip only
      if A3 has not merged; say so if so)
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Rewriting `docs/git-policy.md` (F1). It still describes the worktree model
  that `01_TARGET_STATE.md` D11 supersedes; leave it and note it.
- Fixing stale references in `README.md`, `docs/documentation.md` or
  `docs/experiment-registry.md` (F1).
- Writing `docs/PROJECT_STATE.md`, `SPEC.md`, `SETUP.md`, `ARCHITECTURE.md` or
  `CLI.md` (F1). `AGENTS.md` may point at `docs/PROJECT_STATE.md` before it
  exists.
- Editing any `.py`, `.yaml` or `.slurm` file.
