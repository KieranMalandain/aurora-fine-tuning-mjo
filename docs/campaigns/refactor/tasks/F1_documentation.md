# F1 — `SPEC`, `SETUP`, `ARCHITECTURE`, `CLI`, `PROJECT_STATE`; README; stale refs

| | |
| --- | --- |
| **Phase** | F |
| **Depends on** | E1, E2, E3 (all `PROCEED: YES`) |
| **Base branch** | `epic/refactor` |
| **Budget** | 3.5 h |
| **Touches** | `README.md`, `docs/{SPEC,SETUP,ARCHITECTURE,CLI,PROJECT_STATE}.md`, `docs/git-policy.md`, `docs/documentation.md`, `docs/experiment-registry.md`, `docs/known-gaps.md`, `docs/archive/` |
| **Must not touch** | any `.py`, `.yaml`, `.slurm` or `.sh` file. `AURORA_MJO_GAMEPLAN.md` — **archive it, do not edit it.** Anything under `docs/campaigns/refactor/` except as noted in step 8. |

## Objective

At the end, the documentation describes the repository that now exists: the right
machine, the right package manager, the right entry point, no references to
deleted files, and one living state document that says what the next action is.

## Why this is a separate task

Documentation written before the code settles is wrong by the time it lands. This
runs after E1–E3 so it describes reality. It is also the task that decides what
gets *archived* rather than deleted, and those decisions want to be made in one
place by one agent who can see all of them.

## What you may assume

- The layout in `01_TARGET_STATE.md` D2 is now real.
- A4 rewrote `AGENTS.md`, made `CLAUDE.md`/`GEMINI.md` pointers, vendored
  `docs/REPO_BUILD_GUIDE.md`, moved workflows to `docs/workflows/`, and folded
  `.agent/` in. **Do not redo any of it**; read A4's result file for the list of
  stale references it deliberately left for you.
- C2 and E2 built `docs/CLI.md`: C2 the flag table, E2 the validity matrix. You
  complete it.
- Stale references to deleted configs survive in **markdown only** by now — C3
  fixed the `.py` files. The remaining files (`00_CONTEXT.md` §2.4 item 3):
  `README.md`, `docs/documentation.md`, `docs/experiment-registry.md`, and
  `AURORA_MJO_GAMEPLAN.md`.
- `README.md` describes **Yale Bouchet and H200s**. The work happened on
  **Perlmutter, 1 node × 4 × A100 80G**.
- `docs/verify_output.txt` is the **stale** verify run showing `slt` zeroed;
  `docs/verify_output_slt.txt` is current. (`03_DOMAIN_PRIORS.md` §3)
- Real measured numbers for `SPEC.md` are in `03_DOMAIN_PRIORS.md` §§1–3 and 7.
  **Cite them rather than re-deriving.**

## Steps

1. **Rewrite `README.md`.** Write it for someone who has just cloned the repo.
   Sections: what this is in three sentences; the science goal (MJO sub-seasonal
   prediction, RMM r > 0.5 at 30 days, prognostic state stepping not statistical
   regression); quickstart (`uv sync --all-groups`, `uv run python
   scripts/check.py`, `uv run python run.py train --mode baseline
   --smoke-test`); where the data lives; **Perlmutter, 4×A100 80G**; and
   pointers to the doc set. Keep it short — pointers, not content.

   Delete every Yale Bouchet and H200 reference. Delete every reference to
   deleted per-phase configs.

2. **`docs/SETUP.md`** — clone to green gate. `uv` installation, `uv sync
   --all-groups`, `uv run pre-commit install`, `uv run python scripts/check.py`.
   Then a **separate Perlmutter section**: how to get `uv` there, where the data
   root and `slt_path` are, that `HDF5_USE_FILE_LOCKING` is now handled in
   Python (E1) but is also in `slurm_scripts/env.sh`, and how to run the smoke
   test on a GPU node.

   If A1 or E3 reported `AMBER` on `uv` on Perlmutter, **document the actual
   working procedure**, whatever it is, and mark it as provisional pointing at
   Q-03. An honest provisional instruction beats an aspirational one.

3. **`docs/ARCHITECTURE.md`** — supersede the existing one. The real spine:
   `dataset.py` → `model.py` (Aurora backbone + optional MJO head + optional
   LoRA) → `loss.py` → `trainer.py` → `checkpoint.py`, driven by `run.py`
   through a validated `Config` (E2). State the `src/` imported-never-run vs
   `scripts/` run-never-imported rule, and the `aurora` / `aurora_mjo` name
   collision with a pointer to `00_CONTEXT.md` §4 — that is the single most
   important thing for a new contributor not to trip over.

4. **`docs/SPEC.md`** — the durable design document, the one that outlives this
   campaign. Fill from measured values:

   - purpose and audience
   - domain background: define MJO, RMM, OLR, TCWV, and every abbreviation the
     code uses
   - **the variable table** from `02_UPSTREAM_CONTRACT.md` §2, including that
     native names differ from Aurora names and that `msl` reads `ps`
   - grid, levels, timestep and shape table from `03_DOMAIN_PRIORS.md` §2
   - the sample-count arithmetic from §1, including that 54,060 is the
     signature of the leakage bug
   - splits: train 1980–2015, val 2016–2019, test 2020–2023, **chronological
     only**
   - normalisation, including that `msl` stats are **PLACEHOLDER** and what
     replacing them requires (§4, area-weighting caveat included)
   - config and CLI: point at `docs/CLI.md`
   - **non-goals**, so a future agent does not helpfully implement them: copy
     from `00_CONTEXT.md` §5 — Findings 3/5/6, small-vs-full Aurora, computing
     real norm stats
   - open items table with IDs, what each blocks, and a proposed default

   **Promote the durable findings out of the campaign.** `00_CONTEXT.md` §3's
   seven lessons and `02_UPSTREAM_CONTRACT.md`'s five traps are durably true and
   belong here, not buried in a campaign directory that gets archived. Summarise
   each with a pointer to its test in `tests/` — that pairing is what makes the
   document trustworthy.

5. **Complete `docs/CLI.md`.** Every `run.py` command and flag, the validity
   matrix from E2, and the list from C3 of scripts that have no `--help` so a
   reader knows what is undocumentable.

6. **Write `docs/PROJECT_STATE.md`** — the handover document, and the most
   important thing you write. Format from the template: **Last updated**,
   **Next action** (one sentence, not a list), **Status at a glance** table,
   **Open** checklist, **Unverified assumptions**, **Superseded**.

   Populate from the campaign's result files, not from guesswork:

   - the type-checker exclusion list length (A3) and that it is a ratchet
   - whether `uv` works on Perlmutter (A1, E3) — and if not, what is used
   - B2's verdict on the July runs, with a pointer to
     `docs/findings/2026-09-zeroed-statics.md`
   - test counts by category (D2, D3)
   - the `msl` placeholder as an open item, naming
     `scripts/calc_norm_stats.py`
   - `slt_data.nc` relocation as an open item (Q-07)
   - whether C3 split the RMM code or reported `AMBER` (Q-08)
   - `auto_scale_memory`'s post-validation mutation (E2's Observation)
   - branch protection / required status checks as a **human** action (A3)
   - Findings 3, 5 and 6 as the next campaign's subject

   **Next action** should be the single most useful thing to do next. Based on
   everything above, that is most likely computing the real `ps` normalisation
   statistics — it unblocks every scientific result and is the last thing
   standing between the repo and a trustworthy baseline run. Decide from the
   result files, and justify your choice in one sentence.

7. **Rewrite `docs/git-policy.md`.** It currently mandates
   `agent branch -> integration/antigravity -> main` and a git worktree per
   agent. Replace with the epic model from `01_TARGET_STATE.md` D11:
   `epic/<slug>` base branch, one branch per task, no worktrees, human merges to
   `main` once.

   **Say why the worktree model was dropped**, plainly: it produced the
   four-worktree, three-divergent-branch, one-stash state that took nine tasks
   to unwind. A policy change with its reasoning attached survives; one without
   gets reverted by the next person who liked the old way.

   **Preserve** the good parts: `main` is protected; snapshot tags before major
   changes; agents never merge to `main`; commit-message quality examples; the
   recovery-point table in `cleanup-2026-09.md` §8 including
   `origin/archive/pre-antigravity-baseline` marked **do not delete**.

8. **Archive, do not delete.** Create `docs/archive/` and move:

   - `AURORA_MJO_GAMEPLAN.md` → `docs/archive/AURORA_MJO_GAMEPLAN.md`. It is the
     most substantive analysis in the repository and it was written under
     deadline pressure with stale file references. **Do not edit it** — add a
     header note in `docs/archive/README.md` saying what it is, that its config
     references predate consolidation, and that its Findings 1–6 are summarised
     in `docs/SPEC.md`.
   - `docs/verify_output.txt` → `docs/archive/verify_output_pre_slt.txt`, with a
     header note saying it shows `slt` zeroed, that this was fixed, and that
     `docs/verify_output_slt.txt` is current.
   - Any other doc your review finds superseded — `docs/documentation.md`,
     `docs/known-gaps.md` and `docs/development-workflow.md` are candidates.
     **Judge each individually and justify each decision in your result file.**
     `docs/known-gaps.md` in particular still says the code implements things
     that changed; either update it or archive it, but do not leave it
     ambiguous.

   Keep `docs/verify_output_slt.txt` in place — it is cited by
   `03_DOMAIN_PRIORS.md` §§2–3 and it is current evidence.

   Keep `docs/nersc-dataset-information.md`, `docs/nersc_data.md`,
   `docs/evaluation-spec.md`, `docs/compute-environments.md`,
   `docs/data-inventory.md`, `docs/project-brief.md` and
   `docs/experiment-registry.md` — but **read each and fix stale references**.
   Note that `docs/nersc-dataset-information.md` and `docs/nersc_data.md` look
   like duplicates; check, and if so consolidate and archive one.

9. **Reconcile every remaining stale reference.** Then prove it:

   ```bash
   grep -rn "phase1_baseline\|phase2_physics\|phase3_longrun\|dummy_dataset\|\
   slurm_scripts/train\.slurm\|Bouchet\|H200\|conda\|environment\.yml\|\
   integration/antigravity\|from src\." \
     --include=*.md . | grep -v "docs/archive/" | grep -v "docs/campaigns/"
   ```

   Must return nothing, except deliberate mentions in `SETUP.md` or
   `git-policy.md` explaining the migration. **List every remaining hit with its
   justification.**

   `docs/campaigns/` and `docs/archive/` are excluded because they are
   historical record.

10. Update `docs/experiment-registry.md`. `cleanup-2026-09.md` §10.4 asks
    whether it is still accurate — it was not examined during consolidation.
    **Check it**, fix stale references, and record your verdict on its accuracy.
    If it documents experiments whose checkpoints were deleted as non-finite,
    say so per entry rather than leaving results that look meaningful.

## Definition of Done

- [ ] `README.md` rewritten; Perlmutter/4×A100 correct; no Bouchet, H200 or
      deleted-config references
- [ ] `docs/SETUP.md` covers clone-to-green **and** Perlmutter, marked
      provisional where A1/E3 were `AMBER`
- [ ] `docs/ARCHITECTURE.md` describes the real spine, the src/scripts rule, and
      the `aurora`/`aurora_mjo` collision
- [ ] `docs/SPEC.md` complete, with measured values cited from
      `03_DOMAIN_PRIORS.md`, the variable table, split definitions, the
      PLACEHOLDER warning, non-goals, and open items
- [ ] The seven lessons and five upstream traps promoted into `SPEC.md`, each
      paired with a pointer to its test
- [ ] `docs/CLI.md` complete with the validity matrix and the no-`--help` list
- [ ] `docs/PROJECT_STATE.md` written, populated from result files, with a
      one-sentence **Next action** and a justification for it
- [ ] `docs/git-policy.md` rewritten to the epic model, with the worktree
      rationale stated and recovery points preserved
- [ ] `docs/archive/` holds the gameplan and `verify_output_pre_slt.txt`, with a
      `README.md` explaining each entry
- [ ] `AURORA_MJO_GAMEPLAN.md` archived **unedited** — confirm byte-identical
- [ ] `docs/verify_output_slt.txt` still in place
- [ ] Every archive-or-keep decision justified individually
- [ ] The nersc-dataset duplicate question resolved
- [ ] The step-9 grep returns nothing, or every hit is listed with a
      justification — output pasted
- [ ] `docs/experiment-registry.md` accuracy verdict recorded
- [ ] `uv run python scripts/check.py` green — summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- Editing any `.py`, `.yaml`, `.slurm` or `.sh` file. If documenting something
  reveals a code bug, that is an Observation.
- Editing `AURORA_MJO_GAMEPLAN.md` or anything already archived.
- Rewriting `docs/REPO_BUILD_GUIDE.md` — it is vendored verbatim (A4); divergences
  live in `01_TARGET_STATE.md` §8.
- Moving this campaign directory to `docs/archive/campaigns/`. That happens
  when the human merges `epic/refactor` to `main`, after F2.
- Configuring branch protection. Record it as a human action.
