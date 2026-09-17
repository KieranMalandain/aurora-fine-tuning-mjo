# L1 — Synthesise the campaign; promote durable truth; propose the paper campaign

| | |
| --- | --- |
| **Phase** | L |
| **Depends on** | K4 (`PROCEED: YES` or `BLOCKED` with the physics term retired on evidence), and a result file for every prior task |
| **Base branch** | `epic/science-baseline` |
| **Budget** | 4 h |
| **Compute** | **None.** |
| **Touches** | `docs/campaigns/science-baseline/handoff-2026-1x-science.md` (new), `docs/SPEC.md`, `docs/evaluation-spec.md`, `docs/findings/`, `docs/PROJECT_STATE.md` (one pointer line) |
| **Must not touch** | Any `.py`, `.yaml`, `.slurm` or `.sh` file. Any `results/*.md` — they are the permanent record and are read-only to you. `00_`–`04_`, `QUESTIONS.md`, or any task file. |

## Objective

One self-contained document that a person picking this project up cold can read
in twenty minutes and come away knowing what the science now is, what was
measured, what is still wrong, and what to do next. It ends with an argued
recommendation for the paper campaign.

## Why this is a separate task

Twenty result files is a lot of evidence and nobody reads twenty files. The
thing that gets lost is the **Observations** section — every task was told to
record what it noticed outside its scope and move on, which scatters real
discoveries across seventeen documents that will never be opened again.

The other half is that nobody has yet looked at the campaign as a whole and said
what it implies about the science. Every task was deliberately narrow. This is
the one allowed to take a position.

`../refactor/tasks/F3_synthesis_next_steps` is the model for this task and its
output, `../refactor/handoff-2026-09-refactor.md`, is the register to match:
direct, numerate, willing to say "unconfirmed", honest about which parts are
sketch.

**One difference from F3, and it matters.** F3 synthesised an engineering
campaign and its recommendations were about what to build. This one synthesises a
scientific campaign, and its central deliverable is a **defensible claim** — what
this project now knows, at what confidence, with what caveats. Write it as if a
reviewer will read it, because one will.

## What you may assume

- Twenty result files under `results/`: G1–G5, H1–H4, J1–J7, K1–K4.
- `docs/results/skill-curves-2026-1x.md` holds the five-way comparison.
- `docs/findings/2026-1x-rmm-validation.md` (J2) and
  `docs/findings/2026-1x-zeroshot-control.md` (J4) exist and are the two documents
  that must outlive this campaign directory.
- Q-12 to Q-19 in `QUESTIONS.md`, some answered, some not.
- The campaign's own plan **will have been wrong in places**. That is expected —
  `../refactor/00_CONTEXT.md` Lesson 7. Recording where is a deliverable, not a
  criticism.

## Steps

1. Read all twenty result files, `QUESTIONS.md`, `docs/PROJECT_STATE.md`, both
   findings documents, and `00_`–`04_` again — you need to know what was
   *promised* to judge what was delivered. **Take notes into the document as you
   go.**
2. Build the Observations harvest first, before any prose. A flat list of every
   Observation and every AMBER caveat with its source task; then merge duplicates
   and drop trivia. Say how many were found, how many survived, how many were
   dropped.
3. Build the numbers tables next. At minimum: the R1 gradient-share table before
   and after H1; parameter counts by mode at `full`; memory and step time at 1°
   vs 0.25°; sample counts (which must not have moved); the J2 BoM correlations;
   the full five-way skill table with all four baselines; GPU-hours consumed per
   stage against budget.
4. Write the science sections. Required, and in this order:
   - **What is now true.** The four defects and what replaced them. Every claim
     cited to a result file.
   - **The result.** T1, T2, T3, T4 — each with a verdict and the number behind
     it. If T1 failed, say so in the first sentence of the section.
   - **What was ruled out.** A hypothesis eliminated is worth as much as one
     confirmed. Include the H3 Step-1 diagnostic verdict and the K4 ablation.
   - **What the campaign got wrong.** Specific things to check rather than
     assume: did the `03_DOMAIN_PRIORS.md` §6 memory estimates survive G2? Did
     the §5 after-column gradient shares match what H1 produced? Did the §9 skill
     priors bracket the J4 control? Was the G → {H, J} → K ordering right, or did
     something serialise that should not have? Did any gate fire?
   - **What is still broken**, separated into inherited and discovered.
5. **Assess the defensibility of the headline claim.** Walk the argument a
   reviewer would: is the control fair; is the evaluation leakage-free; are the
   weights untuned; is the comparison like-for-like; is the model what the paper
   says it is. Name every place the claim is weaker than it looks. This section
   is the one that makes the paper campaign cheap instead of expensive.
6. Promote durable truth **out** of the campaign directory: the loss
   specification and the parameter table into `docs/SPEC.md`; the RMM
   specification into `docs/evaluation-spec.md` (J1 started this — verify it is
   complete); anything else durable into `docs/findings/`. A campaign directory
   gets archived; these do not.
7. Write the recommendations. A **prioritised sequence**, not a backlog, each with
   what it is, why it is at that position, roughly what it costs, the evidence
   for it, your confidence, and what would change your mind. Weigh at least:
   the paper; ablations (LoRA rank, λ_p, `w_v`, spectral term); enabling the MJO
   head; ensembling and Aurora v1.5; closing the train/evaluate horizon gap; the
   **ENSO-neutral-training ablation** (Q-21); **damped-persistence SST**; BSISO
   indices; and whether the 30-day target is reachable at all on the evidence now
   available.
   **Name one thing as mattering most and argue for it.**
8. Propose the next campaign: a name, a base branch, and three to five phases
   with a sentence each. Enough that someone can create the directory from
   `_TEMPLATE` and start filling it in.
9. Verify every citation. Spot-check at least ten and say which ten and whether
   they checked out.
10. Add **one** pointer line to `docs/PROJECT_STATE.md`. Change nothing else in
    that file.
11. Re-read the document end to end as if you had never seen this project.
    Anything you cannot follow without opening another file is a gap — fix it or
    say in the document that it is a gap.

## Definition of Done

- [ ] All twenty result files read, plus `QUESTIONS.md`, `PROJECT_STATE.md`
      and both findings documents — confirmed by listing them
- [ ] `handoff-2026-1x-science.md` written, 300–500 lines, opening with a status
      block that says plainly it is a synthesis and not the source of truth,
      carrying the three-row truth table
- [ ] Every non-obvious claim carries **MEASURED / INFERRED / RECOMMENDED**
- [ ] Observations harvest covers **all twenty** result files; raw count,
      survivor count and dropped count stated
- [ ] Every AMBER caveat in the campaign consolidated
- [ ] Numbers tables complete, one source cited per row
- [ ] **T1–T6 each given a verdict with the number behind it**, T5 in the
      document's first section if it failed
- [ ] "What the campaign got wrong" answers all five check-rather-than-assume
      questions in Step 4
- [ ] **Defensibility section written**, naming every weak point in the headline
      claim
- [ ] Durable truth promoted into `docs/SPEC.md`, `docs/evaluation-spec.md` and
      `docs/findings/`; what moved where, listed
- [ ] Recommendations are a prioritised sequence with cost, evidence, confidence
      and "what would change my mind" per item
- [ ] **One** thing named as mattering most, and argued
- [ ] Next campaign proposed: name, base branch, 3–5 phases
- [ ] Every `Q-nn` from Q-12 to Q-22 covered with its *current* status
- [ ] Ten citations spot-checked; which ten and the outcome recorded
- [ ] One pointer line added to `docs/PROJECT_STATE.md`;
      `git diff docs/PROJECT_STATE.md` pasted showing nothing else changed
- [ ] `uv run python scripts/check.py` is green; summary table pasted
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Any code, config, test or script change.** If you find a bug while reading,
  it goes in "What is still broken".
- **Editing any `results/*.md`.** They are the permanent record of what each agent
  measured at the time, including where they turned out to be wrong. If a result
  file contains a claim that later measurement contradicted, say so in "What the
  campaign got wrong" with both sources. Do not correct it at source — that
  destroys the trail.
- **Writing the paper.** Recommend it; do not start it.
- Answering any `Q-nn` yourself. Restate, assess, recommend — the human answers.
- **Starting the next campaign.** Do not create a directory under
  `docs/campaigns/`, do not write task files, do not create a branch. Step 8
  proposes; a human decides.
- Moving this campaign directory to `docs/archive/campaigns/`. That happens when
  the human merges `epic/science-baseline` to `main`.
