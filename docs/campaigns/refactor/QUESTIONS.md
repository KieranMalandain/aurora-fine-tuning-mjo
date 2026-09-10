# Questions

Agents **append**. The human answers **in place** — do not move or renumber
existing entries.

Format:

```markdown
## Q-nn — <one-line question>

**Raised by:** <TASK_ID>, <YYYY-MM-DD>
**Blocks:** <what cannot proceed, or "nothing — proceeding on the default">
**Proposed default:** <what the agent will do absent an answer>

**ANSWER (human, YYYY-MM-DD):** <answer>
```

Proposing a default is not optional. A question with no default blocks the
campaign on the human's inbox.

**Q-01 to Q-08 were raised when the campaign was written.** Each has a default
that the task files already assume, so the campaign can start with none of them
answered. If your task depends on an unanswered one, **proceed on the default
and say so in your result file.**

---

## Q-01 — Package name: `aurora_mjo`, given the collision with `microsoft-aurora`?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** nothing — proceeding on the default. But it renames things in C1,
C2, D1 and F1 if the answer changes, so answer it early if you are going to.

The request was for `src/aurora/`. That is not safe: `microsoft-aurora` occupies
the top-level `aurora` import name and is imported nine times across six
first-party files (`00_CONTEXT.md` §4). A first-party package installed as
`aurora` would shadow it, surfacing as `ImportError: cannot import name 'Batch'`
at best and a wrong resolution order at worst.

**Proposed default:** `src/aurora_mjo/`, imported as `aurora_mjo`. It matches
the conda env name (`aurora_mjo`) and the repo name (`aurora-fine-tuning-mjo`),
so nothing else needs renaming. Alternatives considered and rejected: `mjo`
(too generic, and `src/mjo` reads like a subpackage of something), `amjo`
(unsearchable), and vendoring the Microsoft package under a different name (a
fork's worth of maintenance for a cosmetic gain).

**ANSWER (human, YYYY-MM-DD):**

---

## Q-02 — Where do the Gemini agents actually run?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** whether B1, B2 and F2 are agent tasks or human tasks.

Three of the sixteen tasks need the cluster: **B1** (behavioural fingerprint —
needs the conda env and one year of CFS data), **B2** (grep July SLURM logs on
`/pscratch`), and **F2** (acceptance smoke test on a GPU node). The other
thirteen are fully offline and need no data, no GPU and no NERSC credentials.

**Proposed default:** the thirteen offline tasks are dispatched to agents; B1,
B2 and F2 are **run by the human on Perlmutter**, with the agent-facing task
file kept as the script the human follows and the result file written the same
way. The phase gates and dependency graph are identical either way.

If the agents *do* have a shell on Perlmutter with the conda env and CFS read
access, flip these three to agent tasks — nothing else changes. Note that B1 is
the campaign's primary gate, so there is an argument for a human running it
regardless of capability.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-03 — Is `uv` available on Perlmutter, and can it install torch 2.5.1+cu121 there?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** A1's Definition of Done, and therefore all of Phase C. This is the
highest-risk unknown in the campaign.

Unverified: whether `uv` is installed or installable on Perlmutter login nodes,
whether outbound HTTPS to `download.pytorch.org` is permitted from them, and
whether a `uv`-managed CPython 3.10 links correctly against the site MPI/CUDA
stack that `microsoft-aurora` and DDP need. NERSC sites commonly steer users
toward site-provided conda modules and shared read-only Python stacks.

**Proposed default:** A1 does the resolution work **offline first** — write
`pyproject.toml` from `$HOME/aurora-backup/aurora_mjo-explicit-*.txt`, produce a
`uv.lock`, and verify the lock resolves. It does **not** claim GREEN on
Perlmutter installability. A1 raises a follow-up question with whatever it
measured, and the campaign proceeds on the offline lock. The conda env stays on
disk and functional throughout — nothing in this campaign deletes it — so a
failure here degrades to "we have a lockfile and still use conda to run", which
is a strictly better position than today.

If `uv` genuinely cannot work on Perlmutter, the fallback is `uv` for
development and CI plus a generated `requirements.txt` pinned from `uv.lock` for
the cluster. **Do not** decide that unilaterally; raise it.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-04 — Does `train.py` need to keep working after the CLI move?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** C2's scope.

`slurm_scripts/train_auto.slurm` invokes `train.py` under `torchrun`. A chained
SLURM job submitted before the merge and landing after it would fail on a moved
entry point — costing an 11-hour slot plus another queue wait to discover.

**Proposed default:** keep `train.py` as a ~6-line shim that prints a
deprecation notice naming the `run.py` equivalent and `exec`s it with `argv`
preserved. Remove it in a later campaign. Cost: six lines. See
`01_TARGET_STATE.md` D8.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-05 — Which type checker, and is `pyrefly` viable on Python 3.10?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** nothing — A3 decides by measurement.

The template specifies `pyrefly` and states mypy is not used in new repos.
Whether `pyrefly` supports Python 3.10 in this environment is unverified.

**Proposed default:** A3 attempts `pyrefly` first and records what it measured.
If it does not work on 3.10, use `mypy` with the same ratchet scope
(`src/aurora_mjo/` and `tests/` only) and record the substitution in
`docs/PROJECT_STATE.md` as a deliberate divergence from the template. Either
way the gate step is named `types` in `check.py` so the tool can be swapped
without changing the contract.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-06 — Is CI actually available and wanted on this repo?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** A3's Definition of Done.

The repo is `git@github.com:KieranMalandain/aurora-fine-tuning-mjo.git`. Unknown:
whether Actions is enabled, whether there is a minutes budget, and whether the
repo is private (private repos have a tighter free allowance).

**Proposed default:** A3 commits `.github/workflows/ci.yml` targeting
`ubuntu-latest` (**not** the template's `windows-latest` — see
`01_TARGET_STATE.md` §8) and running only `scripts/check.py` with the
`needs_data`/`needs_gpu`/`live` markers excluded. If Actions is disabled the
file is inert and harmless; the same command run locally is the real gate. A3
does not attempt to configure branch protection or required status checks —
that is a human repo-settings action, and F1 documents it as a follow-up.

Note the template's `standards-drift` and `runtime-deps` CI jobs are **not**
adopted: there is no org standards URL for this repo, and the runtime-deps job
would need to install torch on every run.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-07 — Where does `slt_data.nc` live permanently?

**Raised by:** campaign authoring, 2026-09-10 (inherited from
`cleanup-2026-09.md` §10.5)
**Blocks:** nothing in this campaign. E1 makes its absence loud; it does not
relocate it.

`slt_data.nc` is a hard dependency of `dataset.py`, is **not** in the LANL
archive, currently lives on purgeable `/pscratch`, and its only provenance is
the 21-line `scripts/download_slt.py`. Committed evidence shows it working
(`slt` mean 0.6708, std 1.1682) — so it exists and is correct today.

**Proposed default:** out of scope. E1 turns a missing/unreadable `slt_data.nc`
into a hard failure naming the file and the `HDF5_USE_FILE_LOCKING` fix. F1
records the relocation as an open item in `docs/PROJECT_STATE.md`. The human
decides whether "durable" means `$HOME`, a CFS project directory this repo owns,
or committed provenance plus a regeneration script.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-08 — Should `scripts/evaluate_mjo.py` and `compute_rmm.py` be split into the package now?

**Raised by:** campaign authoring, 2026-09-10
**Blocks:** C3's scope. Materially changes C3's budget.

Together they are **1,637 lines** — nearly a quarter of the repo — and they sit
in `scripts/`, which by the layout rule is run-never-imported. Their logic is
exactly what the test suite most needs to cover (RMM computation is where a
leakage bug would hide, per `03_DOMAIN_PRIORS.md` §6). But splitting 1,637 lines
of numerically sensitive code is the single largest and riskiest change
available in this campaign, and it is not what the campaign is for.

**Proposed default:** C3 does the **mechanical** part only — create
`src/aurora_mjo/rmm/{compute,evaluate}.py`, move the pure functions with no
signature changes, and leave `scripts/*.py` as thin argparse wrappers that
import them. No refactoring of the numerics, no restructuring, no renaming. If
C3 finds the two files are not cleanly separable into pure-vs-IO within budget,
it reports **AMBER**, leaves them in `scripts/`, and the split becomes its own
task in a follow-on campaign. Explicitly acceptable outcome — say so plainly
rather than half-doing it.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-09 — Task branch naming convention given Git ref conflict with `epic/refactor`

**Raised by:** A1, 2026-09-10
**Blocks:** nothing — proceeding on default.
**Proposed default:** Use `epic/refactor-<TASK_ID>-<short-slug>` rather than `epic/refactor/<TASK_ID>-<short-slug>`. In standard Git storage, because `epic/refactor` exists as a branch ref file (`.git/refs/heads/epic/refactor`), Git rejects creating any sub-ref `epic/refactor/*` due to directory/file collision (`fatal: cannot lock ref 'refs/heads/epic/refactor/...': 'refs/heads/epic/refactor' exists`). Replacing the slash with a hyphen keeps the branch grouped under `epic/` without conflicting with the base branch name.

**ANSWER (human, YYYY-MM-DD):**

---

## Q-10 — Perlmutter uv deployment: filesystem locking and environment paths

**Raised by:** A1, 2026-09-10
**Blocks:** nothing — proceeding on default.
**Proposed default:** Standardize `UV_CACHE_DIR=/pscratch/sd/k/kam352/.cache/uv`, `UV_PYTHON_INSTALL_DIR=/pscratch/sd/k/kam352/.local/share/uv/python`, and `UV_DATA_DIR=/pscratch/sd/k/kam352/.local/share/uv` in shell startup scripts (`~/.bashrc` and `~/.local/bin/env`). Measurement during A1 proved that `uv` can install and run cleanly on Perlmutter (Python 3.10.21, torch 2.5.1+cu121 with 4 GPUs detected), but default paths in `/global/homes` trigger OS error 524 (flock unsupported on GPFS/Lustre). Moving cache and python directories to `/pscratch` completely resolves the locking issue.

**ANSWER (human, YYYY-MM-DD):**

---