# Git & GitHub Governance Policy

**Status:** Authoritative Repository Policy  
**Branching Model:** Epic Branch Discipline (supersedes the legacy worktree model)  

---

## 1. Core Principles

1. **`main` is protected and authoritative.** It reflects verified, production-ready code.
2. **Sequential execution over parallel worktrees.** Work runs sequentially on task branches cut from an epic branch.
3. **Automated verification is mandatory.** Every commit and branch must pass `scripts/check.py` before integration.
4. **Agents propose; humans integrate.** Agents never merge to `main` or to an epic branch. All merges are executed by human maintainers.
5. **Historical recovery points are immutable.** Archive tags, backup bundles, and historical baseline refs must never be deleted or overwritten.

---

## 2. The Epic Branching Model

This repository operates on an **Epic Branch** strategy:

```text
main (protected)
  └── epic/refactor (or epic/<slug>)
        ├── epic/refactor-A1-uv-pyproject
        ├── epic/refactor-A2-hygiene-untrack
        └── ...
```

### Branch Hierarchy
- **`main`**: Long-term stable branch. Protected in GitHub; requires PR review and green CI.
- **`epic/<slug>`** (e.g. `epic/refactor`): The active integration branch for a campaign. Created from `main` at campaign initiation; merged back to `main` by the human maintainer only after the full campaign acceptance gate is green.
- **`epic/<slug>-<TASK_ID>-<short-slug>`**: Short-lived, single-task branches cut directly from `epic/<slug>`. Each task in a campaign receives its own branch, commits at internal milestones, and pushes for human review.

### Allowed Pull Request Flow
1. Task branch pushed: `epic/refactor-<TASK_ID>-<short-slug>`.
2. Human reviews task result file (`results/<TASK_ID>_result.md`) and verifies `scripts/check.py`.
3. Human merges task branch into `epic/<slug>`.
4. After all campaign tasks pass, human merges `epic/<slug>` into `main`.

---

## 3. Why the Legacy Worktree Model Was Dropped

> [!WARNING]
> **Rationale for Dropping Git Worktrees:**
> The earlier repository policy mandated a "git worktree per agent" model. In practice, this produced four uncoordinated worktrees (`human-integration`, `campaign-fixes`, `agent-simul-training-one`, `consolidation`), three divergent remote branches, and detached stashes. Changes were hand-carried across trees, tracking branches drifted, and nobody could determine which codebase was authoritative. Unwinding this required a nine-task emergency consolidation.
> 
> **Rule:** Git worktrees are strictly prohibited for agent operations in this repository. One clone, one active branch per task, executed sequentially.

---

## 4. Commit Message Quality & Hygiene

Commits must be:
- Focused on a single logical milestone.
- Accompanied by a descriptive message explaining **why** the change was made, not just what was changed.
- Verified: `uv run pre-commit run --all-files` must pass without `--no-verify`.
- Synchronized: If `pyproject.toml` is modified, `uv.lock` must be committed in the **same commit**.

### Good Commit Message Examples
```text
C1: move src/ to src/aurora_mjo/ to resolve third-party import collision

microsoft-aurora owns the top-level `aurora` namespace in site-packages.
A first-party package called `aurora` shadows it, causing immediate ImportError
on Batch. Renamed source package to aurora_mjo and repointed 27 imports.
```

```text
E1: configure HDF5_USE_FILE_LOCKING=FALSE at process startup

CFS parallel mounts require advisory locking disabled to prevent Errno -101
NetCDF errors. Configured automatically in src/aurora_mjo/env.py and verified
before C-libraries initialize.
```

### Bad Commit Message Examples
- `fix stuff`
- `wip`
- `update docs`
- `many changes`

---

## 5. Protected Historical Recovery Points

The following historical recovery points are preserved in Git history and backup storage. **Do not delete, move, rebase, or overwrite any ref in this table:**

| Reference Name | Type | Historical Context | Protection Level |
| :--- | :--- | :--- | :--- |
| `origin/archive/pre-antigravity-baseline` | Remote branch (`800455a`) | Authoritative baseline before agent onboarding | **DO NOT DELETE** |
| `backup/pre-cleanup/main` | Git Tag / Ref (`07a9ce0`) | State of `main` before September 2026 consolidation | Immutable |
| `backup/pre-cleanup/antigravity` | Git Tag / Ref (`b0cff2d`) | Pre-cleanup integration branch | Immutable |
| `backup/pre-cleanup/campaign-fixes` | Git Tag / Ref (`b9ffe63`) | Pre-cleanup bugfix branch | Immutable |
| `backup/pre-cleanup/simul-training` | Git Tag / Ref (`6862fdf`) | Pre-cleanup parallel training worktree | Immutable |
| `v0.1.0-pre-agents` | Annotated Tag | Clean historical tag | Immutable |
| `v0.1.1-agent-scaffold` | Annotated Tag | Clean historical tag | Immutable |
| `v0.1.2-pre-campaign-fixes` | Annotated Tag | Clean historical tag | Immutable |

### Offline Backups
Incremental bundles and conda environment exports are stored in `$HOME/aurora-backup/`:
- `aurora_mjo-explicit-*.txt`: Ground-truth dependency list for initial `uv` lockfile creation.
- `cleanup-preserve-*.tar.gz`: Preserved patch sets from the September 2026 consolidation.

---

## 6. Main Branch Safety & CI Requirements

Repository administrators must configure GitHub repository rulesets for `main`:
- Require pull request before merging (no direct pushes to `main`).
- Require status check `check` from `.github/workflows/ci.yml` to pass.
- Require linear history (rebase or squash merges preferred).
- Disallow force pushes and branch deletions on `main`.