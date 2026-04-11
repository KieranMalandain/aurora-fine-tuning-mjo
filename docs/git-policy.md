# Git and GitHub Policy

## Purpose

This repository uses a controlled Git workflow designed for safe parallel development with human and AI agents.

The goals are:
- preserve stable historical states
- support parallel feature development
- prevent agents from interfering with one another
- keep `main` stable
- allow fast rollback and recovery

## Core principles

1. `main` is protected and stable.
2. Agents do not work directly on `main`.
3. Parallel work happens on isolated branches and isolated Git worktrees.
4. Merges flow through pull requests.
5. Human review is required before code reaches `main`.
6. Historical restore points must be preserved with tags and archive branches.
7. Environment-specific changes must not be mixed casually with model-science changes.

## Branch model

### Protected branches
- `main`: stable branch, human-approved only
- `integration/antigravity`: integration branch for combining agent work before merging to `main`

### Archival branch
- `archive/pre-antigravity-baseline`: preserved pre-agent baseline

### Agent branches
Agent branches should be task-specific and short-lived.

Examples:
- `agent/docs-context`
- `agent/env-portability`
- `agent/rmm-eval`
- `agent/mjo-head`
- `agent/rollout-training`
- `agent/physics-loss`

## Snapshot policy

Before major workflow changes or agent-driven refactors:
- create an annotated tag
- push the tag to GitHub
- create and push an archive branch if the state may need to be revisited or patched

Example tags:
- `v0.1.0-pre-agents`
- `v0.1.1-agent-scaffold`

## Worktree policy

Parallel work must use Git worktrees.

Rules:
- each active agent gets its own worktree
- no two agents should work in the same worktree
- no two agents should edit the same file concurrently
- the human integrator should use a separate integration worktree

Worktrees are branch-isolated working copies of the same repository.
They are not separate projects and should not change the logical repository structure.

## Commit policy

Commits must be:
- small
- coherent
- task-specific
- readable

Good examples:
- `Document current architecture and known gaps`
- `Add environment-specific path configuration`
- `Implement RMM evaluation scaffold`
- `Add optional MJO head interface`

Bad examples:
- `update project`
- `fix stuff`
- `many changes`

## Pull request policy

### Allowed flow
- agent branch -> `integration/antigravity`
- `integration/antigravity` -> `main`

### Rules
- agents must not merge their own PRs to `main`
- human review is required before merge to `main`
- PR descriptions should summarize:
  - purpose
  - changed files
  - commands run
  - risks / assumptions

## Main branch safety policy

`main` should be protected with:
- required pull requests
- required status checks
- no force pushes
- no branch deletion

## Validation policy

Before merging to `main`, changes should pass lightweight validation appropriate to the task.

Examples:
- Python import / syntax smoke tests
- configuration validation
- tiny model construction tests
- evaluation script smoke tests

Heavy training jobs are not required in GitHub Actions.

## Environment policy

This project currently spans multiple compute environments, including Yale Bouchet HPC and LANL/NERSC.

Rules:
- do not hardcode environment-specific paths unless explicitly intended
- path roots must be configurable
- Yale subset assumptions must not be mistaken for full-dataset assumptions
- portability changes should be isolated in dedicated branches when possible

## Agent policy

Agents may:
- inspect repository state
- modify files within their task scope
- commit to their own branch
- push their own branch if authorized

Agents may not:
- push directly to `main`
- merge to `main`
- force-push shared branches without approval
- delete branches or worktrees without approval
- rewrite large parts of the repository without a reviewed plan

## Human integrator policy

The human integrator is responsible for:
- creating snapshot tags and archive branches
- assigning agent task scopes
- reviewing PRs
- resolving cross-branch conflicts
- deciding merge order
- approving final merge to `main`

## Rollback policy

Rollback options include:
- checkout of annotated tags
- comparison against archive branches
- cherry-picking prior commits
- restoration from git bundle backups

## Recommended merge order

Suggested order for this repository:
1. docs/context
2. environment portability
3. evaluation pipeline
4. MJO model head
5. rollout training
6. physics-informed extensions

## Repository structure policy

The repository should remain organized by code function, not by agent identity.

Preferred structure:
- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `slurm/`

Do not create permanent top-level directories for individual agents.
Agent isolation is handled by branches and worktrees, not by repository layout.

Worktrees are local development conveniences and are environment-specific. They should not be treated as durable cross-cluster artifacts. The portable source of truth is Git history on remote branches, tags, and pull requests. If development shifts from Bouchet to NERSC, recreate worktrees there from the pushed branches rather than trying to move local worktree directories directly.