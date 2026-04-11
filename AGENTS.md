# Agent instructions

Read and follow:
1. ./GEMINI.md
2. ./docs/project-brief.md
3. relevant files in ./.agent/rules/

## Working rules
- Prefer small, reversible changes.
- Before implementing, identify the exact files to inspect.
- For ML changes, update code, config, and docs together.
- For experiments, never modify defaults silently; create or edit explicit config files.
- For evaluation work, verify metric definitions against docs/evaluation-spec.md.

## Git policy
Read and follow `docs/git-policy.md`.

Do not push to `main`.
Do not merge to `main`.
Do not assume shared working tree collaboration; each task should use its own branch/worktree.