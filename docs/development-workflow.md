# Development Workflow

## Principle

Development happens against the real codebase, not against memory.

Agents must inspect the current repository before proposing changes.

## Recommended workflow

1. Read project context docs.
2. Inspect current code.
3. Summarize current implementation and identify gaps vs roadmap.
4. Propose a small patch.
5. Run smoke tests only.
6. Update docs.
7. Only then propose real training/evaluation runs.

## For outdated code

If existing code is older than the current roadmap:
- do not delete it immediately
- inspect and summarize what it currently does
- identify what remains useful
- mark outdated assumptions explicitly
- refactor incrementally rather than rewriting blindly

## Deliverables for meaningful tasks

- files changed
- summary of changes
- exact commands to run
- known risks / assumptions

See also: `docs/git-policy.md` for branching, worktrees, PR flow, and rollback policy.