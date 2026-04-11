---
trigger: always_on
---

# Agent coordination

When multiple agents are used:
- Assign each agent a distinct file/domain ownership area.
- Do not have two agents edit the same file concurrently.
- One agent should act as coordinator and merge reviewer.
- Research agents may read broadly but should not rewrite training code directly.
- Implementation agents must report:
  - files changed
  - commands to run
  - unresolved assumptions