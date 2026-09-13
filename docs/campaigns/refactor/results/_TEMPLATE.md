STATUS: GREEN | AMBER | RED | BLOCKED
PROCEED: YES | YES-WITH-CAVEATS | NO
BLOCKED-ON: NONE
SUMMARY: <one sentence, under 120 characters, no line break>

<!--
The four lines above are MACHINE-READ by the next agent. Keep them as the first
four lines of the file, one per line, in this order, with these exact key names.
Everything below is for humans.
-->

# <TASK_ID> — <task title>

| | |
| --- | --- |
| **Branch** | `epic/refactor/<TASK_ID>-<short-slug>` |
| **Agent / date** | <model>, <YYYY-MM-DD> |
| **Wall clock** | <actual> (budget: <from the task file>) |
| **Commits** | <n>, listed below |

---

## 1. What was done

Two or three sentences. What changed, in what files, and why that shape.

---

## 2. Definition of Done

Copy the checklist from the task file **verbatim** and tick each box. An unticked
box means the status cannot be `GREEN`. For every item with a command, paste the
actual output.

- [ ] …

---

## 3. The gate

```text
<paste the summary table from `uv run python scripts/check.py`>
```

---

## 4. Measurements

The numbers, not the adjectives. Tables beat prose. Include, where the task
produced them: row / session / date-range counts; before/after comparisons with
the delta stated explicitly; test counts and runtime; fingerprints, hashes,
versions.

```text
<paste real command output here>
```

---

## 5. What was ruled out, and by what evidence

A hypothesis eliminated is worth as much as one confirmed. If you considered an
approach and rejected it, say which and why — the next agent should not have to
re-reason it.

---

## 6. Caveats

**Required if `STATUS: AMBER`.** For each caveat state three things: what is
unknown or incomplete, what it affects downstream, and what would resolve it (a
command, a question ID, a measurement).

---

## 7. Observations

Things you noticed that are **outside your task**. Do not act on them. This is
where a defect, a doc drift, or a refactor opportunity gets recorded so a later
task can pick it up.

---

## 8. Questions raised

`Q-nn` IDs appended to `QUESTIONS.md`, one line each on what they block. `NONE`
is a fine answer.

---

## 9. Commits

```text
<sha>  <subject>
```

## 10. Files changed

```text
<git diff --stat epic/refactor...HEAD>
```
