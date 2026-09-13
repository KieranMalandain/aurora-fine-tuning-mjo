# B2 — Grep July SLURM logs for `Using zeros`

| | |
| --- | --- |
| **Phase** | B |
| **Depends on** | none |
| **Base branch** | `epic/refactor` |
| **Budget** | 45 min |
| **Touches** | `docs/findings/2026-09-zeroed-statics.md` (new), `results/B2_result.md` |
| **Must not touch** | any `.py`, `.yaml` or `.slurm` file. **Do not delete or move any log file.** |
| **Needs** | Perlmutter, read access to the July SLURM logs on `/pscratch` |

## Objective

Answer, with evidence, open question 1 from `cleanup-2026-09.md` §10: **were the
July 2026 training runs affected by the zeroed-statics bug?** The answer
determines whether any prior result on this project means anything.

## Why this is a separate task

It is read-only, fast, needs no code changes, and it is the highest
information-per-minute task in the campaign. It also has no dependencies, so it
can run concurrently with all of Phase A. And it answers a question that outlives
this campaign — which is why the deliverable is a document in `docs/findings/`,
not just a result file.

## What you may assume

`src/dataset.py::_load_static_vars` catches `Exception` around the invariant `z`
and `lsm` loads and substitutes **zero tensors**, warning only:

```python
except Exception as e:
    warnings.warn(f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros.")
```

There is a matching handler for `lsm`. **`slt` has no such handler** — it would
raise. (`02_UPSTREAM_CONTRACT.md` §4.3)

The exception this would most likely catch is the HDF5 file-locking error, which
requires `HDF5_USE_FILE_LOCKING=FALSE` to avoid and which is currently set only
in `slurm_scripts/train_auto.slurm`. (`02_UPSTREAM_CONTRACT.md` §4.5)

If it fired, those runs trained on **zero surface geopotential and a zero
land-sea mask** — a planet with no topography and no continents — while printing
two `UserWarning` lines into an 11-hour log. This compounds with the `msl`
mis-normalisation (Lesson 1).

Search strings, in order of specificity:

```text
Using zeros
Failed to load invariant
NetCDF: HDF error
Errno -101
HDF5_USE_FILE_LOCKING
```

## Steps

1. Locate the July logs. Likely candidates, but **verify rather than assume**:
   `/pscratch/sd/k/kam352/Aurora/aurora-fine-tuning-mjo/slurm_logs/`, any
   `slurm_logs/` in the former worktree directories, and
   `/pscratch/sd/k/kam352/Aurora/cleanup-reports/`. Record every directory you
   searched **and every one you expected to exist but did not** — `/pscratch` is
   purgeable, so absence is itself a finding.

2. Inventory before searching. For each log file: path, size, mtime, and the
   SLURM job ID from the filename. Establish which are July 2026 and which are
   not. A `Using zeros` hit in a September diagnostic run means something
   completely different from a hit in a July training run.

3. Search. Use `grep -rn` with `-i` off (the strings are known-case) and record
   counts per file:

   ```bash
   grep -rc "Using zeros" <logdir>/ 2>/dev/null | grep -v ':0$'
   grep -rn "Failed to load invariant" <logdir>/ | head -50
   grep -rn "NetCDF: HDF error\|Errno -101" <logdir>/ | head -50
   ```

4. For every hit, capture **20 lines of context** and record: job ID, timestamp,
   which variable (`Z` or `LSM` or both), and the underlying exception text. The
   exception text is the diagnostically important part — an HDF5 locking error
   means one thing, a `FileNotFoundError` means something quite different.

5. Cross-reference against whether that job set the environment variable. Either
   the log contains the `[chain]` banner from `train_auto.slurm` (which does set
   it), or it does not. A run launched from an interactive `salloc` would not
   have it. Record, per job, whether the variable was plausibly set.

6. Correlate with outcomes. `docs/archive/metrics/baseline-2026-07.jsonl` and
   `lora-2026-07.jsonl` (relocated by A2; at `checkpoints/*/metrics.jsonl` if A2
   has not merged) hold the metric traces for those runs. Check whether jobs
   showing `Using zeros` correspond to the runs whose losses went non-finite.
   **State clearly whether this is correlation or established causation** — with
   two candidate causes already known (zeroed statics and `msl`
   mis-normalisation), and Lesson 1 already *confirmed* as a trigger, a
   correlation here does not establish that zeroed statics contributed. Do not
   overclaim.

7. Write `docs/findings/2026-09-zeroed-statics.md` in the build guide's findings
   format: **Symptom**, **Root cause**, **Evidence**, **What this RULES OUT**,
   **The fix**, **Open**.

   The "What this RULES OUT" section is the valuable one. If no July log contains
   `Using zeros`, that **rules out** zeroed statics as a contributor to the July
   failures and leaves Lesson 1 as the sole confirmed cause — which is real
   information and makes the July metrics traces interpretable. Say so plainly.

   If the logs are simply gone, the finding is "unresolvable, and here is why,
   and here is what would have resolved it". That is a legitimate and useful
   outcome. Report **AMBER**, not RED, and set the finding's status to
   `UNRESOLVED — logs purged`.

## Definition of Done

- [ ] Every directory searched is listed, including ones that did not exist
- [ ] Log inventory table: path, size, mtime, job ID, July-or-not
- [ ] Hit counts per file for all five search strings — output pasted
- [ ] For each hit: job ID, timestamp, variable, and the underlying exception
      text, with 20 lines of context
- [ ] Per-job assessment of whether `HDF5_USE_FILE_LOCKING` was plausibly set
- [ ] Correlation against the July metrics traces, with correlation-vs-causation
      stated explicitly
- [ ] `docs/findings/2026-09-zeroed-statics.md` written with all six sections,
      including **What this RULES OUT**
- [ ] A one-line verdict: `AFFECTED` / `NOT AFFECTED` / `UNRESOLVABLE`, with the
      evidence that supports it
- [ ] **No log file moved, modified or deleted** — stated explicitly
- [ ] Result file written from `results/_TEMPLATE.md`

## Out of scope

- **Fixing the fallback.** That is E1, and E1 goes ahead regardless of what you
  find. The code should hard-fail either way; your finding only determines
  whether past results mean anything.
- Re-running any July job.
- Deleting logs, even if they are large. `/pscratch` capacity is not this
  campaign's problem, and a deleted log cannot be re-read.
- Interpreting the July *scientific* results. Whether the runs were affected is
  your question; what the losses mean is not.
