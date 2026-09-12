# Aurora–MJO Fine-Tuning — Vision, Root-Cause Diagnosis & Execution Plan

**Author:** Orchestrator (Opus 4.8) · **Executor:** Worker agent (Sonnet 5)
**Status:** Baseline run confirmed fully non-finite. Two-part root cause identified. Fixes specified below.
**Hard constraint:** Perlmutter goes offline in ~2 days. Batch queue (~50 h to schedule) is unusable. Everything runs in chained ≤4 h interactive `salloc` sessions.

---

## PART 0 — Strategic direction (read this first)

### 0.1 The deadline dictates the architecture of the whole campaign
- **Batch queue is dead.** A ~50 h scheduling latency against a ~48 h runway means we cannot use `sbatch`. Every phase must run inside interactive allocations obtained via `salloc` (fast to grant, **4 h wall-clock cap each**).
- **Therefore every phase must be checkpoint-resumable across sessions.** The machinery already exists — SIGUSR1 handler (`trainer.py:_on_usr1`), `ResumableDistributedSampler`, step-level checkpointing (`_maybe_step_checkpoint`), and `math.isfinite` best-tracking. We lean on it. We do **not** need one 4 h session to finish a phase; we need N sessions that each make durable, resumable progress.
- **Testing shares the training allocation.** One interactive node at a time; smoke tests and training run in the same `salloc`. Budget ~30 min of each session for verification before committing the rest to training.

### 0.2 Root cause in one paragraph
The baseline dies because of **a trigger plus a corruption path**. Trigger: the `msl` channel actually carries surface pressure (`ps`) but is normalized with Aurora's mean-sea-level-pressure climatology, driving high-terrain inputs to **−36σ** on essentially every global field, which destabilizes the (unstabilized, bf16) small-model attention. Corruption: under bf16 the `GradScaler` is disabled, so the optimizer boundary steps with **no gradient-finiteness check**; a finite-loss/non-finite-gradient batch poisons Adam's moment buffers (this happens even at lr=0 because Adam updates moments before applying lr), after which every step and every forward is NaN forever. **We must fix both.** msl renorm removes the trigger; the gradient guard removes the persistence.

### 0.3 Model-size decision: decide on evidence, not a priori
Your instinct ("why ship the debug model as the final run?") is correct in the abstract. The deadline changes the calculus:
- **Full Aurora** (~1.3 B params) is slower per step and its activation memory at 0.25° (720×1440) across pressure levels is the open question. It *also* sets `stabilise_level_agg=True` by default (adds K/Q LayerNorm in level aggregation) → **inherently more NaN-resistant** than the small model, and with a frozen backbone + LoRA the trainable/optimizer footprint is tiny (activation memory dominates, not optimizer state).
- **Small Aurora** is explicitly "for debugging" but is the only config already known to fit 4×A100 and is guaranteed to finish inside the runway.
- **Decision procedure:** Task S0 runs a 20-minute memory+throughput probe of *both* sizes in the first session. Pick full **iff** it fits in memory with headroom and its per-step time lets baseline finish in ≤~2 sessions. Otherwise small. Small is the guaranteed-deliverable fallback; never bet the whole runway on an untested full-model run.

### 0.4 Priority-ordered deliverables (do them in this order; stop when the machine dies)
1. **Correct, complete baseline** (single-step, grid loss) → a usable MJO model. *This alone is a defensible result.*
2. **Evaluate baseline** (`evaluate_mjo.py` → RMM/MJO skill, bivariate correlation, RMSE).
3. **LoRA rollout** (warm-started from baseline; real k=1→4 curriculum). *High scientific value — extends the forecast horizon that matters for MJO's intraseasonal timescale.*
4. **Physics-informed** *only if* time remains **and** the freezing is fixed (§Finding 5). Lowest value as currently wired.
5. **Combined**, time permitting.

Phases 3–5 are explicitly conditional. A trained + evaluated baseline is the floor we must clear.

---

## PART 1 — Authoritative root-cause diagnosis

### Finding 1 — `msl` is surface pressure normalized as MSL (the trigger) — CONFIRMED
`dataset.py:61` maps `'msl' → (…/PS, 'ps')` (documented proxy: LANL data has no MSL). But Aurora normalizes the `msl` channel with `location=100958, scale=1332` (MSL). Using the real first-non-finite val batch ranges and Aurora's real stats, normalized surface inputs are:

| var | physical range | normalized range | note |
|-----|----------------|------------------|------|
| 2t | [228.7, 316.2] K | [−2.35, 1.78] | ok |
| 10u | [−18.1, 22.9] | [−3.25, 4.14] | ok |
| 10v | [−22.1, 20.5] | [−4.69, 4.26] | ok |
| **msl** | **[52740, 104400] Pa** | **[−36.19, 2.58]** | **FAR OOD** |
| ttr | [−363, −90] | [−2.79, 2.77] | ok (injected) |
| tcwv | [0.34, 78.6] | [−1.09, 3.67] | ok |

527 hPa only occurs over the Tibetan Plateau / Antarctic ice sheet — this is unmistakably surface pressure. Renormalizing the channel with `ps` statistics brings it to ≈[−4.6, 0.8]σ. High terrain exists in *every* global field, which is why validation is 100% non-finite and training fails on ~every batch.

### Finding 2 — no gradient-finiteness check at the optimizer step (the corruption path) — NEW, explains "NaN forever"
`trainer.py:480-482`: `GradScaler(enabled = use_amp and amp_dtype==float16)`. Under bf16 this is **disabled**. `trainer.py:885-893`:
```python
if is_boundary:
    self.grad_scaler.unscale_(self.optimizer)      # no-op (scaler disabled)
    if self.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(...)              # clips norm, does NOT reject NaN/Inf
    self.grad_scaler.step(self.optimizer)          # degrades to bare optimizer.step(); NO inf-check
    self.grad_scaler.update()                      # no-op
    ...
```
The nan-guard (`_guarded_backward`) only checks `isfinite(loss)`. A **finite loss with non-finite gradients** slips straight through — and that is exactly what bf16 attention overflow produces (saturated softmax → finite activation, NaN in its Jacobian). `clip_grad_norm_` computes a total norm of NaN and does not sanitize. So NaN reaches `optimizer.step()`. Adam then does `m = β1·m + (1−β1)·g` and `v = β2·v + (1−β2)·g²` **before** multiplying by lr, so the moment buffers are poisoned even during warmup at lr≈0. Every subsequent step drives weights to NaN → every forward NaN. This is the persistence you observed (finite at step 0, dead by step 50).

The surrogate backward (`_surrogate_sync_backward`, `(z*0.0).backward()` built from parameters, not the loss) is **correct** and does not need changing.

### Finding 3 — LoRA rollout feeds untrained heads back unclamped (amplifier for phase 3) — CONFIRMED
`_advance_batch` splices the model's own prediction into the next input for rollout, without clamping, and bypasses Aurora's built-in `apply_rollout_input_clipping`. From a cold start the fresh ttr/tcwv heads emit garbage; step-2 forward is OOD → NaN (observed: k=4 step-0 pristine weights, step 1 finite, steps 2–4 NaN). Fix = warm-start from baseline + ramp k from 1 + clamp fed-back predictions.

### Finding 4 — validation diagnostic never logs the prediction (why this stayed unsolved)
`_log_bad_val_batch` (`trainer.py:949-966`) logs surface inputs, only `q`/`t` atmospherics, and targets — never `z`/`u`/`v`, never statics, and **never the model prediction**, which is the tensor that actually goes NaN. We add prediction logging so the fixes can be *confirmed* rather than assumed, and so any residual per-variable issue (e.g. a `z` geopotential units mismatch) is caught in one smoke run.

### Finding 5 — physics-informed freezes the entire backbone (correctness, not a crash)
`freeze_backbone` defaults `True` and the physics config sets neither `use_lora` nor `freeze_backbone`, so only the ttr/tcwv embeddings + heads train. The moisture-budget loss depends on predicted `q,u,v`, whose decoders are frozen — it cannot enforce moisture conservation in the fields it scores. Flag for a human decision (§Part 4); do not spend runway on this phase until resolved.

### Finding 6 — msl output head is frozen and MSL-calibrated (companion to Finding 1)
`freeze_backbone` only unfreezes the *injected* vars (ttr, tcwv); the native `msl` head stays frozen and was trained to emit MSL-scaled values. After the input renorm (Finding 1), that frozen head denormalizes with `ps` stats and becomes miscalibrated (finite but wrong, ~7× spread), inflating grid loss over terrain. Fix = unfreeze the `msl` head so it re-learns ps-scaled output, making the channel coherent end-to-end.

---

## PART 2 — The fixes (exact, implementable)

> Apply FIX 1–4 before any run. FIX 5 is the model-size probe (Task S0). FIX 6 is conditional.

### FIX 1 — Gradient-finiteness guard at the optimizer boundary  *(highest priority — makes the whole pipeline robust)*
**File:** `trainer.py`, replace the boundary block at ~lines 885-893.
```python
if is_boundary:
    self.grad_scaler.unscale_(self.optimizer)  # no-op under bf16, harmless

    # --- NEW: reject non-finite gradients before they poison Adam moments ---
    local_finite = True
    for p in self.model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            local_finite = False
            break
    # Make the decision collective so all DDP ranks step-or-skip together.
    grads_finite = self._collective_all_finite(local_finite)

    if grads_finite:
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
    else:
        self._nonfinite_grad_steps = getattr(self, "_nonfinite_grad_steps", 0) + 1
        if self.is_main:
            log.warning(f"[grad-guard] non-finite grads at step {self.global_step} "
                        f"— optimizer step skipped (total={self._nonfinite_grad_steps}).")
        # Do NOT step optimizer or scheduler; just clear the poisoned grads.

    self.optimizer.zero_grad(set_to_none=True)
    self.global_step += 1
    self._steps_since_save += 1
    ...
```
Notes for the executor:
- Initialize `self._nonfinite_grad_steps = 0` in `__init__` next to `self._nonfinite_train_batches`, and add it to the metrics dict + the checkpoint state dict so it survives resume.
- `_collective_all_finite` already exists (`trainer.py:515`) and uses `all_reduce(MIN)`; it must be called once per boundary by all ranks — it is, because boundaries are symmetric. Keep the collective even though DDP-averaged grads are usually identical across ranks; it protects against a rank with a `None` grad.
- This is the fix that turns "NaN forever" into "skip the bad step and keep going."

### FIX 2 — Renormalize `msl` as surface pressure  *(removes the trigger)*
**Step 2a — extend the stats calculator.** `calc_norm_stats.py`, add `msl` to `VARIABLES_TO_CALC`:
```python
VARIABLES_TO_CALC = {
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS', 'ps'),
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', 'mtnlwrf'),
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv', 'tcwv'),
}
```
Run it against the baseline config to get exact `ps` mean/std over the training years:
```bash
python calc_norm_stats.py --config phase1_baseline.yaml
```
**Step 2b — inject the result.** Add `msl` to `model.norm_stats` in **every** phase config (`phase1_baseline.yaml`, `phase2_physics.yaml`, `phase2_rollout*.yaml`, `phase3_longrun.yaml`, and `unified.yaml`):
```yaml
model:
  norm_stats:
    msl:  { mean: <computed_ps_mean>, std: <computed_ps_std> }   # e.g. ~96647 / ~9587
    ttr:  { mean: -226.0498, std: 49.2158 }
    tcwv: { mean: 18.2967,   std: 16.3265 }
```
`load_model` (`model.py:381-385`) writes these into Aurora's global `locations`/`scales`, overriding the built-in MSL stats. Fallback if the stats job can't run in time: Aurora's own `sp` stats `mean=96647.375, std=9586.6914` (pulls msl to ≈−4.6σ — good enough to unblock).

### FIX 3 — Log the prediction in the val diagnostic  *(confirms FIX 1–2 work)*
**File:** `trainer.py`, replace `_log_bad_val_batch` (~lines 949-966):
```python
@torch.no_grad()
def _log_bad_val_batch(self, in_batch, target_dict_list):
    def rng(t):
        t = t.float()
        fin = torch.isfinite(t)
        tag = "" if fin.all() else f" NON-FINITE({int((~fin).sum())})"
        v = t[fin]
        return (f"[{v.min():.3e}, {v.max():.3e}]{tag}" if v.numel() else "[empty]")
    lines = []
    # inputs: ALL surface + ALL atmos (not just q,t)
    for name, t in in_batch.surf_vars.items():
        lines.append(f"in.surf.{name}={rng(t)}")
    for name, t in in_batch.atmos_vars.items():
        lines.append(f"in.atmos.{name}={rng(t)}")
    # the missing measurement: the model's own prediction, per variable
    self.model.eval()
    with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
        out = self.model(in_batch)
    pred = out[0] if isinstance(out, tuple) else out
    for attr in ("surf_vars", "atmos_vars"):
        for name, t in getattr(pred, attr).items():
            lines.append(f"PRED.{attr[:4]}.{name}={rng(t)}")
    tgt = target_dict_list[0] if isinstance(target_dict_list, list) else target_dict_list
    for name, t in list(tgt.items())[:6]:
        lines.append(f"tgt.{name}={rng(t)}")
    log.error("[val-diag] first non-finite val batch: " + " | ".join(lines))
```
On the first smoke test **read this line**: it tells you which prediction variable (if any) is first to go non-finite. Expectation after FIX 1–2: no `PRED.*` is non-finite. If one still is, that variable needs its own attention (most likely candidate: `z` geopotential units — verify `gopt` is m²/s², not geopotential height in m).

### FIX 4 — Rollout robustness (phase 3 only)
**File:** `trainer.py`, in `_advance_batch`, clamp fed-back predictions before splicing:
```python
_ROLLOUT_CLAMP = {  # physical bounds; tune to your normalization convention
    "msl": (3.0e4, 1.1e5), "2t": (150., 350.), "10u": (-120., 120.), "10v": (-120., 120.),
    "ttr": (-600., 50.), "tcwv": (0., 120.), "q": (0., 0.1),
}
def _clamp_fed(name, t):
    b = _ROLLOUT_CLAMP.get(name)
    return t.clamp(*b) if b else t
# apply _clamp_fed to each predicted var right before torch.cat into the next input.
```
Plus, at the campaign level (not code): warm-start phase 3 from the finished baseline checkpoint, and use the real curriculum (`rollout.start_steps` ramp from k=1), **not** the `start_steps=4` debug override.

### FIX 5 — Model-size probe + bf16 robustness knobs (Task S0)
- Probe both `model_type: small` and full in a throwaway 30-step run; record peak `torch.cuda.max_memory_allocated()` per rank and seconds/step.
- If retaining the small model, add a robustness margin: set `training.sdpa_backend: "math"` if that knob exists in `train.py`/config (avoids the mem-efficient attention path), or run the encoder/level-agg forward in fp32. The full model already has `stabilise_level_agg=True`, so if it fits, it is the more robust choice.
- Do **not** enable gradient checkpointing casually — it previously caused an illegal-memory-access crash (`train_rollout_52464118`). Only revisit if the full model needs it for memory, and smoke-test it in isolation first.

### FIX 6 — Unfreeze the `msl` head (companion to FIX 2; apply with FIX 2)
**File:** `model.py`, in `freeze_backbone`, add `'msl'` to the set of decoder heads that get `requires_grad_(True)` (alongside ttr/tcwv). This lets the head re-learn ps-scaled output so the renormalized channel is coherent end-to-end. Small parameter count; low risk. (Alternative if you prefer minimal change: down-weight or exclude `msl` in the grid loss — but unfreezing is cleaner and keeps msl a learned variable.)

---

## PART 3 — Sequential execution plan (interactive sessions)

Standard allocation (yours, works):
```bash
MODE=baseline salloc -A m4946_g -C 'gpu&hbm80g' -q interactive -t 03:50:00 -N 1 --gpus-per-node=4 -c 64
```

### SESSION A — Fix, smoke-test, confirm finite (do not start a long run until this passes)
- **Task A1.** Apply FIX 1, FIX 3, FIX 4, FIX 6 (code). Apply FIX 2a (calc script edit).
- **Task A2.** In an allocation, run FIX 2b: `python calc_norm_stats.py --config phase1_baseline.yaml`; paste the `msl` mean/std into all phase configs.
- **Task A3.** Run Task S0 (FIX 5 probe): 30 steps each of small vs full; record peak memory + s/step. **Decide model size.** Write the decision + numbers into the run log.
- **Task A4.** Baseline smoke test: ~100 training steps + one validation pass with the chosen model. **Success criteria, all required:**
  1. Training loss stays finite for 100 consecutive steps (`nan_skipped_total` flat at 0).
  2. `_nonfinite_grad_steps` is 0 (or a tiny transient that the guard absorbs without the run dying).
  3. Validation reports `n_ok > 0` (not 200 skipped).
  4. The `[val-diag]` line (force one by temporarily lowering the val NaN bar, or inspect a healthy batch) shows **no** `PRED.*` non-finite and physically sane prediction ranges.
- **Task A5.** If any `PRED.*` variable is still non-finite, stop and report which one + its input range (most likely `z` units). Do not proceed to training.

### SESSION B (and C…) — Baseline training, resumable
- **Task B1.** Launch full baseline with aggressive step-checkpointing (checkpoint interval sized so ≤~20 min of work is ever at risk; SIGUSR1 will also flush on the 4 h boundary).
- **Task B2.** When the allocation ends, re-`salloc` and resume from `latest`. Verify the resume loads optimizer + scheduler + sampler position and that loss continues finite. Repeat until baseline completes its scheduled epochs (throughput from A3 tells you how many sessions).
- **Task B3.** On completion, snapshot the baseline checkpoint under a stable name (e.g. `checkpoints/baseline_final`) so phase 3 can warm-start from it deterministically.

### SESSION D — Evaluate baseline (this is the result)
- **Task D1.** Run `evaluate_mjo.py` (+ `compute_rmm.py`) against the val/test years to produce RMM indices, bivariate correlation, and RMSE vs lead time. Save artifacts. **This is a complete, reportable deliverable even if nothing else finishes.**

### SESSION E — LoRA rollout (stretch, high value)
- **Task E1.** Configure phase 3: `init_from: checkpoints/baseline_final`, `use_lora: true`, real k=1→4 curriculum (drop the `start_steps=4` debug override), FIX 4 clamps active.
- **Task E2.** Smoke test (Session-A criteria) *before* the long run — especially confirm step-2+ rollout stays finite now that heads are warm-started and clamped.
- **Task E3.** Train resumably as in Session B; evaluate as in Session D.

### SESSION F — Physics-informed (only if time AND Finding 5 resolved)
- **Task F1.** Get the human decision on §Part 4 item 2 (what to unfreeze). Do not run the frozen-backbone version — it cannot learn what the physics loss scores.

---

## PART 4 — Decisions the human must make
1. **Model size:** ratify or override the Task-A3 probe result. Default recommendation: full if it fits with headroom and finishes baseline in ≤~2 sessions; else small.
2. **Physics phase scope:** if we run it, which parameters train? Options: (a) LoRA adapters + injected heads (physics loss can shape moisture via adapters), (b) unfreeze relevant decoders, (c) skip the phase entirely for this runway. Recommendation: (c) unless baseline+lora finish early, then (a).
3. **Runway allocation:** if forced to choose, is a fully-evaluated baseline+lora preferable to attempting all four phases and risking none finishing cleanly? Recommendation: yes — depth over breadth given the deadline.

---

## Appendix — What is NOT the problem (so we don't chase it)
- **Data alignment** (dataset v3 timestamp intersection): structurally sound; ruled out.
- **Input data validity:** the val-diag input ranges are all physically sane and finite; the NaN is born in the forward, not the data.
- **Eval-vs-train mode:** Aurora's forward is numerically identical in train and eval (dropout/droppath = 0; the only `self.training` branch is a memory `del`). Eval mode does not cause the val NaN.
- **The surrogate backward:** correct as written; leave it alone.
- **Historical errors** `52464118` (cuBLAS/bf16 checkpointing IMA) and `52565795` (state_dict mismatch): already addressed; not active causes.
