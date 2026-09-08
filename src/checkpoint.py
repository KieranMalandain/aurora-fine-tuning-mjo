# src/checkpoint.py
"""
CheckpointManager — atomic, resumable, step-granular checkpointing.

Why this exists
===============
The two failed Perlmutter jobs exposed three resume failures:

  1. Job 52565795 crashed on resume with strict load_state_dict because the
     on-disk checkpoint predated the ttr/tcwv variable injection.  We now
     load with strict=False and LOG every missing/unexpected key instead of
     dying, and we refuse silently-broken optimizer restores.
  2. Checkpoints were only written at epoch boundaries.  One epoch of the
     LANL dataset is ~13.5k steps/rank (>6 h) - longer than a 12 h SLURM
     allocation minus dataset-init time, so a job could die with ZERO
     checkpoints.  We now save every N optimizer steps (default 500),
     on loss plateau, on SIGUSR1 (SLURM timeout warning), and on wall-clock
     guard expiry.
  3. Nothing recorded the global step / epoch / scheduler / AMP-scaler /
     RNG state, so "resume" silently restarted the cosine schedule and the
     rollout curriculum from scratch.  All of that state is now saved.

Atomicity: we write to `<name>.pt.tmp` then os.replace() - a job killed
mid-save can never leave a truncated file that later poisons `find_latest`.

A `latest.txt` pointer file (not a symlink - safer on CFS/scratch) always
names the most recent complete checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

_POINTER = "latest.txt"


class CheckpointManager:
    """Save/load full training state with atomic writes and pruning.

    Args:
        save_dir:    Directory for checkpoints (created if missing).
        keep_last_n: How many rolling step/epoch checkpoints to keep.
                     `best.pt` and the file named in latest.txt are NEVER pruned.
        is_main:     True only on rank 0; other ranks no-op on save.
    """

    def __init__(self, save_dir: str | Path, keep_last_n: int = 3, is_main: bool = True):
        self.save_dir = Path(save_dir)
        self.keep_last_n = max(1, keep_last_n)
        self.is_main = is_main
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self._rolling: list[Path] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        tag: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        grad_scaler=None,
        *,
        epoch: int = 0,
        global_step: int = 0,
        batch_in_epoch: int = 0,
        best_val: float = float("inf"),
        val_loss: float = float("nan"),
        config: dict | None = None,
        is_best: bool = False,
        prune: bool = True,
        nonfinite_train_batches: int = 0,
        nonfinite_grad_steps: int = 0,
    ) -> Path | None:
        """Write one checkpoint atomically.  Rank-0 only (others return None).

        The payload contains EVERYTHING needed for bit-faithful resume:
        model weights (DDP-unwrapped), optimizer moments, LR-scheduler step,
        AMP GradScaler state, counters, RNG states, and the config that
        produced the run (so a resumed job can detect config drift).
        """
        if not self.is_main:
            return None

        t0 = time.time()
        # CRITICAL: always unwrap DDP so keys are stable regardless of how
        # many GPUs the *next* job uses.
        raw = model.module if hasattr(model, "module") else model

        payload = {
            "format_version": 2,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler else None,
            # ---- resume counters (all three matter) -----------------------
            "epoch": epoch,                    # epoch currently in progress
            "global_step": global_step,        # optimizer steps completed
            "batch_in_epoch": batch_in_epoch,  # batches consumed THIS epoch
            "best_val": best_val,
            "val_loss": val_loss,
            # FIX 1 (AURORA_MJO_GAMEPLAN): survive resume across interactive
            # sessions so the lifetime skip counts stay accurate.
            "nonfinite_train_batches": nonfinite_train_batches,
            "nonfinite_grad_steps": nonfinite_grad_steps,
            "config": config,
            # ---- RNG so shuffles/dropout continue deterministically -------
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "timestamp": time.time(),
        }

        final = self.save_dir / f"{tag}.pt"
        tmp = self.save_dir / f"{tag}.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, final)  # atomic on POSIX — no torn checkpoints

        # Update the latest-pointer atomically too.
        ptr_tmp = self.save_dir / (_POINTER + ".tmp")
        ptr_tmp.write_text(final.name)
        os.replace(ptr_tmp, self.save_dir / _POINTER)

        if is_best:
            best_tmp = self.save_dir / "best.pt.tmp"
            torch.save(payload, best_tmp)
            os.replace(best_tmp, self.save_dir / "best.pt")

        size_mb = final.stat().st_size / 1e6
        log.info(
            f"[ckpt] saved {final.name}  size={size_mb:.1f} MB  "
            f"epoch={epoch} step={global_step} batch_in_epoch={batch_in_epoch} "
            f"took={time.time() - t0:.1f}s" + ("  (new best)" if is_best else "")
        )

        if prune:
            self._rolling.append(final)
            latest_name = (self.save_dir / _POINTER).read_text().strip()
            while len(self._rolling) > self.keep_last_n:
                old = self._rolling.pop(0)
                # Never delete best.pt or whatever latest.txt points at.
                if old.name in ("best.pt", latest_name):
                    continue
                if old.exists():
                    old.unlink()
                    log.info(f"[ckpt] pruned {old.name}")
        return final

    # ------------------------------------------------------------------
    # Discover
    # ------------------------------------------------------------------

    @staticmethod
    def find_latest(save_dir: str | Path) -> Path | None:
        """Locate the newest complete checkpoint in a directory.

        Order of trust:
          1. latest.txt pointer (only ever names a fully-written file)
          2. best.pt
          3. newest *.pt by mtime (ignoring *.tmp)
        """
        d = Path(save_dir)
        if not d.is_dir():
            return None
        ptr = d / _POINTER
        if ptr.exists():
            cand = d / ptr.read_text().strip()
            if cand.exists():
                return cand
        best = d / "best.pt"
        if best.exists():
            return best
        pts = sorted(
            (p for p in d.glob("*.pt") if not p.name.endswith(".tmp")),
            key=lambda p: p.stat().st_mtime,
        )
        return pts[-1] if pts else None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        grad_scaler=None,
        *,
        map_location="cpu",
        restore_rng: bool = True,
        weights_only_model: bool = False,
    ) -> dict:
        """Restore state from `path`.  Never hard-crashes on key mismatch.

        This directly fixes the job-52565795 failure mode:
        - model weights load with strict=False; missing keys (e.g. a newly
          injected variable's embedding/decoder head that the old checkpoint
          predates) keep their fresh initialization and are LOGGED loudly.
        - optimizer/scheduler/scaler restores are individually try/except-ed:
          if the trainable-parameter set changed between runs, we warn and
          continue with fresh optimizer state rather than aborting the job.

        Args:
            weights_only_model: if True, ONLY model weights are restored and
                counters are zeroed - use when warm-starting a new phase
                (e.g. combined training initialized from the physics run).

        Returns:
            dict with epoch / global_step / batch_in_epoch / best_val.
        """
        path = Path(path)
        t0 = time.time()
        # weights_only=False is required: payload holds config dicts + RNG
        # tuples. These are our own files from our own runs.
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        raw = model.module if hasattr(model, "module") else model
        state = ckpt.get("model_state_dict", ckpt)  # tolerate bare state dicts
        missing, unexpected = raw.load_state_dict(state, strict=False)
        if missing:
            log.warning(
                f"[ckpt] {len(missing)} keys MISSING from checkpoint (kept at "
                f"fresh init — expected when new variables were added): {missing}"
            )
        if unexpected:
            log.warning(f"[ckpt] {len(unexpected)} UNEXPECTED keys ignored: {unexpected}")

        counters = {
            "epoch": 0, "global_step": 0, "batch_in_epoch": 0, "best_val": float("inf"),
            "nonfinite_train_batches": 0, "nonfinite_grad_steps": 0,
        }
        if weights_only_model:
            log.info(f"[ckpt] warm-start (weights only) from {path.name} in {time.time()-t0:.1f}s")
            return counters

        if optimizer is not None and ckpt.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError, RuntimeError) as e:
                log.warning(f"[ckpt] optimizer restore FAILED ({e}); continuing with fresh optimizer.")
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError, RuntimeError) as e:
                log.warning(f"[ckpt] scheduler restore FAILED ({e}); continuing with fresh scheduler.")
        if grad_scaler is not None and ckpt.get("grad_scaler_state_dict"):
            try:
                grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
            except (ValueError, KeyError, RuntimeError) as e:
                log.warning(f"[ckpt] grad-scaler restore FAILED ({e}); continuing fresh.")

        if restore_rng and ckpt.get("rng"):
            try:
                rng = ckpt["rng"]
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(torch.as_tensor(rng["torch"], dtype=torch.uint8))
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [torch.as_tensor(s, dtype=torch.uint8) for s in rng["cuda"]]
                    )
            except Exception as e:  # RNG restore is best-effort, never fatal
                log.warning(f"[ckpt] RNG restore skipped: {e}")

        for k in counters:
            counters[k] = ckpt.get(k, counters[k])
        log.info(
            f"[ckpt] resumed from {path.name}: epoch={counters['epoch']} "
            f"step={counters['global_step']} batch_in_epoch={counters['batch_in_epoch']} "
            f"({time.time()-t0:.1f}s)"
        )
        return counters


class MetricsLogger:
    """Append-only JSONL metrics - safe to resume (we just keep appending).

    One line per record: {"step": ..., "epoch": ..., "split": "train"|"val",
    "loss": ..., component losses..., "lr": ..., "t": unix_time}.
    Rank-0 only; other ranks construct a no-op instance.
    """

    def __init__(self, path: str | Path, is_main: bool = True):
        self.is_main = is_main
        self.path = Path(path)
        if is_main:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        if not self.is_main:
            return
        record = {**record, "t": time.time()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            