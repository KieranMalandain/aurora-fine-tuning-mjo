#!/usr/bin/env python3
"""Test num_workers compatibility with the fork-safe LANLMJODataset."""
import time, signal, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from src.dataset import LANLMJODataset, collate_fn
from torch.utils.data import DataLoader

BATCHES = 4
TIMEOUT = 90

print("Creating fork-safe LANLMJODataset (1980-1981, 2 years)...")
t0 = time.time()
ds = LANLMJODataset(
    start_year=1980, end_year=1981,
    root_dir="/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results",
)
print(f"Init: {time.time()-t0:.1f}s  |  len={len(ds)}")

results = {}

for nw in [0, 2, 4]:
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  num_workers={nw}  (timeout={TIMEOUT}s)")
    print(sep)

    def handler(signum, frame):
        raise TimeoutError(f"DEADLOCK num_workers={nw}")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT)

    try:
        kwargs = {}
        if nw > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = True
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=nw,
            pin_memory=False, collate_fn=collate_fn, **kwargs,
        )
        times = []
        t0 = time.time()
        for i, batch in enumerate(loader):
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  Batch {i}: {elapsed:.2f}s", flush=True)
            if i >= BATCHES - 1:
                break
            t0 = time.time()

        signal.alarm(0)
        avg = sum(times) / len(times)
        avg_warm = sum(times[1:]) / max(len(times) - 1, 1)
        print(f"  OK  avg={avg:.2f}s  warm_avg={avg_warm:.2f}s  (first={times[0]:.2f}s)")
        results[nw] = {"status": "OK", "avg": avg, "warm_avg": avg_warm, "first": times[0]}
        del loader
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  DEADLOCKED: {e}")
        results[nw] = {"status": "DEADLOCK"}
        try: del loader
        except: pass
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        results[nw] = {"status": f"ERROR: {e}"}
        try: del loader
        except: pass

print(f"\n{'='*50}")
print("  SUMMARY")
print(f"{'='*50}")
for nw, r in results.items():
    if r["status"] == "OK":
        print(f"  num_workers={nw}:  OK  avg={r['avg']:.2f}s  warm={r['warm_avg']:.2f}s")
    else:
        print(f"  num_workers={nw}:  {r['status']}")

if all(r["status"] == "OK" for r in results.values()):
    # Find the best warm avg
    best = min(results.items(), key=lambda x: x[1].get("warm_avg", 999))
    print(f"\n  Recommendation: num_workers={best[0]}")
print()
