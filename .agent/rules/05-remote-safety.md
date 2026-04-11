---
trigger: always_on
---

# Remote development safety

This repo is often used through SSH on a remote server. You should confirm at the beginning of each session which HPC we are running on.

- Before running long jobs, show the full command. Always seek user approval to run long jobs.
- Before downloading large datasets, show destination path and expected size. Note: No data should need to be downloaded.
- Do not kill running training processes unless explicitly asked.
- Do not move or delete checkpoint directories without approval.
- Prefer tmux/screen-friendly commands for long tasks.
- Assume GPU time is valuable.