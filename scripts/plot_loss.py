
import re
import matplotlib.pyplot as plt
import pandas as pd

import os
log_path = 'slurm_logs/train_52429909.out'
output_dir = 'evaluation/plots'
os.makedirs(output_dir, exist_ok=True)
output_plot = os.path.join(output_dir, 'loss_curve_52429909.pdf')

steps = []
losses = []

# Pattern to match the loss lines
# 2026-05-04 14:17:01,182 | INFO | src.trainer | Epoch 001 | step 00000/54058 | rollout_k=1 | loss=25.2207 (grid=25.2207, spec=0.0000, mjo=0.0000, phys=0.0000) | lr=1.00e-06
pattern = re.compile(r'step (\d+)/\d+ .* loss=([\d\.]+)')

with open(log_path, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))

if not steps:
    print("No loss data found in log.")
else:
    df = pd.DataFrame({'step': steps, 'loss': losses})
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], label='Training Loss (Total)')
    
    # Add a rolling average to smooth the curve
    if len(df) > 50:
        df['smoothed'] = df['loss'].rolling(window=50).mean()
        plt.plot(df['step'], df['smoothed'], label='50-step Moving Average', linewidth=2)
    
    plt.title('Aurora MJO Phase 1 Baseline - Training Loss (Job 52429909)')
    plt.xlabel('Step')
    plt.ylabel('Loss (Tropical-Weighted MAE)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add annotation for start and end
    plt.annotate(f'Start: {losses[0]:.2f}', xy=(steps[0], losses[0]), xytext=(10, 10), 
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'End: {losses[-1]:.2f}', xy=(steps[-1], losses[-1]), xytext=(10, 10), 
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
