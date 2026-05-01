import sys
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

try:
    from src.dataset import LANLMJODataset
    # Provide root_dir explicitly to avoid the default warning
    # and restrict to a single year to speed up the lazy loading test.
    dataset = LANLMJODataset(
        start_year=2000, 
        end_year=2000, 
        root_dir="/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"
    )
    print("Dataset initialized successfully.")
    print("Static Vars Z shape:", dataset.static_vars['z'].shape)
    print("Static Vars LSM shape:", dataset.static_vars['lsm'].shape)
    print("Number of samples:", len(dataset))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
