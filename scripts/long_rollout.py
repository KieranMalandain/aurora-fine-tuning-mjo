import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
from aurora import AuroraSmallPretrained, Batch, Metadata
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import dataclasses

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

# Update this path to match your actual output folder!
RUN_DIR = Path("/home/kam352/aurora-fine-tuning-mjo/experiment_outputs/20251130_231354_RUN2")
MODEL_PATH = RUN_DIR / "best_model.pt"
OUTPUT_DIR = RUN_DIR / "rollout_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
STATIC_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_static_data")

# Stats (Run 2)
TTR_MEAN_VAL = -225.89
TTR_STD_VAL = 45.87
TCWV_MEAN_VAL = 17.21
TCWV_STD_VAL = 16.59

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROLLOUT_STEPS = 40  # 10 Days (4 steps per day)

def load_and_combine_files(file_list, engine='netcdf4'):
    ds = xr.open_mfdataset(file_list, combine='by_coords', engine=engine, parallel=False)
    if 'valid_time' in ds.dims: ds = ds.rename({'valid_time': 'time'})
    ds = ds.sortby("time")
    if 'latitude' in ds.sizes and ds.sizes['latitude'] == 721:
        ds = ds.isel(latitude=slice(0, 720))
    return ds

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds):
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds
        static_ds = xr.open_dataset(STATIC_DIR / 'static_data.nc', engine='netcdf4')
        if 'latitude' in static_ds.sizes and static_ds.sizes['latitude'] == 721:
            static_ds = static_ds.isel(latitude=slice(0, 720))

        def get_static(var):
            if 'time' in static_ds.dims: arr = static_ds[var].isel(time=0).values
            else: arr = static_ds[var].values
            return torch.nan_to_num(torch.from_numpy(arr).float()).squeeze()

        self.static_vars = {"lsm": get_static("lsm"), "z": get_static("z"), "slt": get_static("slt")}
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.lon = torch.from_numpy(surface_ds['longitude'].values)
        self.atmos_levels = tuple(int(l) for l in pressure_ds['pressure_level'].values)

    def __len__(self):
        # We need enough future steps for ground truth comparison
        return len(self.surface_ds['time']) - ROLLOUT_STEPS - 2

    def __getitem__(self, idx):
        # Initial Input: 2 steps
        start_idx = idx
        input_slice = slice(start_idx, start_idx + 2)
        
        # Ground Truth Rollout: Next 40 steps
        gt_slice = slice(start_idx + 2, start_idx + 2 + ROLLOUT_STEPS)
        
        surf_data = self.surface_ds.isel(time=input_slice).load()
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        
        gt_surf = self.surface_ds.isel(time=gt_slice).load()
        
        def clean(arr): return torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))
        def scale_olr(arr): return clean(arr) / 3600.0

        # Construct Initial Batch
        in_batch = Batch(
            surf_vars={
                '2t': clean(surf_data['t2m'].values)[None], '10u': clean(surf_data['u10'].values)[None],
                '10v': clean(surf_data['v10'].values)[None], 'msl': clean(surf_data['msl'].values)[None],
                'ttr': scale_olr(surf_data['ttr'].values)[None], 'tcwv': clean(surf_data['tcwv'].values)[None],
            },
            atmos_vars={
                'z': clean(pres_data['z'].values)[None], 'u': clean(pres_data['u'].values)[None],
                'v': clean(pres_data['v'].values)[None], 't': clean(pres_data['t'].values)[None],
                'q': clean(pres_data['q'].values)[None],
            },
            static_vars=self.static_vars,
            metadata=Metadata(lat=self.lat, lon=self.lon, time=(surf_data.time.values.astype('datetime64[s]').tolist()[1],), atmos_levels=self.atmos_levels)
        )
        
        # Ground Truth Dictionary (Just TTR/OLR for visual comparison)
        gt_dict = {
            'ttr': scale_olr(gt_surf['ttr'].values), # Shape (Steps, H, W)
            'time': gt_surf.time.values
        }

        return in_batch, gt_dict

def run_rollout():
    print(f"Loading data from {DATA_DIR}...")
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    pressure_files = sorted(DATA_DIR.glob("pressure_*.nc"))
    surface_ds = load_and_combine_files(surface_files)
    pressure_ds = load_and_combine_files(pressure_files)

    # Validation Split (Day 21+)
    val_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day >= 21)
    val_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day >= 21)
    
    val_dataset = MJODataset(val_surface_ds, val_pressure_ds)
    # Just grab the first sequence (starts at Day 21 00:00)
    start_batch, gt_data = val_dataset[0] 

    print("Loading model...")
    ext_surf_vars = ('2t', '10u', '10v', 'msl', 'ttr', 'tcwv')
    model = AuroraSmallPretrained(surf_vars=ext_surf_vars, autocast=True) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Inject Stats
    from aurora.normalisation import locations, scales
    locations['ttr'], scales['ttr'] = TTR_MEAN_VAL, TTR_STD_VAL
    locations['tcwv'], scales['tcwv'] = TCWV_MEAN_VAL, TCWV_STD_VAL

    model.to(DEVICE)
    model.eval()

    # --- THE ROLLOUT LOOP ---
    print(f"Starting {ROLLOUT_STEPS}-step rollout (10 days)...")
    
    current_batch = start_batch
    # Move to GPU
    current_batch = current_batch.to(DEVICE)
    
    predictions = []
    
    with torch.no_grad():
        for step in range(ROLLOUT_STEPS):
            # 1. Predict
            pred_batch = model(current_batch)
            
            # 2. Store Result (Extract TTR for analysis)
            # Pred shape: (1, 1, H, W) -> (H, W)
            ttr_pred = pred_batch.surf_vars['ttr'].squeeze().cpu().numpy()
            predictions.append(ttr_pred)
            
            # 3. Prepare Next Input (Autoregressive Step)
            new_surf_vars = {}
            for k, v in pred_batch.surf_vars.items():
                # current_batch.surf_vars[k] is (1, 2, H, W). We want [:, 1:] -> (1, 1, H, W)
                hist = current_batch.surf_vars[k][:, 1:]
                new_surf_vars[k] = torch.cat([hist, v], dim=1)
            
            new_atmos_vars = {}
            for k, v in pred_batch.atmos_vars.items():
                hist = current_batch.atmos_vars[k][:, 1:]
                new_atmos_vars[k] = torch.cat([hist, v], dim=1)
                
            # Create new batch
            current_batch = dataclasses.replace(
                pred_batch,
                surf_vars=new_surf_vars,
                atmos_vars=new_atmos_vars
            )
            
            if step % 4 == 0:
                print(f"  Step {step}/{ROLLOUT_STEPS} complete")

    # --- ANALYSIS ---
    predictions = np.array(predictions) # (Steps, H, W)
    
    # FIX: Convert Tensor to Numpy here
    gt_ttr = gt_data['ttr'].numpy()     # (Steps, H, W)
    
    # Calculate RMSE over time
    rmse_series = np.sqrt(np.mean((predictions - gt_ttr)**2, axis=(1, 2)))
    
    # Save Metrics
    df = pd.DataFrame({'step': range(1, ROLLOUT_STEPS+1), 'rmse': rmse_series})
    df.to_csv(OUTPUT_DIR / "rollout_metrics.csv", index=False)
    
    # Plot RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['rmse'], marker='o')
    plt.title("Aurora Fine-Tuned: 10-Day Forecast Error (TTR)")
    plt.xlabel("Lead Time (Steps of 6h)")
    plt.ylabel("RMSE (Watts/m^2)")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "rollout_rmse.png")
    plt.close()
    
    # Plot Visual Comparisons at Day 1, 5, 10
    display_steps = [3, 19, 39] # Indices for Day 1, 5, 10
    days = [1, 5, 10]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    for i, step_idx in enumerate(display_steps):
        # GT
        im1 = axes[i, 0].imshow(gt_ttr[step_idx], cmap='RdBu_r', vmin=-350, vmax=-100)
        axes[i, 0].set_title(f"Day {days[i]} Ground Truth")
        axes[i, 0].axis('off')
        
        # Pred
        im2 = axes[i, 1].imshow(predictions[step_idx], cmap='RdBu_r', vmin=-350, vmax=-100)
        axes[i, 1].set_title(f"Day {days[i]} Prediction")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_visuals.png")
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_rollout()