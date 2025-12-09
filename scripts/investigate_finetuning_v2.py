# scripts/investigate_finetuning_v2.py

# scripts/investigate_finetuning_v2.py

import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
from aurora import AuroraSmallPretrained, Batch, Metadata
from itertools import chain
from datetime import datetime
import sys
import traceback
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

# --- 1. CONFIGURATION ---
warnings.filterwarnings("ignore")

DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
STATIC_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_static_data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a DISTINCT output folder for Run 2
OUTPUT_DIR = Path(f"experiment_outputs/{TIMESTAMP}_RUN2") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.txt"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
METRICS_FILE = OUTPUT_DIR / "metrics.csv"
LOSS_PLOT_FILE = OUTPUT_DIR / "loss_curve.png"

# --- RUN 2 HYPERPARAMETERS (STABLE & SMOOTH) ---
NUM_EPOCHS = 30          
LR_MAX = 5e-5            # Slightly lower max LR for stability
WEIGHT_DECAY = 0.02      # Slightly higher decay to prevent overfitting
FFT_WEIGHT = 0.05        # Reduced from 0.1 to reduce "gritty" artifacts
GRAD_ACCUM_STEPS = 4     # Simulates Batch Size = 4 (Smoother gradients)
TRAIN_SPLIT_DAY = 21     # Day of month to split train/val

# Stats (Precomputed)
TTR_MEAN_VAL = -225.89430236816406
TTR_STD_VAL = 45.86821746826172
TCWV_MEAN_VAL = 17.214859008789062
TCWV_STD_VAL = 16.590120315551758

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TROPICS_LAT_BBOX = [-20, 20]

def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# --- 2. HELPERS ---
def spectral_loss(input, target):
    """Calculates L1 loss in the frequency domain."""
    input_fft = torch.fft.rfft2(input, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    loss = torch.abs(input_fft - target_fft).mean()
    return loss

def save_loss_plot(history):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='x')
    plt.title(f'Run 2 Loss (Best Val: {df["val_loss"].min():.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(LOSS_PLOT_FILE, dpi=150)
    plt.close()

def save_comparison_plot(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        inp, target = next(iter(loader))
        inp = inp.to(DEVICE)
        pred = model(inp)
        
        pred_ttr = pred.surf_vars['ttr'].squeeze().cpu().numpy()
        target_ttr = target['ttr'].squeeze().cpu().numpy()
        
        diff = pred_ttr - target_ttr
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        im1 = axes[0].imshow(target_ttr, cmap='RdBu_r', vmin=-350, vmax=-100)
        axes[0].set_title(f"Ground Truth OLR (Watts/m^2)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(pred_ttr, cmap='RdBu_r', vmin=-350, vmax=-100)
        axes[1].set_title(f"Aurora Prediction (Epoch {epoch})")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        im3 = axes[2].imshow(diff, cmap='seismic', vmin=-100, vmax=100)
        axes[2].set_title(f"Difference (Pred - GT)")
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparison_epoch_{epoch}.png", dpi=150)
        plt.close()
        log_print(f"Saved visualization to comparison_epoch_{epoch}.png")

def load_and_combine_files(file_list, engine='netcdf4'):
    log_print(f"Lazy loading {len(file_list)} files...")
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
        self.num_samples = len(self.surface_ds['time']) - 2
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.lon = torch.from_numpy(surface_ds['longitude'].values)
        self.atmos_levels = tuple(int(l) for l in pressure_ds['pressure_level'].values)

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        input_slice = slice(idx, idx + 2)
        target_slice = slice(idx + 2, idx + 3)
        surf_data = self.surface_ds.isel(time=input_slice).load()
        surf_target = self.surface_ds.isel(time=target_slice).load()
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        pres_target = self.pressure_ds.isel(time=target_slice).load()

        def clean(arr): return torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))
        def scale_olr(arr): return clean(arr) / 3600.0

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
        target_dict = {
            '2t': clean(surf_target['t2m'].values), '10u': clean(surf_target['u10'].values),
            '10v': clean(surf_target['v10'].values), 'msl': clean(surf_target['msl'].values),
            'ttr': scale_olr(surf_target['ttr'].values), 'tcwv': clean(surf_target['tcwv'].values),
            'z': clean(pres_target['z'].values), 'u': clean(pres_target['u'].values),
            'v': clean(pres_target['v'].values), 't': clean(pres_target['t'].values),
            'q': clean(pres_target['q'].values),
        }
        return in_batch, target_dict

def collate_fn(batch_list): return batch_list[0]

def tropical_loss_weights(lat_coords):
    weights = torch.ones_like(lat_coords)
    tropical_mask = (lat_coords >= TROPICS_LAT_BBOX[0]) & (lat_coords <= TROPICS_LAT_BBOX[1])
    weights[~tropical_mask] = 0.1
    return weights.view(1, 1, -1, 1)

# --- 3. MAIN LOOP ---
def main():
    log_print(f'Experiment ID: {TIMESTAMP} (RUN 2 - STABLE)')
    log_print(f'Using device: {DEVICE}')
    torch.backends.cudnn.benchmark = False

    log_print('Loading data...')
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    pressure_files = sorted(DATA_DIR.glob("pressure_*.nc"))
    surface_ds = load_and_combine_files(surface_files)
    pressure_ds = load_and_combine_files(pressure_files)

    train_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day >= TRAIN_SPLIT_DAY)
    train_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day >= TRAIN_SPLIT_DAY)

    train_loader = DataLoader(MJODataset(train_surface_ds, train_pressure_ds), batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MJODataset(val_surface_ds, val_pressure_ds), batch_size=1, shuffle=False, collate_fn=collate_fn)

    log_print("Loading pre-trained Aurora Small model...")
    ext_surf_vars = ('2t', '10u', '10v', 'msl', 'ttr', 'tcwv')
    model = AuroraSmallPretrained(surf_vars=ext_surf_vars, autocast=True) 
    model.load_checkpoint(strict=False)

    from aurora.normalisation import locations, scales
    locations['ttr'], scales['ttr'] = TTR_MEAN_VAL, TTR_STD_VAL
    locations['tcwv'], scales['tcwv'] = TCWV_MEAN_VAL, TCWV_STD_VAL

    model.to(DEVICE)
    model.configure_activation_checkpointing()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    
    # CHANGE: OneCycleLR for smooth convergence
    total_steps = NUM_EPOCHS * len(train_loader)
    # We must account for Gradient Accumulation in the total steps for the scheduler!
    # Because scheduler.step() is called every batch, but optimizer.step() only every 4th.
    # WAIT: Standard implementation steps scheduler every optimizer step.
    # But often people step scheduler every batch. Let's step it every batch to keep the curve smooth.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR_MAX, 
        total_steps=total_steps, 
        pct_start=0.3
    )
    
    l1_loss_fn = torch.nn.L1Loss(reduction='none')
    lat_coords = torch.from_numpy(surface_ds.latitude.values).to(DEVICE)
    loss_weights = tropical_loss_weights(lat_coords)

    log_print(f'\nStarting fine-tuning for {NUM_EPOCHS} epochs with Grad Accumulation={GRAD_ACCUM_STEPS}...')
    best_val_loss = float('inf')
    history = []

    optimizer.zero_grad(set_to_none=True) # Init zero grad

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for i, (inp, target) in enumerate(train_loader):
            inp = inp.to(DEVICE)
            target = {k: v.to(DEVICE) for k, v in target.items()}

            pred = model(inp)
            batch_loss = 0.0
            n_vars = 0
            
            for var_name, pred_tens in chain(pred.surf_vars.items(), pred.atmos_vars.items()):
                target_tens = target[var_name]
                if pred_tens.shape[1] == 1 and pred_tens.ndim == target_tens.ndim + 1:
                     pred_tens = pred_tens.squeeze(1)
                
                l1 = (l1_loss_fn(pred_tens, target_tens) * loss_weights).mean()
                fft = spectral_loss(pred_tens, target_tens)
                batch_loss += l1 + (FFT_WEIGHT * fft)
                n_vars += 1
            
            batch_loss /= n_vars
            
            # --- GRADIENT ACCUMULATION LOGIC ---
            # Normalize loss by accumulation steps so the gradients are averaged, not summed
            loss_to_backward = batch_loss / GRAD_ACCUM_STEPS
            loss_to_backward.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Step scheduler every batch for smooth LR curve
            scheduler.step()
            
            total_train_loss += batch_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(DEVICE)
                target = {k: v.to(DEVICE) for k, v in target.items()}
                pred = model(inp)
                batch_loss = 0.0
                n_vars = 0
                for var_name, pred_tens in chain(pred.surf_vars.items(), pred.atmos_vars.items()):
                    target_tens = target[var_name]
                    if pred_tens.shape[1] == 1 and pred_tens.ndim == target_tens.ndim + 1:
                         pred_tens = pred_tens.squeeze(1)
                    l1 = (l1_loss_fn(pred_tens, target_tens) * loss_weights).mean()
                    batch_loss += l1
                    n_vars += 1
                batch_loss /= n_vars
                total_val_loss += batch_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        log_print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        })
        pd.DataFrame(history).to_csv(METRICS_FILE, index=False)
        save_loss_plot(history)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            save_comparison_plot(model, val_loader, epoch+1)

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    log_print('Fine-tuning complete.')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open(LOG_FILE, "a") as f: f.write(traceback.format_exc() + "\n")
        print(traceback.format_exc())
        raise e