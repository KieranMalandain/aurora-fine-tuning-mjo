# scripts/investigate_finetuning.py

### Imports ###

import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import numpy as np
from aurora import AuroraSmallPretrained, Batch, Metadata
from itertools import chain
from datetime import datetime
import sys
import traceback

### Configuration ###

DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
STATIC_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_static_data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"experiment_outputs/{TIMESTAMP}") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.txt"
MODEL_CHECKPOINT_PATH = OUTPUT_DIR / "finetuned_aurora_small.pt"

### Training Config ###
NUM_EPOCHS = 3
LR = 1e-5
BATCH_SIZE = 1
TRAIN_SPLIT_DAY = 21

### Model / Loss Config ###
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TROPICS_LAT_BBOX = [-20, 20]

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_and_combine_files(file_list, engine='netcdf4'):
    print(f"Lazy loading {len(file_list)} files using engine='{engine}'...")
    try:
        ds = xr.open_mfdataset(
            file_list, combine='by_coords', engine=engine, 
            parallel=False, chunks={'valid_time': 1}
        )
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        ds = ds.sortby("time")
        
        # Fix Latitude Dimension
        if 'latitude' in ds.dims and ds.dims['latitude'] == 721:
            print("Slicing latitude from 721 to 720 points...")
            ds = ds.isel(latitude=slice(0, 720))
        return ds
    except Exception as e:
        print(f"Error opening dataset: {e}")
        raise e

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds):
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds

        # Load and Slice Static Data
        static_ds = xr.open_dataset(STATIC_DIR / 'static_data.nc', engine='netcdf4')
        if 'latitude' in static_ds.dims and static_ds.dims['latitude'] == 721:
            print("Slicing static latitude from 721 to 720 points...")
            static_ds = static_ds.isel(latitude=slice(0, 720))

        if 'time' in static_ds.dims:
            z_data = static_ds['z'].isel(time=0).values
            lsm_data = static_ds['lsm'].isel(time=0).values
            slt_data = static_ds['slt'].isel(time=0).values
        else:
            z_data = static_ds['z'].values
            lsm_data = static_ds['lsm'].values
            slt_data = static_ds['slt'].values

        # SAFETY FIX: Check for NaNs in static data
        self.static_vars = {
            "z": torch.nan_to_num(torch.from_numpy(z_data).float().squeeze()),
            "lsm": torch.nan_to_num(torch.from_numpy(lsm_data).float().squeeze()),
            "slt": torch.nan_to_num(torch.from_numpy(slt_data).float().squeeze()),
        }
        
        self.num_samples = len(self.surface_ds['time']) - 2
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.lon = torch.from_numpy(surface_ds['longitude'].values)
        self.atmos_levels = tuple(int(l) for l in pressure_ds['pressure_level'].values)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_slice = slice(idx, idx + 2)
        target_slice = slice(idx + 2, idx + 3)
        
        surf_data = self.surface_ds.isel(time=input_slice).load()
        surf_target = self.surface_ds.isel(time=target_slice).load()
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        pres_target = self.pressure_ds.isel(time=target_slice).load()

        # Helper to clean data (NaNs -> 0)
        def clean(arr):
            return torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))

        # Helper to scale OLR (Joules -> Watts)
        # ERA5 'ttr' is accumulated Joules over 3600s. We divide by 3600 to get Watts/m^2.
        # This keeps the numbers manageable.
        def scale_olr(arr):
            return clean(arr) / 3600.0

        in_batch = Batch(
            surf_vars={
                '2t': clean(surf_data['t2m'].values)[None],
                '10u': clean(surf_data['u10'].values)[None],
                '10v': clean(surf_data['v10'].values)[None],
                'msl': clean(surf_data['msl'].values)[None],
                'ttr': scale_olr(surf_data['ttr'].values)[None], # Scale OLR
                'tcwv': clean(surf_data['tcwv'].values)[None],
            },
            atmos_vars={
                'z': clean(pres_data['z'].values)[None],
                'u': clean(pres_data['u'].values)[None],
                'v': clean(pres_data['v'].values)[None],
                't': clean(pres_data['t'].values)[None],
                'q': clean(pres_data['q'].values)[None],
            },
            static_vars=self.static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(surf_data.time.values.astype('datetime64[s]').tolist()[1],),
                atmos_levels=self.atmos_levels,
            )
        )

        target_dict = {
            '2t': clean(surf_target['t2m'].values),
            '10u': clean(surf_target['u10'].values),
            '10v': clean(surf_target['v10'].values),
            'msl': clean(surf_target['msl'].values),
            'ttr': scale_olr(surf_target['ttr'].values), # Scale OLR
            'tcwv': clean(surf_target['tcwv'].values),
            'z': clean(pres_target['z'].values),
            'u': clean(pres_target['u'].values),
            'v': clean(pres_target['v'].values),
            't': clean(pres_target['t'].values),
            'q': clean(pres_target['q'].values),
        }

        return in_batch, target_dict

def collate_fn(batch_list):
    return batch_list[0]

def tropical_loss_weights(lat_coords):
    weights = torch.ones_like(lat_coords)
    tropical_mask = (lat_coords >= TROPICS_LAT_BBOX[0]) & (lat_coords <= TROPICS_LAT_BBOX[1])
    weights[~tropical_mask] = 0.1
    return weights.view(1, 1, -1, 1)

def main():
    sys.stdout = Logger(LOG_FILE)
    print(f'Experiment ID: {TIMESTAMP}')
    print(f'Using device: {DEVICE}')

    torch.backends.cudnn.benchmark = False

    print(f'\nLoading data from {DATA_DIR}...')
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    pressure_files = sorted(DATA_DIR.glob("pressure_*.nc"))
    surface_ds = load_and_combine_files(surface_files, engine='netcdf4')
    pressure_ds = load_and_combine_files(pressure_files, engine='netcdf4')

    train_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day >= TRAIN_SPLIT_DAY)
    train_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day >= TRAIN_SPLIT_DAY)

    train_dataset = MJODataset(train_surface_ds, train_pressure_ds)
    val_dataset = MJODataset(val_surface_ds, val_pressure_ds)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("Calculating normalisation statistics for the new variables (ttr, tcwv)")
    # Note: These should be scaled to Watts too if we scaled them in the loader
    # But here we are calculating from the raw ds. Let's adjust manually.
    ttr_mean = float(train_surface_ds['ttr'].mean()) / 3600.0 
    ttr_std = float(train_surface_ds['ttr'].std()) / 3600.0
    tcwv_mean = float(train_surface_ds['tcwv'].mean())
    tcwv_std = float(train_surface_ds['tcwv'].std())
    
    print(f"ttr mean: {ttr_mean}, ttr std: {ttr_std} (Watts/m2)")

    print('Loading pre-trained Aurora Small model...')
    ext_surf_vars = ('2t', '10u', '10v', 'msl', 'ttr', 'tcwv')
    
    # Removed autocast=True here to avoid double-wrapping conflicts
    model = AuroraSmallPretrained(surf_vars=ext_surf_vars) 
    model.load_checkpoint(strict=False)

    from aurora.normalisation import locations, scales
    locations['ttr'] = ttr_mean
    scales['ttr'] = ttr_std
    locations['tcwv'] = tcwv_mean
    scales['tcwv'] = tcwv_std

    model.to(DEVICE)
    print("Configuring activation checkpointing...")
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss(reduction='none')
    lat_coords = torch.from_numpy(surface_ds.latitude.values).to(DEVICE)
    loss_weights = tropical_loss_weights(lat_coords)

    print(f'\nStarting fine-tuning for {NUM_EPOCHS} epochs...')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for i, (inp, target) in enumerate(train_loader):
            inp = inp.to(DEVICE)
            target = {k: v.to(DEVICE) for k, v in target.items()}

            optimizer.zero_grad(set_to_none=True)

            # Use bfloat16 (Safest for A100)
            with autocast(dtype=torch.bfloat16):
                pred = model(inp)
                batch_loss = 0.0
                n_vars = 0
                for var_name, pred_tens in chain(pred.surf_vars.items(), pred.atmos_vars.items()):
                    target_tens = target[var_name]
                    weighted_loss = loss_fn(pred_tens, target_tens) * loss_weights
                    batch_loss += weighted_loss.mean()
                    n_vars += 1
                batch_loss /= n_vars

            # Standard Backward Pass (No Scaler)
            batch_loss.backward()
            
            # SAFETY FIX: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += batch_loss.item()
            torch.cuda.empty_cache()
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {batch_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(DEVICE)
                target = {k: v.to(DEVICE) for k, v in target.items()}
                with autocast(dtype=torch.bfloat16):
                    pred = model(inp)
                    batch_loss = 0.0
                    n_vars = 0
                    for var_name, pred_tens in chain(pred.surf_vars.items(), pred.atmos_vars.items()):
                        weighted_loss = loss_fn(pred_tens, target[var_name]) * loss_weights
                        batch_loss += weighted_loss.mean()
                        n_vars += 1
                    batch_loss /= n_vars
                    total_val_loss += batch_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        epoch_ckpt_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_ckpt_path)

    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    print(f'Model saved to {MODEL_CHECKPOINT_PATH}')
    print('Fine-tuning complete.')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        raise e