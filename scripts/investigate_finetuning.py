# scripts/investigate_finetuning.py

### Imports ###
import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
# NOTE: We do NOT import autocast here, we let the model handle it.
import numpy as np
from aurora import AuroraSmallPretrained, Batch, Metadata
from itertools import chain
from datetime import datetime
import sys
import traceback

### DEBUG ###

print("started script...")

### Configuration ###
DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_jan2015_daily")
STATIC_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_static_data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"experiment_outputs/{TIMESTAMP}") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.txt"
MODEL_CHECKPOINT_PATH = OUTPUT_DIR / "finetuned_aurora_small.pt"

### Hardcoded Stats ###
TTR_MEAN_VAL = -225.89430236816406
TTR_STD_VAL = 45.86821746826172
TCWV_MEAN_VAL = 17.214859008789062
TCWV_STD_VAL = 16.590120315551758

### Training Config ###
NUM_EPOCHS = 3
LR = 1e-5
# BATCH_SIZE = 1 # Handled by DataLoader
TRAIN_SPLIT_DAY = 21

### Model / Loss Config ###
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TROPICS_LAT_BBOX = [-20, 20]

# We override print to ensure it hits the log file immediately
def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    print(msg, flush=True) # Flush to Slurm out
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

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
    log_print(f"Lazy loading {len(file_list)} files using engine='{engine}'...")
    # REMOVED: chunks={'valid_time': 1}. Let xarray decide the chunks based on the files.
    ds = xr.open_mfdataset(
        file_list, combine='by_coords', engine=engine, 
        parallel=False
    )
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    ds = ds.sortby("time")
    
    if 'latitude' in ds.dims and ds.dims['latitude'] == 721:
        log_print("Slicing latitude from 721 to 720 points...")
        ds = ds.isel(latitude=slice(0, 720))
    return ds

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds):
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds

        # Load and Slice Static Data
        static_ds = xr.open_dataset(STATIC_DIR / 'static_data.nc', engine='netcdf4')
        if 'latitude' in static_ds.dims and static_ds.dims['latitude'] == 721:
            static_ds = static_ds.isel(latitude=slice(0, 720))

        def get_static(var):
            # Handle potential expver dimension or time dimension
            if 'time' in static_ds.dims:
                arr = static_ds[var].isel(time=0).values
            else:
                arr = static_ds[var].values
            # clean and squeeze
            return torch.nan_to_num(torch.from_numpy(arr).float()).squeeze()

        # Explicitly constructing the dictionary to ensure clean data
        self.static_vars = {
            "lsm": get_static("lsm"),
            "z": get_static("z"),
            "slt": get_static("slt"),
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

        def clean(arr):
            # Ensure float32 for Encoder stability
            return torch.nan_to_num(torch.from_numpy(arr.astype(np.float32)))

        def scale_olr(arr):
            return clean(arr) / 3600.0

        in_batch = Batch(
            surf_vars={
                '2t': clean(surf_data['t2m'].values)[None],
                '10u': clean(surf_data['u10'].values)[None],
                '10v': clean(surf_data['v10'].values)[None],
                'msl': clean(surf_data['msl'].values)[None],
                'ttr': scale_olr(surf_data['ttr'].values)[None],
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
            'ttr': scale_olr(surf_target['ttr'].values),
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
    log_print(f'Experiment ID: {TIMESTAMP}')
    log_print(f'Using device: {DEVICE}')

    torch.backends.cudnn.benchmark = False

    # ... Data Loading Code (Same as before) ...
    log_print(f'\nLoading data from {DATA_DIR}...')
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


    # Stats calculation
    # ttr_mean = float(train_surface_ds['ttr'].mean()) / 3600.0 
    # ttr_std = float(train_surface_ds['ttr'].std()) / 3600.0
    # tcwv_mean = float(train_surface_ds['tcwv'].mean())
    # tcwv_std = float(train_surface_ds['tcwv'].std())
    # log_print(f"ttr mean: {ttr_mean}, ttr std: {ttr_std} (Watts/m2)")

    log_print(f"Using pre-computed stats:")
    log_print(f"ttr mean: {TTR_MEAN_VAL}, ttr std: {TTR_STD_VAL}")
    log_print(f"tcwv mean: {TCWV_MEAN_VAL}, tcwv std: {TCWV_STD_VAL}")

    log_print('Loading pre-trained Aurora Small model...')
    ext_surf_vars = ('2t', '10u', '10v', 'msl', 'ttr', 'tcwv')
    
    # CRITICAL FIX: Enable autocast INSIDE the model, not via context manager wrapper
    model = AuroraSmallPretrained(surf_vars=ext_surf_vars, autocast=True) 
    model.load_checkpoint(strict=False)

    from aurora.normalisation import locations, scales
    locations['ttr'] = TTR_MEAN_VAL
    scales['ttr'] = TTR_STD_VAL
    locations['tcwv'] = TCWV_MEAN_VAL
    scales['tcwv'] = TCWV_STD_VAL

    model.to(DEVICE)
    model.configure_activation_checkpointing()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss(reduction='none')
    lat_coords = torch.from_numpy(surface_ds.latitude.values).to(DEVICE)
    loss_weights = tropical_loss_weights(lat_coords)

    log_print(f'\nStarting fine-tuning for {NUM_EPOCHS} epochs...')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for i, (inp, target) in enumerate(train_loader):
            inp = inp.to(DEVICE)
            target = {k: v.to(DEVICE) for k, v in target.items()}

            optimizer.zero_grad(set_to_none=True)

            # Aurora handles mixed precision internally
            pred = model(inp)
            
            batch_loss = 0.0
            n_vars = 0
            for var_name, pred_tens in chain(pred.surf_vars.items(), pred.atmos_vars.items()):
                target_tens = target[var_name]
                
                # --- VERIFY THIS BLOCK IS PRESENT ---
                # Squeeze the history/time dimension from (B, 1, H, W) to (B, H, W)
                # or (B, 1, Lev, H, W) to (B, Lev, H, W)
                if pred_tens.shape[1] == 1 and pred_tens.ndim == target_tens.ndim + 1:
                     pred_tens = pred_tens.squeeze(1)
                # ------------------------------------
                
                weighted_loss = loss_fn(pred_tens, target_tens) * loss_weights
                batch_loss += weighted_loss.mean()
                n_vars += 1
            batch_loss /= n_vars

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += batch_loss.item()
            
            if i % 10 == 0:
                log_print(f"  Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {batch_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)

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
                    # --- FIX ADDED HERE ---
                    if pred_tens.shape[1] == 1 and pred_tens.ndim == target_tens.ndim + 1:
                         pred_tens = pred_tens.squeeze(1)
                    weighted_loss = loss_fn(pred_tens, target[var_name]) * loss_weights
                    batch_loss += weighted_loss.mean()
                    n_vars += 1
                batch_loss /= n_vars
                total_val_loss += batch_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        log_print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        epoch_ckpt_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_ckpt_path)

    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    log_print(f'Model saved to {MODEL_CHECKPOINT_PATH}')
    log_print('Fine-tuning complete.')

if __name__ == "__main__":
    try:
        log_print("Started main...")
        main()
    except Exception as e:
        log_print(traceback.format_exc())
        raise e