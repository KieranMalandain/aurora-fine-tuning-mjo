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

NUM_EPOCHS = 3 # init 3
LR = 1e-5 #init 1e-5
BATCH_SIZE = 1 # init 1
TRAIN_SPLIT_DAY = 21 # init 21

### Model / Loss Config ###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TROPICS_LAT_BBOX = [-20, 20]

### Logger Class ###
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

### Data Loading Helper ###
def load_and_combine_files(file_list, engine='netcdf4'):
    """
    Lazy loads files. DOES NOT call .load().
    """
    datasets = []
    print(f"Lazy loading {len(file_list)} files using engine='{engine}'...")
    
    # We open them, but we DO NOT load values into RAM yet.
    try:
        # 1. Open with 'valid_time' chunking
        ds = xr.open_mfdataset(
            file_list, 
            combine='by_coords', 
            engine=engine, 
            parallel=False, 
            chunks={'valid_time': 1} 
        )
        
        # 2. Rename to 'time' if needed
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
            
        # 3. Sort
        ds = ds.sortby("time")

        # slice if needed

        if 'latitude' in ds.dims and ds.dims['latitude'] == 721:
            print("Slicing latitude from 721 to 720 points...")
            ds = ds.isel(latitude=slice(0, 720))

        return ds
        
    except Exception as e:
        print(f"Error opening dataset: {e}")
        raise e

### The New Dataset Class (The Memory Fix) ###

class MJODataset(Dataset):
    def __init__(self, surface_ds, pressure_ds):
        """
        This class just holds the file handles. It does NOT load data.
        """
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds

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

        self.static_vars = {
            "z": torch.from_numpy(z_data).float().squeeze(),
            "lsm": torch.from_numpy(lsm_data).float().squeeze(),
            "slt": torch.from_numpy(slt_data).float().squeeze(),
        }
        
        # Pre-calculate length
        self.num_samples = len(self.surface_ds['time']) - 2
        
        # Pre-load coordinates into CPU memory once (they are small)
        self.lat = torch.from_numpy(surface_ds['latitude'].values)
        self.lon = torch.from_numpy(surface_ds['longitude'].values)
        self.atmos_levels = tuple(int(l) for l in pressure_ds['pressure_level'].values)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        This is where the work happens. This is called ONLY when the GPU needs a batch.
        It loads only 3 time steps of data into RAM.
        """
        # 1. Define slices
        # Input: time i and i+1
        input_slice = slice(idx, idx + 2)
        # Target: time i+2
        target_slice = slice(idx + 2, idx + 3)

        # 2. Extract data for this specific slice (Reads from disk here!)
        # We use .load() here on the tiny slice only.
        
        # Surface variables
        surf_data = self.surface_ds.isel(time=input_slice).load()
        surf_target = self.surface_ds.isel(time=target_slice).load()
        
        # Pressure variables
        pres_data = self.pressure_ds.isel(time=input_slice).load()
        pres_target = self.pressure_ds.isel(time=target_slice).load()

        # 3. Construct Input Batch
        in_batch = Batch(
            surf_vars={
                '2t': torch.from_numpy(surf_data['t2m'].values.astype(np.float32))[None],
                '10u': torch.from_numpy(surf_data['u10'].values.astype(np.float32))[None],
                '10v': torch.from_numpy(surf_data['v10'].values.astype(np.float32))[None],
                'msl': torch.from_numpy(surf_data['msl'].values.astype(np.float32))[None],
                'ttr': torch.from_numpy(surf_data['ttr'].values.astype(np.float32))[None],
                'tcwv': torch.from_numpy(surf_data['tcwv'].values.astype(np.float32))[None],
            },
            atmos_vars={
                'z': torch.from_numpy(pres_data['z'].values.astype(np.float32))[None],
                'u': torch.from_numpy(pres_data['u'].values.astype(np.float32))[None],
                'v': torch.from_numpy(pres_data['v'].values.astype(np.float32))[None],
                't': torch.from_numpy(pres_data['t'].values.astype(np.float32))[None],
                'q': torch.from_numpy(pres_data['q'].values.astype(np.float32))[None],
            },
            static_vars=self.static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(surf_data.time.values.astype('datetime64[s]').tolist()[1],),
                atmos_levels=self.atmos_levels,
            )
        )

        # 4. Construct Target Dictionary
        # Note: We squeeze the batch dim here because DataLoader adds it back
        target_dict = {
            '2t': torch.from_numpy(surf_target['t2m'].values.astype(np.float32)),
            '10u': torch.from_numpy(surf_target['u10'].values.astype(np.float32)),
            '10v': torch.from_numpy(surf_target['v10'].values.astype(np.float32)),
            'msl': torch.from_numpy(surf_target['msl'].values.astype(np.float32)),
            'ttr': torch.from_numpy(surf_target['ttr'].values.astype(np.float32)),
            'tcwv': torch.from_numpy(surf_target['tcwv'].values.astype(np.float32)),
            'z': torch.from_numpy(pres_target['z'].values.astype(np.float32)),
            'u': torch.from_numpy(pres_target['u'].values.astype(np.float32)),
            'v': torch.from_numpy(pres_target['v'].values.astype(np.float32)),
            't': torch.from_numpy(pres_target['t'].values.astype(np.float32)),
            'q': torch.from_numpy(pres_target['q'].values.astype(np.float32)),
        }

        return in_batch, target_dict

# Custom collate function to handle the Aurora Batch object
def collate_fn(batch_list):
    # Since batch size is 1, we just take the first element
    # If batch size > 1, we would need to stack them. 
    # For now, Aurora Batch objects are tricky to stack automatically, 
    # so we stick to BATCH_SIZE=1 for simplicity.
    return batch_list[0]

### Loss Mask ###
def tropical_loss_weights(lat_coords):
    weights = torch.ones_like(lat_coords)
    tropical_mask = (lat_coords >= TROPICS_LAT_BBOX[0]) & (lat_coords <= TROPICS_LAT_BBOX[1])
    weights[~tropical_mask] = 0.1
    return weights.view(1, 1, -1, 1)

### Main Script ###

def main():
    sys.stdout = Logger(LOG_FILE)
    print('=== Starting Investigation of microFine-Tuning Aurora ===')
    print(f'Experiment ID: {TIMESTAMP}')
    print(f'Using device: {DEVICE}')

    torch.backends.cudnn.benchmark = False  # safer for variable input sizes

    ### Load Data ###
    print(f'\nLoading data from {DATA_DIR}...')
    surface_files = sorted(DATA_DIR.glob("surface_*.nc"))
    pressure_files = sorted(DATA_DIR.glob("pressure_*.nc"))

    if not surface_files or not pressure_files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")

    surface_ds = load_and_combine_files(surface_files, engine='netcdf4')
    pressure_ds = load_and_combine_files(pressure_files, engine='netcdf4')
    print("Successfully initialized lazy datasets.")

    ### Splitting ###
    train_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day >= TRAIN_SPLIT_DAY)
    train_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day >= TRAIN_SPLIT_DAY)

    # Initialize Datasets
    print("Initializing PyTorch Datasets...")
    train_dataset = MJODataset(train_surface_ds, train_pressure_ds)
    val_dataset = MJODataset(val_surface_ds, val_pressure_ds)

    # Initialize DataLoaders
    # num_workers=0 is safer for debugging. Increase to 2 or 4 for speed later.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    ### Load Model ###
    print("Calculating normalisation statistics for the new variables (ttr, tcwv)")

    # first ttr (OLR)
    ttr_mean = float(train_surface_ds['ttr'].mean())
    ttr_std = float(train_surface_ds['ttr'].std())
    # second tcwv (total column water vapour)
    tcwv_mean = float(train_surface_ds['tcwv'].mean())
    tcwv_std = float(train_surface_ds['tcwv'].std())

    print(f"ttr mean: {ttr_mean}, ttr std: {ttr_std}")
    print(f"tcwv mean: {tcwv_mean}, tcwv std: {tcwv_std}")


    print('Loading pre-trained Aurora Small model...')
    ext_surf_vars = ('2t', '10u', '10v', 'msl', 'ttr', 'tcwv')
    model = AuroraSmallPretrained(surf_vars=ext_surf_vars, autocast=True)
    model.load_checkpoint(strict=False)

    from aurora.normalisation import locations, scales
    locations['ttr'] = ttr_mean
    scales['ttr'] = ttr_std
    locations['tcwv'] = tcwv_mean
    scales['tcwv'] = tcwv_std

    model.to(DEVICE)

    print("Configuring activation checkpointing...")
    model.configure_activation_checkpointing()

    # scaler = GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss(reduction='none')

    lat_coords = torch.from_numpy(surface_ds.latitude.values).to(DEVICE)
    loss_weights = tropical_loss_weights(lat_coords)

    ### Training Loop ###
    print(f'\nStarting fine-tuning for {NUM_EPOCHS} epochs...')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        # Iterate via the DataLoader
        for i, (inp, target) in enumerate(train_loader):
            # Move inputs to device (Batch object handles this internally if implemented, otherwise manual)
            # Since Aurora Batch object doesn't always have .to(), we might need to handle parts.
            # But normally model(inp) handles CPU inputs if the model is on GPU. 
            # Let's assume standard behavior for now.
            
            # Explicitly move target tensors to device
            target = {k: v.to(DEVICE) for k, v in target.items()}

            optimizer.zero_grad(set_to_none=True)

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

            batch_loss.backward()
            optimizer.step()

            # scaler.scale(batch_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            total_train_loss += batch_loss.item()

            torch.cuda.empty_cache()
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {batch_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)

        ### Validation Loop ###
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for inp, target in val_loader:
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

        epoch_ckpt_path = OUTPUT_DIR / f"checkppoint_epoch_{epoch+1}.pt"
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