# scripts/investigate_finetuning.py

### Imports ###

import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from aurora import AuroraSmallPretrained, Batch, Metadata

### Configuration ###

DATA_DIR = Path("/gpfs/gibbs/project/lu_lu/kam352/era5_sample_jan2015")
SURFACE_DIR = DATA_DIR / "surface_jan2015.nc"
PRESSURE_DIR = DATA_DIR / "pressure_jan2015.nc"

### Training Config ###

NUM_EPOCHS = 3 # initially 3 
LR = 1e-5 # initially 1e-5
BATCH_SIZE = 1 # initially 1
TRAIN_SPLIT_DAY = 21 # initially 21

### Model / Loss Config ###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# so we weight the loss more heavily in the tropics
TROPICS_LAT_BBOX = [-20, 20]

### Main Script Functions ###

### Creating the dataset ###

def create_ds(surface_ds, pressure_ds):
    """
    Converts xarray datasets to torch tensors and creates a DataLoader.
    Prepares the data for training and validation splits.
    """

    # Note that Aurora needs two time steps, (t-1, t), to predict the next time, (t+1).
    # so the input will be a pair of time steps, and the target will be the next time step.

    surface_vars = {
        't2m': surface_ds['t2m'].values,
        'u10': surface_ds['u10'].values,
        'v10': surface_ds['v10'].values,
        'msl': surface_ds['msl'].values,
        'ttr': surface_ds['ttr'].values,
        'tcwv': surface_ds['tcwv'].values,
    }

    pressure_vars = {
        'z': pressure_ds['z'].values,
        'u': pressure_ds['u'].values,
        'v': pressure_ds['v'].values,
        't': pressure_ds['t'].values,
        'q': pressure_ds['q'].values,
    }

    # begin the index at 2 because we can't predict those
    # obviously because we don't have t-1 and t0 for them

    num_samples = len(surface_ds['time']) - 2
    all_inputs = []
    all_targets = []

    for i in range(num_samples):
        input_slice = slice(i, i+2)
        target_slice = slice(i+2, i+3)

        # we now create the Batch object directly

        in_batch = Batch(
            surf_vars={
                k: torch.from_numpy(v[input_slice][None]).float() for k, v in surface_vars.items()
            },
            atmos_vars={
                k: torch.from_numpy(v[input_slice][None]).float() for k, v in pressure_vars.items()
            },
            metadata=Metadata(
                lat=torch.from_numpy(surface_ds['latitude'].values[None]).float(),
                lon=torch.from_numpy(surface_ds['longitude'].values[None]).float(),
                time=(surface_ds.valid_time.values.astype('datetime64[s]').tolist()[i+1],),
                atmos_levels=tuple(int(level) for level in pressure_ds['isobaricInhPa'].values),
            )
        )

        target_dict = {
            **{
                k: torch.from_numpy(v[target_slice][None]).float() for k, v in surface_vars.items()
            },
            **{
                k: torch.from_numpy(v[target_slice][None]).float() for k, v in pressure_vars.items()
            }
        }

        all_inputs.append(in_batch)
        all_targets.append(target_dict)

    return all_inputs, all_targets

### Loss function mask ###

# the goal here is to weight the loss more heavily in the tropics

def tropical_loss_weights(lat_coords):
    """
    Create a weight mask for the loss function that emphasises the tropics.
    """

    weights = torch.ones_like(lat_coords)
    tropical_mask = (lat_coords >= TROPICS_LAT_BBOX[0]) & (lat_coords <= TROPICS_LAT_BBOX[1])
    weights[~tropical_mask] = .1
    return weights.view(1,1,-1,1)  # reshape for broadcasting

### Main Script ###

def main():
    print('=== Starting Investigation of microFine-Tuning Aurora ===')
    print(f'Using device: {DEVICE}')

    ### Load Data ###

    print(f'Loading data from {SURFACE_DIR} and {PRESSURE_DIR}...')

    surface_ds = xr.open_dataset(SURFACE_DIR)
    pressure_ds = xr.open_dataset(PRESSURE_DIR)

    ### Splitting...

    print('Creating training and validation datasets...')

    train_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_surface_ds = surface_ds.sel(time=surface_ds.time.dt.day >= TRAIN_SPLIT_DAY)

    train_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day < TRAIN_SPLIT_DAY)
    val_pressure_ds = pressure_ds.sel(time=pressure_ds.time.dt.day >= TRAIN_SPLIT_DAY)

    print(f'Creating training dataset for days 1--{TRAIN_SPLIT_DAY - 1}...')
    train_inputs, train_targets = create_ds(train_surface_ds, train_pressure_ds)

    print(f'Creating validation dataset for days {TRAIN_SPLIT_DAY}--end...')
    val_inputs, val_targets = create_ds(val_surface_ds, val_pressure_ds)

    ### Loading Model and Optimiser

    print('Loading pre-trained Aurora Small model...')
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss(reduction='none')

    lat_coords = torch.from_numpy(surface_ds.latitude.values).to(DEVICE)
    loss_weights = tropical_loss_weights(lat_coords)

    ### Training Loop

    print('\nStarting fine-tuning...')

    for epoch in range(NUM_EPOCHS):
        ### Put into training
        model.train()
        total_train_loss = .0

        for i in range(len(train_inputs)):
            inp = train_inputs[i].to(DEVICE)
            target = {k: v.to(DEVICE) for k, v in train_targets[i].items()}

            optimizer.zero_grad()
            
            pred = model(inp)

            batch_loss = .0
            n_vars = 0

            for var_name, pred_tens in pred.items():
                target_tens = target[var_name]
                weighted_loss = loss_fn(pred_tens, target_tens) * loss_weights
                batch_loss += weighted_loss.mean()
                n_vars += 1
            
            batch_loss /= n_vars

            batch_loss.backwards()
            optimizer.step()

            total_train_loss += batch_loss.item()
        
        avg_train_loss = total_train_loss / len(train_inputs)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss:.4f}')

        ### Validation Loop

        model.eval()
        total_val_loss = .0

        with torch.no_grad():
            for i in range(len(val_inputs)):
                inp = val_inputs[i].to(DEVICE)
                target = {k: v.to(DEVICE) for k, v in val_targets[i].items()}

                pred = model(inp)

                batch_loss = .0
                n_vars = 0

                for var_name, pred_tens in pred.items():
                    target_tens = target[var_name]
                    weighted_loss = loss_fn(pred_tens, target_tens) * loss_weights
                    batch_loss += weighted_loss.mean()
                    n_vars += 1
                
                batch_loss /= n_vars

                total_val_loss += batch_loss.item()

        avg_val_loss = total_val_loss / len(val_inputs)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    print('Fine-tuning complete.')

if __name__ == "__main__":
    main()