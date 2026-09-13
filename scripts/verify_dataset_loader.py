import sys
import warnings

# Suppress xarray/netcdf warnings for clean output
warnings.filterwarnings("ignore")

from aurora_mjo.dataset import LANLMJODataset


def main():
    print("Initializing dataset for year 1980...")
    root_dir = "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"

    try:
        dataset = LANLMJODataset(start_year=1980, end_year=1980, root_dir=root_dir)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        sys.exit(1)

    print("\n--- Dataset Initialization Successful ---")
    print(f"Number of samples in year 1980: {len(dataset)}")

    # Check static variables
    print("\n--- Static Variables ---")
    static = dataset.static_vars
    for k, v in static.items():
        print(
            f"{k}: shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}"
        )

    print("\n--- Fetching first sample (index 0) ---")
    try:
        in_batch, surf_out, atmos_out = dataset[0]
    except Exception as e:
        print(f"Failed to fetch sample: {e}")
        sys.exit(1)

    print("\n--- Input Batch Details ---")
    print(f"Initialization Time: {in_batch.metadata.time[0]}")
    print(f"Latitude shape: {in_batch.metadata.lat.shape}")
    print(f"Longitude shape: {in_batch.metadata.lon.shape}")
    print(f"Atmos Levels: {in_batch.metadata.atmos_levels}")

    print("\n--- Input Surface Variables ---")
    for k, v in in_batch.surf_vars.items():
        print(f"  {k}: shape={v.shape}")

    print("\n--- Input Atmos Variables ---")
    for k, v in in_batch.atmos_vars.items():
        print(f"  {k}: shape={v.shape}")

    print("\n--- Output Surface Variables ---")
    surf_dict = surf_out[0] if isinstance(surf_out, list) else surf_out
    for k, v in surf_dict.items():
        print(f"  {k}: shape={v.shape}")

    print("\n--- Output Atmos Variables ---")
    atmos_dict = atmos_out[0] if isinstance(atmos_out, list) else atmos_out
    for k, v in atmos_dict.items():
        print(f"  {k}: shape={v.shape}")

    print("\nVerification completed successfully.")


if __name__ == "__main__":
    main()
