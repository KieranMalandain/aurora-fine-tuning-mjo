import pathlib
import xarray as xr

def explore_data():
    base_path = pathlib.Path("/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results")

    targets = [
        ("z_invariant", ["Step00/ERA5.invariant"], "*z*.nc"),
        ("lsm_invariant", ["Step00/ERA5.invariant"], "*lsm*.nc"),
        ("z_atmos", ["gopt"], "*.nc"),
        ("q_atmos", ["sphu"], "*.nc"),
        ("t_atmos", ["t", "tprt"], "*.nc"), 
        ("u_atmos", ["uWnd"], "*.nc"),
        ("v_atmos", ["vWnd"], "*.nc"),
        ("2t_surface", ["T2"], "*.nc"),
        ("10u_surface", ["U10"], "*.nc"),
        ("10v_surface", ["V10"], "*.nc"),
        ("msl_surface", ["PS", "P"], "*.nc"),
        ("ttr_surface", ["meanTNLWFLX"], "*.nc"),
        ("tcwv_surface", ["tcwv"], "*.nc"),
    ]

    report = []

    for name, dir_hints, pattern in targets:
        print(f"\n{'='*60}\nSearching for {name}\n{'-'*60}")
        
        found_file = None
        best_dir = None
        
        for hint in dir_hints:
            if "/" in hint:
                target_dir = base_path / hint
                if target_dir.exists():
                    try:
                        found_file = next(f for f in target_dir.rglob(pattern) if f.is_file() and not f.name.startswith('.'))
                        best_dir = target_dir
                        break
                    except StopIteration:
                        pass
            else:
                potential_dirs = list(base_path.glob(f"Step*/*/{hint}"))
                if not potential_dirs:
                    potential_dirs = list(base_path.glob(f"Step*/{hint}"))
                
                for p_dir in potential_dirs:
                    if p_dir.is_dir():
                        try:
                            found_file = next(f for f in p_dir.rglob(pattern) if f.is_file() and not f.name.startswith('.'))
                            best_dir = p_dir
                            break
                        except StopIteration:
                            continue
                if found_file:
                    break
        
        if not found_file:
            print(f"[MISSING] Could not find any files for {name} using hints {dir_hints} and pattern {pattern}")
            report.append((name, "NOT_FOUND", None, None))
            continue
            
        print(f"Found directory: {best_dir.relative_to(base_path)}")
        print(f"Sample file: {found_file.name}")
        
        try:
            ds = xr.open_dataset(found_file, engine='netcdf4')
            coords = list(ds.coords.keys())
            data_vars = list(ds.data_vars.keys())
            dims = dict(ds.sizes)
            
            print(f"Dimensions: {dims}")
            print(f"Coordinates: {coords}")
            print(f"Data Variables: {data_vars}")
            
            if data_vars:
                for v in data_vars:
                    dtype = ds[v].dtype
                    print(f"  Variable '{v}': dtype={dtype}")
                    plev_coords = [c for c in ['lev', 'isobaricInhPa', 'level'] if c in ds.coords]
                    if plev_coords and plev_coords[0] in ds[v].dims:
                        levels = ds[plev_coords[0]].values
                        print(f"  Pressure levels ({len(levels)}): {levels[:5]} ... {levels[-5:]}")
            else:
                print("No data variables found!")
                
            report.append((name, "FOUND", best_dir.relative_to(base_path), data_vars))
            ds.close()
        except Exception as e:
            print(f"Error reading file {found_file}: {e}")
            report.append((name, "ERROR", best_dir.relative_to(base_path), None))
            
    print("\n\n" + "="*110)
    print(f"{'SUMMARY REPORT':^110}")
    print("="*110)
    print(f"{'Variable':<15} | {'Status':<9} | {'Resolved Directory Path':<50} | {'Native NetCDF Vars'}")
    print("-" * 110)
    for name, status, rel_dir, vars_found in report:
        if status == "FOUND":
            v_str = ", ".join(vars_found) if vars_found else "None"
            print(f"{name:<15} | {status:<9} | {str(rel_dir):<50} | {v_str}")
        else:
            print(f"{name:<15} | {status:<9} | {'N/A':<50} | N/A")

if __name__ == "__main__":
    explore_data()
