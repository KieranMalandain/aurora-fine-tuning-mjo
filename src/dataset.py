# src/dataset.py
#
# NERSC/LANL ERA5 dataset for Aurora MJO fine-tuning - v3, timestamp-aligned.
#
# WHAT CHANGED IN v3 AND WHY (read this before touching indexing code)
# ====================================================================
# The previous design built ONE global index map from the *reference* surface
# variable's file list and reused (file_idx, local_time_idx) pairs for every
# other variable - silently assuming all 11 variables have identical file
# chunking and identical time axes.  Verification against Perlmutter logs
# proved this assumption false in at least one way:
#
#   * The 1980–2015 "train" dataset reported 54,060 timesteps.  That is
#     EXACTLY 1980–2016 inclusive (13,515 days x 4).  The requested range is
#     52,596.  The glob was pulling one extra year - 2016, the first
#     VALIDATION year - into training (train/val leakage), and nothing
#     enforced the year range on what the glob returned.
#   * If per-variable file lists can differ by a whole year, they can differ
#     in chunking too, which is the suspected mechanism behind the 100%
#     non-finite validation losses (cross-variable timestep misalignment).
#
# v3 makes both failure modes STRUCTURALLY IMPOSSIBLE:
#
#   1. Every variable gets its OWN {timestamp -> (file_idx, local_idx)} map,
#      built from the actual `time` coordinates in its own files.
#   2. Timestamps outside [start_year, end_year] are dropped at build time
#      (hard year-range enforcement, independent of file naming/globbing).
#   3. The sample timeline is the SORTED INTERSECTION of all variables'
#      timestamps, and a sample is only valid if every timestep it needs
#      (t-6h, t, t+6h, ..., t+(k)*6h) exists in that intersection at exact
#      6-hour spacing — gaps in any variable simply remove those samples
#      instead of producing misaligned or fill-value reads.
#   4. __init__ prints a per-variable alignment report (files, timesteps,
#      dropped-out-of-range, coverage) so any future data drift is visible
#      in the first 90 seconds of every job instead of surfacing as NaN.
#
# Fork-safety (unchanged from the known-good version): only file paths and
# plain-Python index structures are stored on `self`; every __getitem__
# opens, reads, and closes NetCDF files fresh.  Safe for any num_workers.
#
# GPU upsampling (unchanged): data is returned at native 1deg (180x360); the
# trainer upsamples to 0.25def (720x1440) on GPU.  Static vars are upsampled
# here once.

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

from aurora import Batch, Metadata

_DEFAULT_NERSC_ROOT = "/global/cfs/cdirs/m4946/xiaoming/zm4946.MachLearn/PrcsPrep/prcs.ERA5/prcs.ERA5.Remap/Results"

AURORA_PLEVS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# {aurora_name: (step_subdir, glob_pattern, native_var_name)}
# NOTE — msl proxy: 'Ps' (surface pressure) stands in for 'msl' (missing from
# the LANL dataset).  Swap step_subdir + native name when real MSL lands.
SURFACE_VAR_MAP = {
    '2t':   ('Step02/ERA5.remap_180x360MODIS_6hrInst/T2',          '*', 't2'),
    '10u':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/U10',         '*', 'u10'),
    '10v':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/V10',         '*', 'v10'),
    'msl':  ('Step02/ERA5.remap_180x360MODIS_6hrInst/PS',          '*', 'ps'),
    'ttr':  ('Step03/ERA5.remap_180x360MODIS_6hrInst/meanTNLWFLX', '*', 'mtnlwrf'),
    'tcwv': ('Step02/ERA5.remap_180x360MODIS_6hrInst/tcwv',        '*', 'tcwv'),
}

ATMOS_VAR_MAP = {
    'z': ('Step01/ERA5.remap_180x360MODIS_6hrInst/gopt', '*', 'z'),
    'q': ('Step01/ERA5.remap_180x360MODIS_6hrInst/sphu', '*', 'q'),
    't': ('Step01/ERA5.remap_180x360MODIS_6hrInst/tprt', '*', 't'),
    'u': ('Step01/ERA5.remap_180x360MODIS_6hrInst/uWnd', '*', 'u'),
    'v': ('Step01/ERA5.remap_180x360MODIS_6hrInst/vWnd', '*', 'v'),
}

STEP_HOURS = 6  # dataset cadence; must match Aurora's 6 h step


def _upsample_to_aurora(tensor):
    """Upsample 1deg (180x360) to Aurora 0.25deg (720x1440). Used for statics only."""
    original_shape = tensor.shape
    if len(original_shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 3:
        tensor = tensor.unsqueeze(0)
    upsampled = F.interpolate(tensor, size=(720, 1440), mode='bilinear', align_corners=False)
    if len(original_shape) == 2:
        return upsampled.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        return upsampled.squeeze(0)
    return upsampled


def _to_seconds_i64(times) -> np.ndarray:
    """Normalize any datetime64/cftime-decodable time axis to int64 unix seconds.

    int64 seconds (not datetime64 objects) are the map keys: exact, hashable,
    cheap to store 54k x 11 of, and immune to ns-vs-s resolution mismatches.
    """
    arr = np.asarray(times)
    if not np.issubdtype(arr.dtype, np.datetime64):
        # cftime or object datetimes → go through pandas-free conversion
        arr = np.array([np.datetime64(t) for t in arr.tolist()])
    return arr.astype("datetime64[s]").astype(np.int64)


class LANLMJODataset(Dataset):
    """Fork-safe, timestamp-aligned ERA5 dataset for Aurora MJO fine-tuning."""

    native_resolution = (180, 360)

    def __init__(self, start_year: int, end_year: int,
                 root_dir: str | Path | None = None,
                 slt_path: str | Path | None = None,
                 max_rollout_steps: int = 1):
        if root_dir is None:
            warnings.warn(
                f"root_dir not provided; falling back to default NERSC path: "
                f"{_DEFAULT_NERSC_ROOT}.", stacklevel=2)
            root_dir = _DEFAULT_NERSC_ROOT
        self.root_dir = Path(root_dir)

        if slt_path is None:
            warnings.warn("slt_path not provided; falling back to default.", stacklevel=2)
            slt_path = "/pscratch/sd/k/kam352/Aurora/slt/slt_data.nc"
        self.slt_path = Path(slt_path)

        self.start_year = start_year
        self.end_year = end_year
        self.max_rollout_steps = max(1, max_rollout_steps)

        print(f"Initializing LANL MJO Dataset ({start_year}-{end_year}) [timestamp-aligned v3]...")

        # Aurora metadata coordinates are at the UPSAMPLED 0.25° resolution.
        self.lat = torch.linspace(90, -90, 720)
        self.lon = torch.linspace(0, 360, 1441)[:-1]
        self.atmos_levels = tuple(AURORA_PLEVS)

        # Hard year-range bounds in unix seconds — enforced on timestamps,
        # NOT on file names.  This is what closes the 2016-leakage hole.
        self._t_min = int(np.datetime64(f"{start_year}-01-01T00:00:00").astype("datetime64[s]").astype(np.int64))
        self._t_max = int(np.datetime64(f"{end_year}-12-31T23:59:59").astype("datetime64[s]").astype(np.int64))

        # 1. Static variables (small, loaded once, fork-safe tensors)
        self.static_vars = self._load_static_vars()

        # 2. Per-variable file lists
        self.surf_file_map = self._collect_var_files(SURFACE_VAR_MAP)
        self.atmos_file_map = self._collect_var_files(ATMOS_VAR_MAP)

        # 3. Per-variable {ts_seconds -> (file_idx, local_idx)} maps +
        #    intersection timeline + contiguity-checked sample starts.
        self._build_aligned_index()

        # 4. Pressure-level subset indices (29 LANL levels → 13 Aurora levels)
        self._plev_indices = self._probe_pressure_level_indices()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _collect_var_files(self, var_map: dict) -> dict:
        """Glob candidate files per variable.  Over-matching here is now
        harmless: the timestamp filter in _build_aligned_index() is what
        decides which timesteps are actually used."""
        result = {}
        for aurora_name, (step_subdir, glob_pattern, native_name) in var_map.items():
            var_dir = self.root_dir / step_subdir
            files: list[Path] = []
            for year in range(self.start_year, self.end_year + 1):
                year_dir = var_dir / str(year)
                matched = sorted(year_dir.glob(glob_pattern + ".nc"))
                if not matched:
                    matched = sorted(var_dir.glob(f"*{year}*" + ".nc"))
                files.extend(matched)
            # De-duplicate while preserving order (fallback glob can re-match).
            seen, uniq = set(), []
            for f in files:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            if not uniq:
                warnings.warn(
                    f"[LANLMJODataset] No files found for '{aurora_name}' in {var_dir} "
                    f"for years {self.start_year}–{self.end_year}. Skipping variable.",
                    stacklevel=2)
                continue
            result[aurora_name] = (uniq, native_name)
        return result

    def _build_aligned_index(self):
        """Build per-variable timestamp maps and the common sample timeline.

        This is the heart of the v3 fix.  For each variable independently:
          probe each file's `time` coordinate (coords only - cheap, no data),
          convert to int64 unix seconds, DROP anything outside the requested
          year range, and record ts -> (file_idx, local_idx).  If the same
          timestamp appears in two files (overlapping files), the FIRST
          occurrence wins and the duplicate is counted + reported.

        The usable timeline is the sorted intersection across all variables;
        a sample with start index i is valid only if timeline contains the
        exact 6-hourly chain [t_i, t_i+6h, ..., t_i+(1+max_rollout_steps)*6h].
        """
        all_maps: dict[str, dict[int, tuple[int, int]]] = {}
        report_rows = []

        for aurora_name, (files, _native) in {**self.surf_file_map, **self.atmos_file_map}.items():
            ts_map: dict[int, tuple[int, int]] = {}
            n_total, n_dropped, n_dupes = 0, 0, 0
            for fi, f in enumerate(files):
                with xr.open_dataset(str(f), engine="netcdf4", decode_times=True) as ds:
                    secs = _to_seconds_i64(ds.time.values)
                n_total += len(secs)
                for li, s in enumerate(secs.tolist()):
                    if s < self._t_min or s > self._t_max:
                        n_dropped += 1          # out-of-range (this was the leak)
                        continue
                    if s in ts_map:
                        n_dupes += 1            # overlapping/duplicate file content
                        continue
                    ts_map[s] = (fi, li)
            all_maps[aurora_name] = ts_map
            report_rows.append((aurora_name, len(files), n_total, len(ts_map), n_dropped, n_dupes))

        # Intersection across all variables.
        keysets = [set(m.keys()) for m in all_maps.values()]
        common = sorted(set.intersection(*keysets)) if keysets else []

        # Contiguity-checked sample starts:  a sample at timeline position i
        # consumes timestamps  t_i (history t-6h), t_i+6h (current t),
        # then targets t_i+12h ... t_i+(1+max_rollout_steps)*6h.
        step_s = STEP_HOURS * 3600
        need = 1 + self.max_rollout_steps + 1   # 2 inputs + max_rollout targets
        common_arr = np.asarray(common, dtype=np.int64)
        common_pos = {s: i for i, s in enumerate(common)}
        valid_starts: list[int] = []
        for i in range(len(common) - need + 1):
            # exact 6-hourly chain check (vectorized slice compare)
            if np.all(np.diff(common_arr[i:i + need]) == step_s):
                valid_starts.append(i)

        self._var_ts_map = all_maps
        self._timeline = common_arr             # int64 seconds, sorted
        self._valid_starts = np.asarray(valid_starts, dtype=np.int64)
        self.num_samples = len(self._valid_starts)

        # ---- Alignment report (prints once per rank at init) --------------
        union = set.union(*keysets) if keysets else set()
        print(f"  [align] requested range {self.start_year}-{self.end_year} | "
              f"common timeline: {len(common)} steps | union: {len(union)} | "
              f"valid {self.max_rollout_steps}-step samples: {self.num_samples}")
        for name, nf, ntot, nkept, ndrop, ndup in report_rows:
            miss = len(common) and (len(union) - nkept)
            flag = ""
            if ndrop:
                flag += f"  DROPPED-OUT-OF-RANGE={ndrop}"
            if ndup:
                flag += f"  DUPLICATES={ndup}"
            if nkept != len(common):
                flag += f"  (this variable limits/differs from intersection by {nkept - len(common):+d})"
            print(f"  [align]   {name:>5}: files={nf:<4} raw_steps={ntot:<7} kept={nkept:<7}{flag}")
        gaps = (len(common) - need + 1) - self.num_samples if len(common) >= need else 0
        if gaps > 0:
            print(f"  [align]   NOTE: {gaps} potential sample starts excluded by 6-hourly gaps.")

        if self.num_samples <= 0:
            raise RuntimeError(
                "[LANLMJODataset] Zero valid samples after timestamp alignment. "
                "Run tools/diagnose_val_nan.py for the per-variable audit."
            )

    def _probe_pressure_level_indices(self) -> list[int]:
        if not self.atmos_file_map:
            return []
        ref_files = next(iter(self.atmos_file_map.values()))[0]
        with xr.open_dataset(str(ref_files[0]), engine="netcdf4") as ds:
            if 'lev' in ds.coords:
                all_levs = ds.lev.values
            elif 'level' in ds.coords:
                all_levs = ds.level.values
            else:
                warnings.warn("[LANLMJODataset] No 'lev'/'level' coord found; using all levels.",
                              stacklevel=2)
                return []
        return [int(np.argmin(np.abs(all_levs - p))) for p in AURORA_PLEVS]

    def _load_static_vars(self):
        """Z, LSM, SLT as pure tensors, upsampled to 0.25deg.  v3: NaN/Inf are
        zeroed for ALL three statics (previously only slt was sanitized)."""
        static_dir = self.root_dir / "Step00/ERA5.invariant"

        z_files = list(static_dir.glob("*_z.*.nc"))
        if not z_files:
            raise FileNotFoundError(f"Invariant Z file not found in {static_dir}")
        lsm_files = list(static_dir.glob("*_lsm.*.nc"))
        if not lsm_files:
            raise FileNotFoundError(f"Invariant LSM file not found in {static_dir}")

        def _ensure_2d(t):
            while t.ndim > 2:
                t = t.squeeze(0)
            return t

        def _clean(t):
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            with xr.open_dataset(z_files[0], engine="netcdf4") as ds_z:
                z_arr = ds_z['Z'].values
            z_tensor = _upsample_to_aurora(_clean(torch.from_numpy(z_arr).float()))
        except Exception as e:
            warnings.warn(f"[LANLMJODataset] Failed to load invariant Z: {e}. Using zeros.")
            z_tensor = torch.zeros(720, 1440)

        try:
            with xr.open_dataset(lsm_files[0], engine="netcdf4") as ds_lsm:
                lsm_arr = ds_lsm['LSM'].values
            lsm_tensor = _upsample_to_aurora(_clean(torch.from_numpy(lsm_arr).float()))
        except Exception as e:
            warnings.warn(f"[LANLMJODataset] Failed to load invariant LSM: {e}. Using zeros.")
            lsm_tensor = torch.zeros(720, 1440)

        with xr.open_dataset(self.slt_path, engine="netcdf4") as ds_slt:
            slt_arr = ds_slt['slt'].values
        slt_tensor = _clean(torch.from_numpy(slt_arr).float())
        slt_tensor = _ensure_2d(slt_tensor)[:720, :]

        return {"z": _ensure_2d(z_tensor), "lsm": _ensure_2d(lsm_tensor), "slt": slt_tensor}

    # ------------------------------------------------------------------
    # Per-item read (forked workers — must stay fork-safe: open/close fresh)
    # ------------------------------------------------------------------

    def _read_var_at_times(self, aurora_name: str, files, native_name: str,
                           ts_list: list[int]) -> np.ndarray:
        """Read one variable at explicit timestamps via ITS OWN (fi, li) map."""
        ts_map = self._var_ts_map[aurora_name]
        arrays = []
        for s in ts_list:
            fi, li = ts_map[s]     # guaranteed present: timeline ⊆ every var's map
            with xr.open_dataset(str(files[fi]), engine="netcdf4") as ds:
                arr = ds[native_name].isel(time=li).values
            arrays.append(arr)
        return np.stack(arrays, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i0 = int(self._valid_starts[idx])
        step_s = STEP_HOURS * 3600
        t_hist = int(self._timeline[i0])            # t - 6h
        t_curr = t_hist + step_s                    # t
        input_ts = [t_hist, t_curr]

        def process_var(arr):
            # Zero (not clamp-to-3.4e38) BOTH NaN and Inf — the default
            # nan_to_num turns +Inf into the largest finite float, which is a
            # ready-made overflow bomb one matmul later.
            return torch.nan_to_num(torch.from_numpy(arr), nan=0.0, posinf=0.0, neginf=0.0)

        # --- Surface inputs ---
        surf_in = {}
        for aurora_name, (files, native_name) in self.surf_file_map.items():
            raw = self._read_var_at_times(aurora_name, files, native_name, input_ts)
            surf_in[aurora_name] = process_var(raw)[None]      # (1, 2, H, W)

        # --- Atmospheric inputs ---
        atmos_in = {}
        for aurora_name, (files, native_name) in self.atmos_file_map.items():
            raw = self._read_var_at_times(aurora_name, files, native_name, input_ts)
            if self._plev_indices:
                raw = raw[:, self._plev_indices, :, :]
            atmos_in[aurora_name] = process_var(raw)[None]     # (1, 2, 13, H, W)

        # --- Multi-step targets ---
        surf_targets_list, atmos_targets_list = [], []
        for step in range(self.max_rollout_steps):
            tgt_ts = [t_curr + (step + 1) * step_s]
            surf_out = {}
            for aurora_name, (files, native_name) in self.surf_file_map.items():
                raw = self._read_var_at_times(aurora_name, files, native_name, tgt_ts)
                surf_out[aurora_name] = process_var(raw)       # (1, H, W)
            surf_targets_list.append(surf_out)

            atmos_out = {}
            for aurora_name, (files, native_name) in self.atmos_file_map.items():
                raw = self._read_var_at_times(aurora_name, files, native_name, tgt_ts)
                if self._plev_indices:
                    raw = raw[:, self._plev_indices, :, :]
                atmos_out[aurora_name] = process_var(raw)      # (1, 13, H, W)
            atmos_targets_list.append(atmos_out)

        # --- Time tag straight from the timeline (no extra file open) ---
        init_time = datetime.utcfromtimestamp(t_curr)

        in_batch = Batch(
            surf_vars=surf_in,
            atmos_vars=atmos_in,
            static_vars=self.static_vars,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(init_time,),
                atmos_levels=self.atmos_levels,
                rollout_step=0,
            ),
        )
        return in_batch, surf_targets_list, atmos_targets_list


def collate_fn(batch_list):
    return batch_list[0]
