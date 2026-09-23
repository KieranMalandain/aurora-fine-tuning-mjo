#!/usr/bin/env python3
"""fetch_rmm_reference.py — Retrieve and format official BoM RMM reference time series.

Task J2 (Q-15 resolution):
Retrieves the official Wheeler & Hendon (2004) Real-Time Multivariate MJO (RMM)
time series published by the Australian Bureau of Meteorology (BoM) at:
http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt

The script fetches the raw fixed-width text archive, parses dates and RMM
components, maps missing value sentinels (1.E36, 999) to NaN, writes a cleaned
CSV to data/reference/rmm_bom.csv, and computes the SHA256 checksum for
reproducible verification without repeated unversioned network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
DEFAULT_OUTPUT = Path("data/reference/rmm_bom.csv")
MISSING_THRESHOLDS = (900.0, 1e30)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a local file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_bom_rmm(
    url: str = DEFAULT_URL,
    timeout: int = 30,
) -> str:
    """Download the raw text content from BoM URL."""
    headers = {"User-Agent": "Mozilla/5.0 (aurora-fine-tuning-mjo J2 validation)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    return content


def parse_bom_text(content: str) -> list[dict[str, str | int | float]]:
    """Parse BoM fixed-width text format into structured records.

    Header format:
    Line 0: Description of RMM processing history (1974-2013 vs 2014-)
    Line 1: Column names and missing value flags (1.E36 or 999)
    Lines 2+: whitespace-delimited columns:
      year, month, day, RMM1, RMM2, phase, amplitude, [method]
    """
    lines = content.strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"BoM file contains fewer than 3 lines: {len(lines)}")

    records: list[dict[str, str | int | float]] = []
    for line_num, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7:
            raise ValueError(f"Malformed data at line {line_num}: {line}")

        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
            phase = int(parts[5])
            amplitude = float(parts[6])
            method = parts[7] if len(parts) > 7 else ""

            # Check missing value sentinels
            if any(rmm1 >= thresh for thresh in MISSING_THRESHOLDS) or any(
                rmm2 >= thresh for thresh in MISSING_THRESHOLDS
            ):
                rmm1_clean = float("nan")
                rmm2_clean = float("nan")
                phase_clean = 0
                amplitude_clean = float("nan")
            else:
                rmm1_clean = rmm1
                rmm2_clean = rmm2
                phase_clean = phase
                amplitude_clean = amplitude

            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            records.append(
                {
                    "date": date_str,
                    "year": year,
                    "month": month,
                    "day": day,
                    "rmm1": rmm1_clean,
                    "rmm2": rmm2_clean,
                    "phase": phase_clean,
                    "amplitude": amplitude_clean,
                    "method": method,
                }
            )
        except (ValueError, IndexError) as err:
            raise ValueError(f"Error parsing line {line_num}: {line}") from err

    return records


def write_csv(records: list[dict[str, str | int | float]], out_path: Path) -> None:
    """Write parsed records to CSV with standard schema."""
    import pandas as pd

    df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to raw BoM text file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target CSV output path (default: data/reference/rmm_bom.csv)",
    )
    parser.add_argument(
        "--verify-sha256",
        type=str,
        default=None,
        help="Optional expected SHA256 checksum to verify existing file",
    )
    args = parser.parse_args()

    if args.verify_sha256:
        if not args.output.is_file():
            print(f"File {args.output} does not exist for checksum verification.", file=sys.stderr)
            return 1
        current_sha = compute_sha256(args.output)
        if current_sha.lower() == args.verify_sha256.lower():
            print(f"Checksum verified: {current_sha}")
            return 0
        else:
            print(
                f"Checksum mismatch: expected {args.verify_sha256}, got {current_sha}",
                file=sys.stderr,
            )
            return 1

    print(f"Fetching BoM RMM series from {args.url}...")
    raw_text = fetch_bom_rmm(args.url)
    records = parse_bom_text(raw_text)
    print(f"Parsed {len(records)} records spanning {records[0]['date']} to {records[-1]['date']}.")

    write_csv(records, args.output)
    sha256 = compute_sha256(args.output)
    print(f"Wrote cleaned CSV to {args.output}")
    print(f"File size: {args.output.stat().st_size} bytes")
    print(f"SHA256: {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
