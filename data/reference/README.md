# Official BoM Wheeler & Hendon RMM Reference Series

This directory stores the official Wheeler & Hendon (2004) Real-time Multivariate MJO (RMM) reference series published by the Australian Bureau of Meteorology (BoM), resolving **Q-15** for task **J2**.

---

## 1. Source and Retrieval Metadata

- **Provider**: Australian Bureau of Meteorology (BoM), Climate and Oceans Support Program in the Pacific.
- **Reference URL**: `http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt`
- **Retrieval Date**: 2026-09-22
- **Retrieval Script**: [`scripts/fetch_rmm_reference.py`](../../scripts/fetch_rmm_reference.py)
- **Primary Reference**:
  Wheeler, M. C., & Hendon, H. H. (2004). *An All-Season Real-Time Multivariate MJO Index: Development of an Index for Monitoring and Prediction*. Monthly Weather Review, 132(8), 1917–1932. https://doi.org/10.1175/1520-0493(2004)132<1917:AARMMI>2.0.CO;2
- **Licence / Terms of Use**:
  Australian Bureau of Meteorology data is provided for public research and educational purposes under Commonwealth of Australia Copyright (Australian Copyright Act 1968). Reproduction for study, research, and non-commercial scientific verification is permitted under fair dealing provisions.
- **Durable Storage**:
  Committed directly to repository version control at `data/reference/rmm_bom.csv`. Never re-fetched dynamically in automated pipelines or CI.

---

## 2. Integrity and Checksums

| File | Size (bytes) | SHA256 Digest |
| --- | ---: | --- |
| `rmm_bom.csv` | 1,533,498 | `8501dc4dbec5159926e7f0304fec3d0c322c7b625b61b452f1fd4fd4746c92ac` |

Verification command:
```bash
uv run python scripts/fetch_rmm_reference.py --verify-sha256 8501dc4dbec5159926e7f0304fec3d0c322c7b625b61b452f1fd4fd4746c92ac
```

---

## 3. Processing History Recorded by BoM

The raw BoM file header documents two historical processing regimes:
- **1974-06-01 to 2013-12-31**: Both SST1 variability (ENSO proxy linear regression) and the trailing 120-day mean were removed from the underlying OLR and wind anomalies.
- **2014-01-01 to present**: Only the trailing 120-day running mean has been removed (no separate SST1 regression).

For our validation period (**2016–2019**), the BoM processing regime is therefore strictly the 120-day running mean removal (Convention A), matching this project's specification (`02_SCIENTIFIC_CONTRACT.md` §6).

---

## 4. Column Semantics

`rmm_bom.csv` provides daily records formatted with comma separation:

| Column | Type | Description | Valid Range / Format |
| --- | --- | --- | --- |
| `date` | `string` | ISO-8601 calendar date | `YYYY-MM-DD` |
| `year` | `integer` | Calendar year (UTC) | 1974 – 2024 |
| `month` | `integer` | Calendar month (1–12) | 1 – 12 |
| `day` | `integer` | Calendar day of month (1–31) | 1 – 31 |
| `rmm1` | `float` | Leading principal component (RMM1). Missing values mapped to `NaN`. | Typically ~ `[-4.0, +4.0]` |
| `rmm2` | `float` | Second principal component (RMM2). Missing values mapped to `NaN`. | Typically ~ `[-4.0, +4.0]` |
| `phase` | `integer` | Wheeler–Hendon octant phase (1–8). Unclassified/weak MJO ($A < 1$) denoted by 0. | 0 – 8 |
| `amplitude` | `float` | Vector magnitude $\sqrt{\text{RMM1}^2 + \text{RMM2}^2}$. Missing values mapped to `NaN`. | $\ge 0.0$ |
| `method` | `string` | Processing method note from BoM source file. | e.g., `Gottschalk10_method:_OLR_&_ACCESS_wind` |
