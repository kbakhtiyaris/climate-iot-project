
"""
Cleaning & preprocessing for climate data (large-file friendly).

- For large Parquet (e.g., 5M+ rows), use DuckDB to:
  * Convert date to TIMESTAMP
  * Fill nulls with per-column MEDIANs
  * Clip numeric columns to plausible domain ranges
  * Write Parquet out-of-core (no OOM)

- For smaller data, fall back to pandas pipeline.

Run from project root:
    python -m src.data_processing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---- Domain-based clipping ranges for weather metrics ----
# Matches your dataset's numeric columns:
# ['avg_temp_c','min_temp_c','max_temp_c','precipitation_mm','snow_depth_mm',
#  'avg_wind_dir_deg','avg_wind_speed_kmh','peak_wind_gust_kmh',
#  'avg_sea_level_pres_hpa','sunshine_total_min']
CLIP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "avg_temp_c": (-80.0, 60.0),
    "min_temp_c": (-100.0, 60.0),
    "max_temp_c": (-80.0, 70.0),
    "precipitation_mm": (0.0, 500.0),
    "snow_depth_mm": (0.0, 5000.0),        # depth in mm
    "avg_wind_dir_deg": (0.0, 360.0),
    "avg_wind_speed_kmh": (0.0, 250.0),
    "peak_wind_gust_kmh": (0.0, 350.0),
    "avg_sea_level_pres_hpa": (870.0, 1085.0),
    "sunshine_total_min": (0.0, 1440.0),   # minutes per day
}

# ---------------- Small-data (pandas) cleaner ----------------
def _find_date_column(columns: List[str]) -> Optional[str]:
    candidates = [
        "date", "Date", "DATE",
        "datetime", "Datetime", "DATETIME",
        "timestamp", "Timestamp", "TIMESTAMP",
        "time", "Time", "TIME",
        "dt", "DT",
        "day", "Day", "DAY",
    ]
    for c in candidates:
        if c in columns:
            return c
    for c in columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    return None


def clean_weather_data_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight pandas cleaner for smaller datasets."""
    df_clean = df.copy()

    print("\n=== DATA CLEANING (pandas) ===")
    print(f"Original shape: {df_clean.shape}")

    # 1) Convert date column to datetime if present
    date_col = _find_date_column(list(df_clean.columns))
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        print(f"✓ Converted {date_col} to datetime")
    else:
        print("⚠ No date-like column found; skipping datetime conversion")

    # 2) Handle missing values (small data only)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("⚠ No numeric columns found; skipping interpolation and clipping")
    else:
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(
            method='linear', limit_direction='both'
        )
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].mean(numeric_only=True)
        )
        print(f"✓ Handled missing values in {len(numeric_cols)} numeric columns")

        # 3) Domain-based clipping (fast and robust)
        clipped_cols = 0
        for col, (lo, hi) in CLIP_BOUNDS.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lo, hi)
                clipped_cols += 1
        print(f"✓ Clipped outliers using domain ranges ({clipped_cols} columns)")

    print(f"Clean shape: {df_clean.shape}\n")
    return df_clean


def save_processed_data(
    df: pd.DataFrame,
    parquet_path: str = 'data/processed/weather_clean.parquet',
    csv_path: Optional[str] = 'data/processed/weather_clean_sample.csv',
    csv_rows: int = 100_000
) -> tuple[Optional[str], Optional[str]]:
    """Save cleaned data to Parquet and a small CSV preview."""
    pq_out = Path(parquet_path)
    pq_out.parent.mkdir(parents=True, exist_ok=True)

    # Save Parquet
    pq_path_str: Optional[str] = None
    try:
        df.to_parquet(pq_out, index=False)
        print(f"✓ Saved cleaned data (Parquet) to: {pq_out}")
        pq_path_str = str(pq_out)
    except ImportError:
        print("⚠ Parquet engine missing (pyarrow/fastparquet). Install with: pip install pyarrow")
        print("  Skipping Parquet save.")

    csv_out_path = None
    if csv_path:
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.head(csv_rows).to_csv(csv_out, index=False)
        print(f"✓ Saved preview CSV ({csv_rows} rows) to: {csv_out}")
        csv_out_path = str(csv_out)

    return pq_path_str, csv_out_path

# ---------------- Large-data (DuckDB) cleaner ----------------
def process_large_with_duckdb(
    parquet_in: str | Path,
    parquet_out: str | Path = 'data/processed/weather_clean.parquet',
    preview_csv_out: Optional[str] = 'data/processed/weather_clean_sample.csv',
    preview_rows: int = 100_000,
) -> tuple[str, Optional[str]]:
    """
    Clean huge Parquet using DuckDB (out-of-core):
      - cast date -> TIMESTAMP (if needed)
      - fill NULLs with column MEDIANs
      - clip numeric columns to domain bounds
      - write Parquet (and optional preview CSV)
    """
    import duckdb

    parquet_in = str(parquet_in)
    parquet_out = str(parquet_out)
    if preview_csv_out is not None:
        preview_csv_out = str(preview_csv_out)

    # Determine available columns
    try:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(parquet_in).schema.names
    except Exception:
        qcols = duckdb.connect().execute(
            f"DESCRIBE SELECT * FROM parquet_scan('{parquet_in}') LIMIT 0"
        ).fetchdf()
        cols = qcols['column_name'].tolist()

    cols_set = set(cols)
    # Choose numeric columns intersecting our domain map
    numeric_cols_present = [c for c in CLIP_BOUNDS.keys() if c in cols_set]

    # Date column detection
    date_col = None
    for c in ["date", "datetime", "timestamp", "time", "dt"]:
        if c in cols_set:
            date_col = c
            break

    # Build SELECT lists
    dim_cols: list[str] = []
    if "station_id" in cols_set:
        dim_cols.append("station_id")
    if "city_name" in cols_set:
        dim_cols.append("city_name")
    if "season" in cols_set:
        dim_cols.append("season")

    # Base SELECT: dims + date (cast) + numeric raw
    base_select_parts: list[str] = []
    base_select_parts += dim_cols
    if date_col:
        base_select_parts.append(f"try_cast({date_col} AS TIMESTAMP) AS {date_col}")
    base_select_parts += numeric_cols_present

    base_sql = ",\n              ".join(base_select_parts) if base_select_parts else "*"

    # stats: medians for numeric columns
    med_parts = [f"median({c}) AS med_{c}" for c in numeric_cols_present]
    med_sql = ", ".join(med_parts) if med_parts else "1 as dummy_median"

    # filled: coalesce numeric with medians
    filled_parts: list[str] = []
    for part in base_select_parts:
        # normalize alias
        if " AS " in part:
            alias = part.split(" AS ")[-1].strip()
            # date column already aliased; keep as-is
            filled_parts.append(alias)
        else:
            col = part.strip()
            if col in numeric_cols_present:
                filled_parts.append(f"COALESCE({col}, (SELECT med_{col} FROM stats)) AS {col}")
            else:
                filled_parts.append(col)
    filled_sql = ",\n              ".join(filled_parts)

    # final: clip numeric to domain bounds
    final_parts: list[str] = []
    for part in filled_parts:
        col = part.split(" AS ")[-1] if " AS " in part else part
        col = col.strip()
        if col in numeric_cols_present:
            lo, hi = CLIP_BOUNDS[col]
            final_parts.append(f"LEAST(GREATEST({col}, {lo}), {hi}) AS {col}")
        else:
            final_parts.append(col)
    final_sql = ",\n              ".join(final_parts)

    con = duckdb.connect()
    out_dir = Path(parquet_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DuckDB cleaning (large file) ...")
    con.execute(f"""
        COPY (
          WITH base AS (
            SELECT
              {base_sql}
            FROM parquet_scan('{parquet_in}')
          ),
          stats AS (
            SELECT {med_sql} FROM base
          ),
          filled AS (
            SELECT
              {filled_sql}
            FROM base
          )
          SELECT
            {final_sql}
          FROM filled
        ) TO '{parquet_out}' (FORMAT PARQUET);
    """)

    print(f"✓ Saved cleaned data (Parquet) to: {parquet_out}")

    csv_preview_path = None
    if preview_csv_out:
        csv_dir = Path(preview_csv_out).parent
        csv_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY (
              SELECT * FROM read_parquet('{parquet_out}') LIMIT {preview_rows}
            ) TO '{preview_csv_out}' (HEADER, DELIMITER ',');
        """)
        print(f"✓ Saved preview CSV ({preview_rows} rows) to: {preview_csv_out}")
        csv_preview_path = preview_csv_out

    con.close()
    return parquet_out, csv_preview_path


def _detect_large_parquet(parquet_path: Path, row_threshold: int = 5_000_000) -> bool:
    """Use Parquet metadata to check if file is 'large' by row count."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(parquet_path))
        n = pf.metadata.num_rows
        logger.info("Parquet rows detected: %s", n)
        return n >= row_threshold
    except Exception:
        # Fallback: file size heuristic (>= 200MB)
        return parquet_path.stat().st_size >= 200 * 1024 * 1024


# -------------------- Script entry point --------------------
if __name__ == "__main__":
    raw_dir = Path("data/raw").resolve()
    pq_files = sorted(raw_dir.glob("*.parquet"))
    csv_files = sorted(raw_dir.glob("*.csv"))

    if pq_files:
        # Prefer a known filename if present
        preferred = [p for p in pq_files if p.name.lower().startswith("daily_weather")]
        parquet_in = preferred[0] if preferred else pq_files[0]

        is_large = _detect_large_parquet(parquet_in)
        if is_large:
            # Large route (DuckDB)
            process_large_with_duckdb(
                parquet_in=parquet_in,
                parquet_out="data/processed/weather_clean.parquet",
                preview_csv_out="data/processed/weather_clean_sample.csv",
                preview_rows=100_000,
            )
        else:
            # Small enough to use pandas
            df = pd.read_parquet(parquet_in)
            df_clean = clean_weather_data_pandas(df)
            save_processed_data(df_clean)
    elif csv_files:
        # Fall back to loader to pick the most relevant CSV
        try:
            from src.data_loader import load_weather_data
        except ModuleNotFoundError:
            print("ModuleNotFoundError: No module named 'src'.")
            print("Fix by either:\n"
                  "  1) touch src/__init__.py and run: python -m src.data_processing\n"
                  "  2) export PYTHONPATH=$(pwd) and run: python src/data_processing.py")
            raise

        df = load_weather_data()  # loader prefers 'daily'/'climate' CSV or largest one
        df_clean = clean_weather_data_pandas(df)
        save_processed_data(df_clean)
    else:
        # Nothing to process
        raise FileNotFound





