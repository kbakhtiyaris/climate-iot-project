#!/usr/bin/env python3
# scripts/load_data.py
"""
Load cleaned processed data into the Postgres DB.

- Prefers Parquet (data/processed/weather_clean.parquet)
- Falls back to CSV preview if Parquet isn't available
- Robustly maps common column name variants to DB model fields
- Adds project root to sys.path so `src` can be imported when running the script directly
"""

from __future__ import annotations

from pathlib import Path
import sys
import uuid
from datetime import datetime
from typing import Optional, Iterable

# Ensure the project root is in sys.path so `src.*` imports work even when running the script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.database import get_session, WeatherData


def _find_parquet_or_csv(processed_dir: Path) -> Path:
    # Try explicit file name first, then any *.parquet, then *.csv
    candidates = [
        processed_dir / "weather_clean.parquet",
        processed_dir / "weather_clean.pq",
        processed_dir / "weather_clean.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # more general fallback
    pq_files = sorted(processed_dir.glob("*.parquet"))
    if pq_files:
        return pq_files[0]
    csv_files = sorted(processed_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(f"No processed Parquet or CSV found in {processed_dir}")


def _first_existing_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# Mapping from model attribute -> candidate column names in the file (ordered)
COL_CANDIDATES = {
    "date": ["date", "datetime", "timestamp", "time"],
    "city": ["city", "city_name"],
    "country": ["country", "country_name"],
    "latitude": ["latitude", "lat", "latitude_deg"],
    "longitude": ["longitude", "lon", "lng"],
    "temp_avg": ["temp_avg", "avg_temp_c", "avg_temp", "mean_temp"],
    "temp_min": ["temp_min", "min_temp_c", "min_temp"],
    "temp_max": ["temp_max", "max_temp_c", "max_temp"],
    "humidity": ["humidity", "rel_humidity", "relative_humidity", "hum"],
    "precipitation": ["precipitation", "precip_mm", "rain_mm"],
    "wind_speed": ["wind_speed", "avg_wind_speed_kmh", "wind_kmh", "wind_speed_kmh"],
    "pressure": ["pressure", "sea_level_pressure", "avg_sea_level_pres_hpa"],
}


def _resolve_column_map(df: pd.DataFrame):
    # Returns map: model_attr -> df column name (or None to use a default)
    cols = list(df.columns)
    mapping = {}
    for model_key, candidates in COL_CANDIDATES.items():
        found = _first_existing_col(cols, candidates)
        mapping[model_key] = found
    return mapping


def _get_val_safe(row, col_name, default=None, cast=float):
    val = row.get(col_name, default) if col_name else default
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return cast(val) if val != "" else default
    except Exception:
        return default


def load_data_to_db():
    """Load cleaned data into PostgreSQL"""
    try:
        processed_dir = Path("data/processed").resolve()
        data_file = _find_parquet_or_csv(processed_dir)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return

    print("\n=== LOADING DATA INTO DATABASE ===")
    print(f"Loading processed data from: {data_file}")

    # Choose read path
    ext = data_file.suffix.lower()
    if ext in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(data_file)
        except Exception as e:
            print("✗ Error when reading Parquet (pyarrow/fastparquet may be required):", e)
            raise
    elif ext == ".csv":
        df = pd.read_csv(data_file)
    else:
        raise ValueError(f"Unsupported filetype: {ext}")

    print(f"✓ Loaded {len(df)} records from {data_file.name}")

    # Resolve columns that map to the DB fields
    col_map = _resolve_column_map(df)

    # Test the date column presence
    date_col = col_map["date"]
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        print("⚠ No date/ts-like column found; `date` will be set to None for records.")

    session = get_session()

    try:
        # Insert data in batches
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            for _, row in batch.iterrows():
                # Fetch values by mapping; if a mapping is missing fallback to columns common names
                weather = WeatherData(
                    id=str(uuid.uuid4()),
                    date=_get_val_safe(row, col_map.get("date"), default=None, cast=lambda v: pd.to_datetime(v).date() if pd.notna(v) else None),
                    city=str(row.get(col_map.get("city"), "Unknown") or "Unknown"),
                    country=str(row.get(col_map.get("country"), "Unknown") or "Unknown"),
                    latitude=_get_val_safe(row, col_map.get("latitude"), default=0.0, cast=float),
                    longitude=_get_val_safe(row, col_map.get("longitude"), default=0.0, cast=float),
                    temp_avg=_get_val_safe(row, col_map.get("temp_avg"), default=0.0, cast=float),
                    temp_min=_get_val_safe(row, col_map.get("temp_min"), default=0.0, cast=float),
                    temp_max=_get_val_safe(row, col_map.get("temp_max"), default=0.0, cast=float),
                    humidity=_get_val_safe(row, col_map.get("humidity"), default=0.0, cast=float),
                    precipitation=_get_val_safe(row, col_map.get("precipitation"), default=0.0, cast=float),
                    wind_speed=_get_val_safe(row, col_map.get("wind_speed"), default=0.0, cast=float),
                    pressure=_get_val_safe(row, col_map.get("pressure"), default=0.0, cast=float),
                )
                session.add(weather)

            session.commit()
            print(f"  ✓ Loaded batch {i // batch_size + 1} ({min(i + batch_size, len(df))}/{len(df)} records)")

        print(f"✓ All {len(df)} records loaded!\n")
    except Exception as e:
        session.rollback()
        print(f"✗ Error: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    load_data_to_db()
