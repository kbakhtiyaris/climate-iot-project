
# scripts/train_models.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path
import argparse
import difflib
import os

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models import TemperatureForecaster

def detect_target_column(df, requested="temp_avg"):
    cols = list(df.columns)
    # exact / case-insensitive match
    if requested in cols:
        return requested
    for c in cols:
        if c.lower() == requested.lower():
            return c

    # Prefer columns that contain 'temp' and 'avg' or 'mean'
    temp_cols = [c for c in cols if 'temp' in c.lower()]
    if temp_cols:
        for c in temp_cols:
            if 'avg' in c.lower() or 'mean' in c.lower():
                return c
        # fallback to first 'temp' column
        return temp_cols[0]

    # common alternatives (expanded)
    common = ['temp_avg','temperature','temp','tavg','avg_temp','temp_mean','temp_c','avg_temp_c']
    for cand in common:
        for c in cols:
            if c.lower() == cand:
                return c

    # fuzzy match as final fallback
    close = difflib.get_close_matches(requested, cols, n=1, cutoff=0.6)
    if close:
        return close[0]
    return None

def train_and_evaluate(
    data_path="data/processed/weather_clean_sample.csv",
    target_col="temp_avg",
    limit_rows=None,
    show_columns=False,
    city=None,
    station_id=None
):
    """Train models and evaluate performance"""
    print("\n" + "="*60)
    print("🌍 TRAINING ML MODELS")
    print("="*60)
    
    # Load cleaned data (CSV)
    df = pd.read_csv(data_path)
    print("Columns:", list(df.columns))
    if show_columns:
        return

    # Normalize 'date' column
    if 'date' not in df.columns:
        raise KeyError("Expected a 'date' column in the CSV.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # Optional: Filter to a single series (recommended for ARIMA/Prophet)
    if city and 'city_name' in df.columns:
        df = df[df['city_name'] == city].copy()
        print(f"Filter: city_name == {city} → {len(df)} rows")
    if station_id and 'station_id' in df.columns:
        df = df[df['station_id'] == station_id].copy()
        print(f"Filter: station_id == {station_id} → {len(df)} rows")

    # Resolve target column safely
    detected = detect_target_column(df, target_col)
    if detected is None:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}. "
            "Re-run with --target-col <name>."
        )
    if detected != target_col:
        print(f"⚠️ Using detected target column '{detected}' instead of requested '{target_col}'.")
    target_col = detected
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    if limit_rows:
        df = df.iloc[:limit_rows]
        print(f"Limiting to first {limit_rows} rows.")
    
    # Split data chronologically
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    
    print(f"Train set: {len(train_data)} records")
    print(f"Test set: {len(test_data)} records")
    
    # Train ARIMA
    print("\n" + "-"*60)
    arima_forecaster = TemperatureForecaster(model_type='arima')
    arima_forecaster.train_arima(train_data, target_col=target_col, order=(5, 1, 2))
    
    # Make predictions
    arima_forecast = arima_forecaster.predict(periods=len(test_data))
    
    # Evaluate
    y_true = pd.to_numeric(test_data[target_col], errors='coerce')
    y_true = y_true.dropna().values
    # Align prediction length to y_true length in case NA removal changed count
    y_pred = arima_forecast['forecast'].values[:len(y_true)]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n✓ ARIMA Performance:")
    print(f"  - MAE: {mae:.2f}°C")
    print(f"  - RMSE: {rmse:.2f}°C")
    print(f"  - R²: {r2:.4f}")
    
    # Save ARIMA model (pickle ok)
    Path("models").mkdir(parents=True, exist_ok=True)
    arima_forecaster.save_model('models/arima_model.pkl')
    
    # Train Prophet
    print("\n" + "-"*60)
    prophet_forecaster = TemperatureForecaster(model_type='prophet')
    prophet_forecaster.train_prophet(train_data, target_col=target_col)
    
    prophet_forecast = prophet_forecaster.predict(periods=len(test_data))
    
    # Evaluate
    y_pred_prophet = prophet_forecast['forecast'].values[:len(y_true)]
    mae_p = mean_absolute_error(y_true, y_pred_prophet)
    rmse_p = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
    r2_p = r2_score(y_true, y_pred_prophet)
    
    print(f"\n✓ PROPHET Performance:")
    print(f"  - MAE: {mae_p:.2f}°C")
    print(f"  - RMSE: {rmse_p:.2f}°C")
    print(f"  - R²: {r2_p:.4f}")
    
    # Save Prophet model (JSON serialization)
    prophet_forecaster.save_model('models/prophet_model.json')
    
    # Summary
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    print(f"\n{'Model':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*60)
    print(f"{'ARIMA':<15} {mae:<10.2f} {rmse:<10.2f} {r2:<10.4f}")
    print(f"{'Prophet':<15} {mae_p:<10.2f} {rmse_p:<10.2f} {r2_p:<10.4f}")
    
    print("\n✓ Models trained and saved!\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/processed/weather_clean_sample.csv")
    p.add_argument("--target-col", default="temp_avg")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--show-columns", action="store_true")
    p.add_argument("--city", default=None, help="Filter to a single city_name (optional)")
    p.add_argument("--station-id", default=None, help="Filter to a single station_id (optional)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        target_col=args.target_col,
        limit_rows=args.limit_rows,
        show_columns=args.show_columns,
        city=args.city,
        station_id=args.station_id
    )
