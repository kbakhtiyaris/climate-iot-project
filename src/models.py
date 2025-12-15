
# src/models.py
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pandas as pd
import numpy as np
import logging
import pickle
import difflib

logger = logging.getLogger(__name__)

class TemperatureForecaster:
    """Temperature prediction using different models"""
    
    def __init__(self, model_type='arima'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
    
    def train_arima(self, df, target_col='temp_avg', order=(5, 1, 2)):
        """Train ARIMA model"""
        print(f"\n=== TRAINING ARIMA{order} MODEL ===")
        
        try:
            # Sort by date and set index for time series
            if 'date' not in df.columns:
                raise KeyError("Expected a 'date' column in the dataframe.")
            df = df.sort_values('date').copy()
            
            # Resolve target column
            if target_col not in df.columns:
                target_col = _resolve_target_column(df, target_col)
            
            # Make target numeric and drop NA
            series = pd.to_numeric(df[target_col], errors='coerce')
            na_count = series.isna().sum()
            if na_count:
                print(f"[!] Warning: {na_count} NA values in '{target_col}' dropped before ARIMA fit.")
            series = series.dropna()
            
            # Attach a DatetimeIndex (helpful for forecast indexing)
            idx = pd.to_datetime(df.loc[series.index, 'date'], errors='coerce')
            series.index = idx
            
            # Optional: try to infer frequency (doesn't modify values)
            try:
                inferred_freq = pd.infer_freq(series.index)
                if inferred_freq:
                    series = series.asfreq(inferred_freq)
            except Exception:
                pass  # If freq can't be inferred, ARIMA still fits; you may see a benign warning.
            
            # Train ARIMA
            self.model = ARIMA(series, order=order).fit()
            
            print(f"✓ Model trained successfully")
            print(f"✓ AIC: {self.model.aic:.2f}")
            print(f"✓ BIC: {self.model.bic:.2f}\n")
            
            return self.model
            
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def train_prophet(self, df, target_col='temp_avg', city='Global'):
        """Train Prophet model"""
        print(f"\n=== TRAINING PROPHET MODEL ===")
        
        try:
            if 'date' not in df.columns:
                raise KeyError("Expected a 'date' column in the dataframe.")
            
            if target_col not in df.columns:
                target_col = _resolve_target_column(df, target_col)
            
            # Prepare data for Prophet
            prophet_df = df[['date', target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.sort_values('ds')
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            na_before = len(prophet_df)
            prophet_df = prophet_df.dropna(subset=['ds', 'y'])
            na_dropped = na_before - len(prophet_df)
            if na_dropped:
                print(f"[!] Warning: Dropped {na_dropped} rows with invalid ds/y for Prophet.")
            
            # Train Prophet (yearly & weekly; daily off)
            self.model = Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            self.model.fit(prophet_df)
            
            print(f"✓ Model trained successfully")
            print(f"✓ Seasonality components added\n")
            
            return self.model
            
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def predict(self, periods=30):
        """Generate forecast"""
        print(f"\n=== GENERATING {periods}-DAY FORECAST ===")
        
        try:
            if self.model_type == 'arima':
                forecast = self.model.get_forecast(steps=periods)
                conf_int = forecast.conf_int()
                forecast_values = forecast.predicted_mean
                
                result = pd.DataFrame({
                    'forecast': forecast_values,
                    'lower_ci': conf_int.iloc[:, 0],
                    'upper_ci': conf_int.iloc[:, 1]
                })
                
            elif self.model_type == 'prophet':
                future = self.model.make_future_dataframe(periods=periods)
                forecast = self.model.predict(future)
                tail = forecast.tail(periods) if periods > 0 else forecast
                result = tail[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                result.columns = ['date', 'forecast', 'lower_ci', 'upper_ci']
            
            print(f"✓ Forecast generated for {periods} days\n")
            return result
            
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model_type == 'prophet':
            # Use Prophet's serialization (pickle is not recommended)
            with open(filepath, 'w') as f:
                f.write(model_to_json(self.model))
            print(f"✓ Prophet model serialized to JSON at {filepath}")
        else:
            # ARIMA can be pickled
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if self.model_type == 'prophet':
            with open(filepath, 'r') as f:
                self.model = model_from_json(f.read())
            print(f"✓ Prophet model loaded from {filepath}")
        else:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {filepath}")

def _resolve_target_column(df, requested: str):
    cols = list(df.columns)
    if requested in cols:
        return requested
    for c in cols:
        if c.lower() == requested.lower():
            return c
    common = ['temp_avg','temperature','temp','tavg','avg_temp','temp_mean','temp_c','avg_temp_c']
    for cand in common:
        for c in cols:
            if c.lower() == cand:
                return c
    close = difflib.get_close_matches(requested, cols, n=1, cutoff=0.6)
    if close:
        return close[0]
    raise ValueError(f"Target column '{requested}' not found. Available columns: {cols}")

if __name__ == "__main__":
    print("Models module ready!")
