# dashboards/app.py
"""
Streamlit Dashboard for Global Climate Data IoT Platform
Real-time visualization and forecasting of global weather patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import os
import hashlib
from typing import List, Optional
from pathlib import Path

import requests
import time

# -------- Secrets & Env --------
# Robust .env loading regardless of subfolder location
try:
    from dotenv import load_dotenv
    # project root assumed as parent of dashboards/
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")
except Exception:
    pass

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Page Configuration --------
st.set_page_config(
    page_title="🌍 Global Climate IoT Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Safe key resolver --------
def get_openweather_api_key() -> str:
    """Resolve OWM API key from Streamlit secrets (if present), env vars (.env), or session override without crashing."""
    key = ""
    # 1) Try secrets.toml ONLY if available
    try:
        if "OPENWEATHER_API_KEY" in st.secrets:
            key = st.secrets["OPENWEATHER_API_KEY"]
    except Exception:
        # No secrets file; ignore
        pass
    # 2) Fall back to environment (.env or OS env)
    if not key:
        key = os.getenv("OPENWEATHER_API_KEY", "")
    # 3) Session override (e.g., Advanced override in sidebar)
    if st.session_state.get("owm_api_key"):
        key = st.session_state["owm_api_key"]
    # Persist resolved key for later use
    if key and st.session_state.get("owm_api_key") != key:
        st.session_state["owm_api_key"] = key
    return key or ""

# -------- API calls --------
def _fetch_openweather(city: str, api_key: str):
    """Raw fetch for real-time polling (no cache)."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "units": "metric", "appid": api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "timestamp": datetime.utcfromtimestamp(data.get("dt", datetime.utcnow().timestamp())),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "raw": data
        }
    except Exception as e:
        logger.warning(f"OpenWeather fetch error for {city}: {e}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_openweather_snapshot(city: str, api_key: str):
    """Fetch one snapshot for a city (metric units), cached for 5 min."""
    return _fetch_openweather(city, api_key)

# -------- Synthetic helpers --------
def _city_seed(city: str) -> int:
    """Stable seed per city for reproducible synthetic series."""
    return int(hashlib.sha256(city.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)

def get_city_snapshots(cities: List[str], api_key: Optional[str]) -> pd.DataFrame:
    """
    Return a DataFrame with current snapshots for each city:
    columns: city, timestamp, temperature, humidity, lon, lat, source
    Uses OpenWeather if api_key provided; otherwise returns synthetic mock.
    Cached snapshots via fetch_openweather_snapshot() to avoid rate limits.
    """
    rows = []
    for city in cities:
        if api_key:
            snap = fetch_openweather_snapshot(city, api_key)
            if snap:
                raw = snap.get("raw", {})
                coord = raw.get("coord", {})
                rows.append({
                    "city": city,
                    "timestamp": snap["timestamp"],
                    "temperature": snap["temperature"],
                    "humidity": snap.get("humidity"),
                    "lon": coord.get("lon"),
                    "lat": coord.get("lat"),
                    "source": "OpenWeather"
                })
                continue  # next city

        # Fallback synthetic snapshot
        rng = np.random.default_rng(_city_seed(city))
        base_temp = 12 + rng.normal(0, 2) + ((hash(city) % 7) - 3)
        base_hum = 60 + rng.normal(0, 5)
        rows.append({
            "city": city,
            "timestamp": datetime.utcnow(),
            "temperature": round(base_temp, 1),
            "humidity": int(np.clip(base_hum, 25, 95)),
            "lon": None,
            "lat": None,
            "source": "Mock"
        })

    return pd.DataFrame(rows)

def make_synthetic_trend_data(cities: List[str], days: int = 90) -> pd.DataFrame:
    """
    Create synthetic historical trend data for the chosen cities.
    Returns a long-form DataFrame: Date, city, temp, humidity
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    all_rows = []
    for city in cities:
        seed = _city_seed(city)
        rng = np.random.default_rng(seed)
        temp_base = 12 + (seed % 9)  # 12..20
        hum_base = 55 + (seed % 15)  # 55..70
        temp_series = temp_base + 5*np.sin(np.arange(days)/30) + rng.normal(0, 1.0, days)
        hum_series = np.clip(hum_base + rng.normal(0, 4.0, days), 25, 95)
        for d, t, h in zip(dates, temp_series, hum_series):
            all_rows.append({"Date": d, "city": city, "temp": float(t), "humidity": float(h)})
    return pd.DataFrame(all_rows)

def make_synthetic_forecast(city: str, days: int) -> pd.DataFrame:
    """
    Produce a synthetic forecast for one city with CIs, long-form DataFrame
    columns: Date, Forecast, Upper_CI, Lower_CI
    """
    seed = _city_seed(city)
    rng = np.random.default_rng(seed)
    future_days = pd.date_range(end=datetime.now(), periods=days, freq="D")
    baseline = 14 + (seed % 8)  # 14..21
    core = baseline + 5*np.sin(np.arange(days)/10) + rng.normal(0, 0.6, days)
    upper = core + 2 + rng.normal(0, 0.3, days)
    lower = core - 2 + rng.normal(0, 0.3, days)
    return pd.DataFrame({"Date": future_days, "Forecast": core, "Upper_CI": upper, "Lower_CI": lower})

# -------- Session State --------
if "live_data" not in st.session_state:
    st.session_state["live_data"] = {}  # city -> list of readings

if "realtime_running" not in st.session_state:
    st.session_state["realtime_running"] = False

if "extra_cities" not in st.session_state:
    st.session_state["extra_cities"] = []

# Base cities (global, reused across tabs)
BASE_CITIES = [
    "London", "New York", "Tokyo", "Sydney", "Dubai", "Istanbul", "Mumbai",
    "Paris", "Berlin", "Los Angeles", "San Francisco", "Singapore", "Beijing",
    "Seoul", "Cairo", "São Paulo", "Mexico City", "Johannesburg", "Toronto",
    "Bangkok", "Riyadh", "Buenos Aires", "Amsterdam", "Barcelona", "Moscow"
]

# -------- Minimal CSS --------
st.markdown("""
<style>
.metric-card {
    background-color: #f6f8fc;
    padding: 16px;
    border-radius: 12px;
    margin: 8px 0;
    border: 1px solid #e6e9ef;
}
.header-title {
    color: #1f77b4;
    font-size: 28px;
    font-weight: 700;
}
.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -------- Header --------
st.markdown("## 🌍 Global Climate Data IoT Platform")
st.markdown("**Real-time weather monitoring, analysis, and forecasting**")
st.markdown(f"<p class='small-muted'>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>", unsafe_allow_html=True)

# -------- Sidebar --------
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Display ---
    st.caption("🎨 Display")
    plotly_template = st.selectbox(
        "Plotly Theme",
        options=["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        index=0,
        help="Choose a visual style for all charts."
    )

    px.defaults.template = plotly_template

    # --- Cities & Filters ---
    with st.expander("📍 Cities & Filters", expanded=True):
        cities_all = BASE_CITIES + st.session_state.get("extra_cities", [])
        add_col, btn_col = st.columns([0.7, 0.3])
        with add_col:
            new_city = st.text_input("➕ Add City", placeholder="e.g., Ankara")
        with btn_col:
            if st.button("Add"):
                if new_city:
                    if new_city not in st.session_state["extra_cities"]:
                        st.session_state["extra_cities"].append(new_city)
                        st.success(f"Added city: {new_city}")
                    else:
                        st.info(f"City already in list: {new_city}")

        select_all_btn = st.button("Select All Cities")
        default_selection = st.session_state.get("selected_cities", ["Istanbul", "New York", "Tokyo"])
        if select_all_btn:
            default_selection = cities_all.copy()
            st.session_state["selected_cities"] = default_selection

        selected_cities = st.multiselect(
            "📍 Select Cities",
            cities_all,
            default=default_selection,
            help="Choose cities to include in overview and trend analysis."
        )

        date_range = st.date_input(
            "📅 Date Range",
            value=(datetime.now() - timedelta(days=90), datetime.now()),
            max_value=datetime.now()
        )

        metrics = ["Temperature", "Humidity", "Precipitation", "Wind Speed", "Pressure"]
        selected_metrics = st.multiselect(
            "📊 Select Metrics",
            metrics,
            default=["Temperature", "Humidity"]
        )

    # --- Forecast Settings ---
    with st.expander("🔮 Forecast Settings", expanded=False):
        model_type = st.radio(
            "🤖 Forecast Model",
            ("ARIMA", "Prophet", "XGBoost"),
            help="Choose the model used for generating synthetic forecasts."
        )
        forecast_days = st.slider(
            "🔮 Forecast Days",
            min_value=7,
            max_value=90,
            value=30
        )

    # --- Live Data ---
    with st.expander("🔴 Live Data", expanded=False):
        data_source = st.selectbox(
            "🔁 Live Data Source",
            ("Mock", "Weather API (OpenWeatherMap)")
        )

        # Key is auto-resolved. Advanced override optional.
        show_advanced = st.checkbox("Advanced: Override API key", value=False)
        if show_advanced:
            owm_api_key_override = st.text_input("🔑 OpenWeatherMap API Key (override)", value="", type="password")
            if owm_api_key_override:
                st.session_state["owm_api_key"] = owm_api_key_override

        # City for default live feed
        owm_city = st.selectbox("📍 City for Live Feed", cities_all, index=0)

        poll_interval = st.slider(
            "⏱️ Poll Interval (seconds)",
            min_value=10,
            max_value=600,
            value=30,
            step=5
        )

        st.info("💡 Tip: Polling many cities frequently may hit rate limits—use a longer interval.")

# -------- Tabs --------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Trends", "🔮 Forecast", "📉 Analytics", "🔴 Real-time"]
)

# -------- TAB 1: OVERVIEW --------
with tab1:
    st.header("📊 Current Climate Overview")
    resolved_owm_key = get_openweather_api_key()

    overview_cities = selected_cities if selected_cities else ["London", "New York", "Tokyo"]
    snaps_df = get_city_snapshots(overview_cities, api_key=resolved_owm_key)
    snaps_df_display = snaps_df.copy()
    snaps_df_display["timestamp"] = pd.to_datetime(snaps_df_display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    avg_temp = snaps_df["temperature"].mean().round(1) if not snaps_df.empty else np.nan
    avg_hum = snaps_df["humidity"].mean().round(1) if not snaps_df.empty else np.nan
    # Placeholder values for metrics you haven't wired yet
    avg_precip = 12
    avg_wind = 4.2

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌡️ Avg Temperature (Selected Cities)", f"{avg_temp}°C" if not np.isnan(avg_temp) else "N/A")
    with c2:
        st.metric("💧 Avg Humidity (Selected Cities)", f"{avg_hum}%" if not np.isnan(avg_hum) else "N/A")
    with c3:
        st.metric("🌧️ Precipitation (placeholder)", f"{avg_precip} mm")
    with c4:
        st.metric("💨 Wind Speed (placeholder)", f"{avg_wind} m/s")

    st.caption(f"Showing data for: {', '.join(overview_cities)}")

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🗺️ Temperature Map (Selected Cities)")
        map_df = snaps_df.dropna(subset=["lon", "lat"])
        if not map_df.empty:
            fig_map = go.Figure(data=go.Scattergeo(
                lon=map_df["lon"], lat=map_df["lat"],
                text=[f"{c}: {t}°C" for c, t in zip(map_df["city"], map_df["temperature"])],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color=map_df["temperature"],
                    colorscale="RdYlBu_r",
                    showscale=True,
                    colorbar=dict(title="Temp (°C)")
                )
            ))
            fig_map.update_layout(
                geo=dict(projection_type="natural earth"),
                height=420,
                margin=dict(l=0, r=0, t=0, b=0),
                template=plotly_template
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Coordinates unavailable (mock data in use). Provide OpenWeather API key to enable map coordinates.")

    with col2:
        st.subheader("📋 City Summary (Selected)")
        st.dataframe(
            snaps_df_display[["city", "temperature", "humidity", "timestamp", "source"]],
            use_container_width=True, hide_index=True
        )

# -------- TAB 2: TRENDS --------
with tab2:
    st.header("📈 Historical Trends")

    trend_cities = st.multiselect(
        "Select cities for trends",
        options=BASE_CITIES + st.session_state.get("extra_cities", []),
        default=selected_cities[:2] if selected_cities else ["London", "New York"],
        help="Choose which cities to include in the trend charts."
    )

    if not trend_cities:
        st.info("Please select at least one city to show trends.")
    else:
        trend_df = make_synthetic_trend_data(trend_cities, days=90)

        # Temperature
        st.subheader("🌡️ Temperature Trend")
        fig_temp = px.line(
            trend_df, x="Date", y="temp", color="city",
            labels={"temp": "Temperature (°C)", "city": "City"},
            title="Temperature Comparison",
            template=plotly_template
        )
        fig_temp.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_temp, use_container_width=True)

        # Humidity
        st.subheader("💧 Humidity Trend")
        fig_hum = px.line(
            trend_df, x="Date", y="humidity", color="city",
            labels={"humidity": "Humidity (%)", "city": "City"},
            title="Humidity Comparison",
            template=plotly_template
        )
        fig_hum.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_hum, use_container_width=True)

        # Correlation across selected cities (temperature only for simplicity)
        st.subheader("🔗 Metric Correlations (Selected Cities — Temperature)")
        corr_wide = trend_df.pivot(index="Date", columns="city", values="temp")
        corr = corr_wide.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_corr.update_layout(height=420, template=plotly_template)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(f"Trends shown for: {', '.join(trend_cities)}")

# -------- TAB 3: FORECAST --------
with tab3:
    st.header("🔮 Weather Forecast")

    all_options = BASE_CITIES + st.session_state.get("extra_cities", [])
    # Pick the first selected city if available; else index 0
    default_city = selected_cities[0] if selected_cities else all_options[0]
    try:
        default_index = all_options.index(default_city)
    except ValueError:
        default_index = 0

    forecast_city = st.selectbox(
        "City for Forecast",
        options=all_options,
        index=default_index
    )

    st.info(f"📅 Forecasting **{forecast_days}** days ahead for **{forecast_city}** using **{model_type}**")

    forecast_data = make_synthetic_forecast(forecast_city, forecast_days)

    st.subheader(f"📊 Temperature Forecast with Confidence Intervals — {forecast_city}")
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'], y=forecast_data['Forecast'],
        mode='lines', name='Forecast', line=dict(color='blue', width=3)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'].tolist() + forecast_data['Date'][::-1].tolist(),
        y=forecast_data['Upper_CI'].tolist() + forecast_data['Lower_CI'][::-1].tolist(),
        fill='toself', name='95% Confidence Interval',
        fillcolor='rgba(0,100,200,0.2)', line=dict(color='rgba(255,255,255,0)')
    ))
    fig_forecast.update_layout(
        height=420,
        title=f"{forecast_days}-Day Temperature Forecast — {forecast_city}",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template=plotly_template
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📋 Forecast Details")
    forecast_table = forecast_data.copy()
    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
    forecast_table = forecast_table.round(2)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=forecast_table.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{forecast_city}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.subheader("🎯 Model Performance Metrics (synthetic)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("MAE", "1.82°C", "↓ -0.15")
    with c2: st.metric("RMSE", "2.34°C", "↓ -0.22")
    with c3: st.metric("R² Score", "0.87", "↑ +0.03")
    with c4: st.metric("MAPE", "8.2%", "↓ -0.5%")

# -------- TAB 4: ANALYTICS --------
with tab4:
    st.header("📉 Advanced Analytics")

    all_options = BASE_CITIES + st.session_state.get("extra_cities", [])
    default_city = selected_cities[0] if selected_cities else all_options[0]
    try:
        default_index = all_options.index(default_city)
    except ValueError:
        default_index = 0

    analytics_city = st.selectbox(
        "City for Analytics",
        options=all_options,
        index=default_index
    )
    st.caption(f"Showing analytics for: **{analytics_city}**")

    analytics_df = make_synthetic_trend_data([analytics_city], days=180)

    st.subheader(f"📊 Statistical Summary — {analytics_city}")
    stats_data = pd.DataFrame({
        'Metric': ['Temperature (°C)', 'Humidity (%)'],
        'Mean': [analytics_df['temp'].mean().round(2), analytics_df['humidity'].mean().round(2)],
        'Std Dev': [analytics_df['temp'].std().round(2), analytics_df['humidity'].std().round(2)],
        'Min': [analytics_df['temp'].min().round(2), analytics_df['humidity'].min().round(2)],
        'Max': [analytics_df['temp'].max().round(2), analytics_df['humidity'].max().round(2)]
    })
    st.dataframe(stats_data, use_container_width=True, hide_index=True)

    st.subheader(f"📈 Temperature Distribution — {analytics_city}")
    fig_dist = px.histogram(
        x=analytics_df['temp'],
        nbins=30,
        labels={'x': 'Temperature (°C)', 'count': 'Frequency'},
        title=f"Temperature Distribution (Historical) — {analytics_city}",
        template=plotly_template
    )
    fig_dist.add_vline(x=float(analytics_df['temp'].mean()), line_dash="dash", line_color="red",
                      annotation_text="Mean")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader(f"🚨 Anomaly Detection — {analytics_city}")
    mu, sigma = float(analytics_df['temp'].mean()), float(analytics_df['temp'].std())
    analytics_df["Anomaly"] = np.abs(analytics_df["temp"] - mu) > 2.5 * sigma
    fig_anomaly = px.scatter(
        analytics_df,
        x='Date',
        y='temp',
        color='Anomaly',
        color_discrete_map={True: 'red', False: 'blue'},
        title=f"Temperature Anomalies Detection — {analytics_city}",
        labels={'temp': 'Temperature (°C)', 'Anomaly': 'Is Anomaly?'},
        template=plotly_template
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

    detected_anomalies = analytics_df[analytics_df['Anomaly']]
    st.warning(f"⚠️ **{len(detected_anomalies)} anomalies detected for {analytics_city}!**")
    st.dataframe(
        detected_anomalies[['Date', 'temp']].assign(
            Date=lambda d: pd.to_datetime(d["Date"]).dt.strftime('%Y-%m-%d')),
        use_container_width=True, hide_index=True
    )

# -------- TAB 5: REAL-TIME --------
with tab5:
    st.header("🔴 Real-time Temperature & Humidity")

    resolved_owm_key = get_openweather_api_key()  # ensure latest
    if data_source.startswith("Weather API"):
        if not resolved_owm_key:
            st.error("OpenWeatherMap API key not found. Add it to `.env` or `.streamlit/secrets.toml`, or use 'Advanced: Override API key' in the sidebar.")
        else:
            st.success("OpenWeatherMap API key loaded automatically.")

    ui_cols = st.columns([2, 1])
    with ui_cols[1]:
        start = st.button("▶️ Start Live Feed")
        stop = st.button("⏹️ Stop Live Feed")
        max_points = st.number_input("Max Points to Keep", min_value=50, max_value=1000, value=300, step=50)

    if start:
        st.session_state["realtime_running"] = True
    if stop:
        st.session_state["realtime_running"] = False

    selected_live_cities = st.multiselect(
        "📍 Cities for Live Feed",
        options=BASE_CITIES + st.session_state.get("extra_cities", []),
        default=[owm_city] if owm_city else (selected_cities[:3] if selected_cities else [BASE_CITIES[0]])
    )

    st.info(f"Using data source: **{data_source}** | Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC")

    placeholder_chart = st.empty()
    placeholder_table = st.empty()
    placeholder_msg = st.empty()

    # Polling loop
    while st.session_state["realtime_running"]:
        if not selected_live_cities:
            placeholder_msg.info("No cities selected for live feed.")
            time.sleep(1)
            continue

        all_latest = []
        for city in selected_live_cities:
            if data_source.startswith("Weather API"):
                if not resolved_owm_key:
                    continue
                res = _fetch_openweather(city, resolved_owm_key)
                if res is None:
                    time.sleep(1.5)  # simple backoff
                    continue
            else:
                res = {
                    "timestamp": datetime.utcnow(),
                    "temperature": 15 + np.random.normal(0,1),
                    "humidity": 60 + np.random.normal(0,2)
                }

            if res:
                lst = st.session_state["live_data"].setdefault(city, [])
                lst.append(res)
                if len(lst) > max_points:
                    lst[:] = lst[-max_points:]
                all_latest.append({
                    "city": city,
                    "timestamp": res["timestamp"],
                    "temperature": res["temperature"],
                    "humidity": res.get("humidity")
                })

        # Build combined DataFrame and update charts
        if all_latest:
            df_all = []
            for city in selected_live_cities:
                city_df = pd.DataFrame(st.session_state["live_data"].get(city, []))
                if not city_df.empty:
                    city_df["timestamp"] = pd.to_datetime(city_df["timestamp"])
                    city_df["city"] = city
                    df_all.append(city_df[["timestamp", "city", "temperature", "humidity"]])
            if df_all:
                df_combined = pd.concat(df_all, ignore_index=True)
                fig_rt = go.Figure()
                for city in selected_live_cities:
                    city_df = df_combined[df_combined["city"] == city]
                    if not city_df.empty:
                        fig_rt.add_trace(go.Scatter(
                            x=city_df["timestamp"], y=city_df["temperature"],
                            mode="lines+markers", name=f"{city} - Temp"
                        ))
                        fig_rt.add_trace(go.Scatter(
                            x=city_df["timestamp"], y=city_df["humidity"],
                            mode="lines+markers", name=f"{city} - Hum", yaxis="y2"
                        ))
                fig_rt.update_layout(
                    xaxis_title="Time",
                    yaxis=dict(title="Temperature (°C)"),
                    yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                    height=420,
                    template=plotly_template,
                    legend_title="Series"
                )
                placeholder_chart.plotly_chart(fig_rt, use_container_width=True)

                latest_df = pd.DataFrame(all_latest).sort_values("city").set_index("city")
                latest_df = latest_df.assign(
                    timestamp=lambda d: pd.to_datetime(d["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                )
                placeholder_table.table(latest_df[["timestamp", "temperature", "humidity"]])
        else:
            placeholder_msg.info("No data yet. Waiting for first reading...")

        time.sleep(poll_interval)

# -------- Footer --------
st.divider()
st.markdown("""
    ---
    **🌍 Global Climate Data IoT Platform**

    Built with Streamlit | Powered by PostgreSQL + TimescaleDB | Deployed on AWS EC2

    **Last updated: 2025-12-23 23:30 UTC*
	Credit: Khud Bakhtiyar Iqbal Sofi
		co-author:Abadul Rahman
		co-author:Mazen Ibrahem
    """, unsafe_allow_html=True)
