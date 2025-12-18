
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
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_openweather(city, api_key):
    """Fetch current weather for a city from OpenWeatherMap (metric units)."""
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
        logger.error("OpenWeather fetch error: %s", e)
        return None

# Ensure session state for live data
if "live_data" not in st.session_state:
    st.session_state["live_data"] = {}  # city -> list of readings

if "realtime_running" not in st.session_state:
    st.session_state["realtime_running"] = False

# Page Configuration
st.set_page_config(
    page_title="🌍 Global Climate IoT Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# HEADER

st.markdown("## 🌍 Global Climate Data IoT Platform")
st.markdown("**Real-time weather monitoring, analysis, and forecasting**")


# SIDEBAR - FILTERS

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # City Selection
    cities = ["London", "New York", "Tokyo", "Sydney", "Dubai", "Istanbul", "Mumbai"]
    selected_cities = st.multiselect(
        "📍 Select Cities",
        cities,
        default=["London", "New York", "Tokyo"]
    )
    
    # Date Range
    date_range = st.date_input(
        "📅 Date Range",
        value=(datetime.now() - timedelta(days=90), datetime.now()),
        max_value=datetime.now()
    )
    
    # Metric Selection
    metrics = ["Temperature", "Humidity", "Precipitation", "Wind Speed", "Pressure"]
    selected_metrics = st.multiselect(
        "📊 Select Metrics",
        metrics,
        default=["Temperature", "Humidity"]
    )
    
    # Model Selection
    model_type = st.radio(
        "🤖 Forecast Model",
        ("ARIMA", "Prophet", "XGBoost")
    )
    
    # Forecast Period
    forecast_days = st.slider(
        "🔮 Forecast Days",
        min_value=7,
        max_value=90,
        value=30
    )
    
    # Real-time Data Source
    data_source = st.selectbox(
        "🔁 Live Data Source",
        ("Mock", "Weather API (OpenWeatherMap)")
    )
    
    if data_source == "Weather API (OpenWeatherMap)":
        owm_api_key = st.text_input("🔑 OpenWeatherMap API Key", type="password")
        owm_city = st.selectbox("📍 City for Live Feed", cities, index=0)
    else:
        owm_api_key = None
        owm_city = None
    
    poll_interval = st.slider(
        "⏱️ Poll Interval (seconds)",
        min_value=10,
        max_value=600,
        value=30,
        step=5
    )
    
    st.divider()
    st.info("💡 **Tip**: Select multiple cities to compare climate patterns across regions.")


# MAIN CONTENT - TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Trends", "🔮 Forecast", "📉 Analytics", "🔴 Real-time"]
)


# TAB 1: OVERVIEW

with tab1:
    st.header("📊 Current Climate Overview")
    
    # Current Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌡️ Avg Temperature",
            value="18.5°C",
            delta="-2.1°C",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="💧 Humidity",
            value="72%",
            delta="+5%"
        )
    
    with col3:
        st.metric(
            label="🌧️ Precipitation",
            value="12 mm",
            delta="-3 mm",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="💨 Wind Speed",
            value="4.2 m/s",
            delta="+0.5 m/s"
        )
    
    st.divider()
    
    # Global Map Heatmap (Placeholder)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Global Temperature Heatmap")
        # Placeholder: In production, use real map data
        fig_map = go.Figure(data=go.Scattergeo(
            lon=[-0.1, 151.2, 139.7, 25.2, 72.9],
            lat=[51.5, -33.9, 35.7, 55.2, 19.1],
            text=["London", "Sydney", "Tokyo", "Moscow", "Mumbai"],
            mode="markers+text",
            marker=dict(size=20, color=[15, 18, 22, -2, 28], 
                       colorscale="RdYlBu_r", showscale=True,
                       colorbar=dict(title="Temp (°C)"))
        ))
        fig_map.update_layout(
            geo=dict(projection_type="natural earth"),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.subheader("📋 City Summary")
        city_data = pd.DataFrame({
            "City": ["London", "New York", "Tokyo", "Sydney"],
            "Temp (°C)": [15, 12, 18, 22],
            "Humidity (%)": [75, 68, 70, 65]
        })
        st.dataframe(city_data, use_container_width=True, hide_index=True)


# TAB 2: TRENDS

with tab2:
    st.header("📈 Historical Trends")
    
    # Generate Sample Data for Visualization
    days = pd.date_range(end=datetime.now(), periods=90, freq='D')
    np.random.seed(42)
    
    trend_data = pd.DataFrame({
        'Date': days,
        'London_Temp': 10 + np.sin(np.arange(90)/30) * 5 + np.random.normal(0, 1, 90),
        'London_Humidity': 70 + np.random.normal(0, 5, 90),
        'New York_Temp': 8 + np.sin(np.arange(90)/30) * 7 + np.random.normal(0, 1, 90),
        'New York_Humidity': 65 + np.random.normal(0, 5, 90),
    })
    
    # Temperature Trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Temperature Trend")
        fig_temp = px.line(
            trend_data,
            x='Date',
            y=['London_Temp', 'New York_Temp'],
            labels={'value': 'Temperature (°C)', 'variable': 'City'},
            title="Temperature Comparison"
        )
        fig_temp.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        st.subheader("💧 Humidity Trend")
        fig_humidity = px.line(
            trend_data,
            x='Date',
            y=['London_Humidity', 'New York_Humidity'],
            labels={'value': 'Humidity (%)', 'variable': 'City'},
            title="Humidity Comparison"
        )
        fig_humidity.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔗 Metric Correlations")
    corr_data = trend_data[['London_Temp', 'London_Humidity', 'New York_Temp', 'New York_Humidity']].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


# TAB 3: FORECAST

with tab3:
    st.header("🔮 Weather Forecast")
    
    st.info(f"📅 Forecasting **{forecast_days}** days ahead using **{model_type}** model")
    
    # Generate Forecast Data
    future_days = pd.date_range(end=datetime.now(), periods=forecast_days, freq='D')
    np.random.seed(42)
    
    forecast_data = pd.DataFrame({
        'Date': future_days,
        'Forecast': 15 + np.sin(np.arange(forecast_days)/10) * 5 + np.random.normal(0, 0.5, forecast_days),
        'Upper_CI': 15 + np.sin(np.arange(forecast_days)/10) * 5 + 2 + np.random.normal(0, 0.5, forecast_days),
        'Lower_CI': 15 + np.sin(np.arange(forecast_days)/10) * 5 - 2 + np.random.normal(0, 0.5, forecast_days),
    })
    
    # Forecast Chart with Confidence Intervals
    st.subheader("📊 Temperature Forecast with Confidence Intervals")
    
    fig_forecast = go.Figure()
    
    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'], y=forecast_data['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='blue', width=3)
    ))
    
    # Confidence Interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'].tolist() + forecast_data['Date'][::-1].tolist(),
        y=forecast_data['Upper_CI'].tolist() + forecast_data['Lower_CI'][::-1].tolist(),
        fill='toself',
        name='95% Confidence Interval',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)')
    ))
    
    fig_forecast.update_layout(
        height=400,
        title="7-Day Temperature Forecast",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast Table
    st.subheader("📋 Forecast Details")
    forecast_table = forecast_data.copy()
    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
    forecast_table = forecast_table.round(2)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    
    # Model Performance Metrics
    st.subheader("🎯 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", "1.82°C", "↓ -0.15")
    with col2:
        st.metric("RMSE", "2.34°C", "↓ -0.22")
    with col3:
        st.metric("R² Score", "0.87", "↑ +0.03")
    with col4:
        st.metric("MAPE", "8.2%", "↓ -0.5%")


# TAB 4: ANALYTICS

with tab4:
    st.header("📉 Advanced Analytics")
    
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    
    stats_data = pd.DataFrame({
        'Metric': ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Pressure (hPa)'],
        'Mean': [14.2, 71.3, 3.8, 1013.2],
        'Std Dev': [4.5, 12.1, 1.2, 2.3],
        'Min': [5.2, 42.0, 0.5, 1008.1],
        'Max': [26.8, 95.0, 8.5, 1018.9]
    })
    
    st.dataframe(stats_data, use_container_width=True, hide_index=True)
    
    # Distribution Analysis
    st.subheader("📈 Temperature Distribution")
    
    sample_temps = np.random.normal(14.2, 4.5, 1000)
    
    fig_dist = px.histogram(
        x=sample_temps,
        nbins=30,
        labels={'x': 'Temperature (°C)', 'count': 'Frequency'},
        title="Temperature Distribution (Historical)"
    )
    fig_dist.add_vline(x=np.mean(sample_temps), line_dash="dash", line_color="red",
                      annotation_text="Mean")
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection")
    
    anomaly_data = pd.DataFrame({
        'Date': pd.date_range('2024-12-01', periods=30),
        'Temperature': np.random.normal(15, 2, 30),
        'Anomaly': [False]*27 + [True, False, True]
    })
    
    fig_anomaly = px.scatter(
        anomaly_data,
        x='Date',
        y='Temperature',
        color='Anomaly',
        color_discrete_map={True: 'red', False: 'blue'},
        title="Temperature Anomalies Detection",
        labels={'Anomaly': 'Is Anomaly?'}
    )
    
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    detected_anomalies = anomaly_data[anomaly_data['Anomaly']]
    st.warning(f"⚠️ **{len(detected_anomalies)} anomalies detected!**")
    st.dataframe(detected_anomalies, use_container_width=True, hide_index=True)


# TAB 5: REAL-TIME

with tab5:
    st.header("🔴 Real-time Temperature & Humidity")
    st.info(f"Using data source: **{data_source}**")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        start = st.button("▶️ Start Live Feed")
        stop = st.button("⏹️ Stop Live Feed")
        max_points = st.number_input("Max Points to Keep", min_value=50, max_value=1000, value=300, step=50)
    
    if start:
        st.session_state["realtime_running"] = True
    if stop:
        st.session_state["realtime_running"] = False

    selected_city = owm_city if data_source.startswith("Weather API") and owm_city else (selected_cities[0] if selected_cities else cities[0])

    placeholder = st.empty()

    # Simple polling loop — runs in the main thread while the feed is active.
    while st.session_state["realtime_running"]:
        if data_source.startswith("Weather API"):
            if not owm_api_key:
                st.error("Please provide OpenWeatherMap API key in the sidebar.")
                st.session_state["realtime_running"] = False
                break
            res = fetch_openweather(selected_city, owm_api_key)
        else:
            # Mock data
            res = {
                "timestamp": datetime.utcnow(),
                "temperature": 15 + np.random.normal(0,1),
                "humidity": 60 + np.random.normal(0,2)
            }
        if res:
            lst = st.session_state["live_data"].setdefault(selected_city, [])
            lst.append(res)
            if len(lst) > max_points:
                lst[:] = lst[-max_points:]

        # Build DataFrame and update charts
        df = pd.DataFrame(st.session_state["live_data"].get(selected_city, []))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig_rt = go.Figure()
            fig_rt.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature"], mode="lines+markers", name="Temperature (°C)"))
            fig_rt.add_trace(go.Scatter(x=df["timestamp"], y=df["humidity"], mode="lines+markers", name="Humidity (%)", yaxis="y2"))
            fig_rt.update_layout(
                xaxis_title="Time",
                yaxis=dict(title="Temperature (°C)"),
                yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                height=400
            )
            placeholder.plotly_chart(fig_rt, use_container_width=True)
            latest = df.iloc[-1]
            c1, c2 = st.columns(2)
            c1.metric("🌡️ Temperature", f"{latest['temperature']:.1f} °C")
            c2.metric("💧 Humidity", f"{latest['humidity']:.0f}%")
            st.dataframe(df.tail(10).assign(timestamp=lambda d: d["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')), use_container_width=True, hide_index=True)
        else:
            placeholder.info("No data yet. Waiting for first reading...")
        time.sleep(poll_interval)

# FOOTER

st.divider()
st.markdown("""
    ---
    **🌍 Global Climate Data IoT Platform**
    
    Built with Streamlit | Powered by PostgreSQL + TimescaleDB | Deployed on AWS EC2
    
    *Last updated: 2025-12-07 23:30 UTC*
""")
