
# 🌍 Global Climate IoT Weather Platform

An end‑to‑end **IoT‑style weather analytics and forecasting platform** that ingests global daily climate data, stores it in a time‑series database, trains forecasting models, and exposes interactive visualizations and predictions through a Streamlit dashboard deployed on a public domain.

**Live demo:** https://weather-iot.duckdns.org

> **Tech stack:** Python, Pandas, PostgreSQL, TimescaleDB, Streamlit, ARIMA/Prophet, AWS EC2, DuckDNS, Nginx.

***

## 1. Project Overview

This project simulates an IoT weather data platform using a large‑scale public climate dataset instead of physical sensors. It focuses on:

- Collecting and organizing **global daily weather observations**.
- Treating each city as a **virtual IoT node** producing time‑series data.
- Building **forecasting models** for future temperature.
- Visualizing historical trends and forecasts in an **interactive web dashboard**.
- Hosting the app on **AWS EC2** behind **Nginx**, reachable via a **DuckDNS domain**: `weather-iot.duckdns.org`.[1][2]

The project is built as a full pipeline: **data → database → models → dashboard → cloud deployment**.

***

## 2. Problem Statement

- **We want to:** predict future daily average temperature for major global cities.[3][2]
- **System/context:** a cloud‑hosted IoT‑style analytics platform where cities act as logical sensors.[3]
- **Because it impacts:**  
  - Energy demand planning (heating/cooling).[4]
  - Agriculture and irrigation decisions.[5]
  - Climate‑risk and comfort analytics for cities and utilities.[3]
- **Using data from:** a public global daily climate dataset on Kaggle, treated as historical sensor readings, at **daily granularity**.[2][6]
- **Success looks like:**  
  - Mean Absolute Error (MAE) ≤ 2 °C.  
  - Root Mean Squared Error (RMSE) ≤ 2.5 °C.  
  - R² ≥ 0.85 on held‑out data for target cities.[7][8]
- **Constraints:**  
  - Must run on a single **AWS t2.micro Free‑Tier** instance.[1]
  - Storage and bandwidth limits for a multi‑year global dataset.[9]
  - Latency low enough for interactive dashboard use.[10]

***

## 3. Architecture

End‑to‑end architecture:

1. **Data Source**
   - Kaggle: *Global Daily Climate / “The Weather Dataset”* – multiyear daily weather data for many locations.[2]
2. **Ingestion & Processing**
   - Python, Pandas for ingesting CSV files and performing ETL (cleaning, interpolation, outlier removal, feature engineering).[11]
3. **Storage Layer**
   - **PostgreSQL** as the main relational database.  
   - **TimescaleDB** extension to optimize time‑series queries on weather data.[3]
4. **Modeling Layer**
   - Time‑series forecasting with **ARIMA** and **Prophet** for daily average temperature.[7]
5. **Application Layer**
   - **Streamlit** dashboard for:
     - City‑level historical charts.
     - Global summaries.
     - Forecast visualization and metrics.[1]
6. **Deployment Layer**
   - **AWS EC2** Ubuntu instance (Free Tier).[1]
   - **Nginx** as reverse proxy.[12]
   - **DuckDNS** for a free dynamic DNS domain: `weather-iot.duckdns.org`.[13]

***

## 4. Features

### 4.1 Data & Analytics

- Global daily weather observations (temperature, humidity, pressure, wind, precipitation) over multiple years.[11][2]
- Cleaning pipeline:
  - Date parsing and standardization.
  - Missing‑value interpolation for numeric features.
  - Outlier removal using IQR (per feature).[11]
- Aggregations:
  - City‑level and country‑level summaries.
  - Daily and multi‑day rolling statistics.

### 4.2 Machine Learning

- Forecasts **daily average temperature** for selected cities.[3]
- Models:
  - **ARIMA** – classical time‑series model tuned for short‑horizon forecasts.[7]
  - **Prophet** – captures seasonality and holiday‑like patterns.[7]
- Evaluation metrics:
  - MAE, RMSE, R², and MAPE.[7]

### 4.3 Streamlit Dashboard (weather‑iot.duckdns.org)

- Sidebar filters:
  - City selection.
  - Date range.
  - Metric (temperature, humidity, precipitation, wind, pressure).
  - Model type and forecast horizon.
- Tabs (example layout):
  - **Overview** – cards with current averages, min/max, anomaly flags.  
  - **Historical Trends** – multi‑city line plots and correlations.  
  - **Forecasts** – forecast curves with confidence intervals and numeric tables.  
  - **Analytics** – distributions, anomaly detection, and basic feature analysis.

---

## 5. Repository Structure

```text
climate-iot-project/
├── src/
│   ├── config.py          # Configuration (env vars, paths)
│   ├── data_loader.py     # Kaggle/data ingestion
│   ├── data_processing.py # Cleaning, feature engineering
│   ├── database.py        # PostgreSQL/TimescaleDB models & session(couldn't use TimescaleDb on Ubuntu, so i used it by cloning TimescaleDb directly to my repo. from timescaledb github repository, u can easy set it up if u r using windows or mac)
│   ├── models.py          # ARIMA / Prophet training & prediction
├── dashboards/
│   └── app.py             # Streamlit dashboard
├── scripts/
│   ├── load_data.py       # Load cleaned CSV into PostgreSQL
│   └── train_models.py    # Train and evaluate models
├── data/
│   ├── raw/               # Raw Kaggle downloads (git‑ignored,(my actual dataset is in parquet format and its a large file, so just uploaded .csv sample dataset )
│   └── processed/         # Cleaned data 
├── models/                # Saved model artefacts (.pkl(provided in zip format), git‑ignored)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

***

## 6. Getting Started (Local – Ubuntu)

### 6.1 Prerequisites

- Python 3.10+  
- PostgreSQL 14+  
- (Optional) TimescaleDB extension for PostgreSQL.[3]
- Git  
- Kaggle account + API token (kaggle.json).[2]

### 6.2 Clone and Set Up

```bash
# Clone
git clone https://github.com/kbakhtiyaris/climate-iot-project.git
cd climate-iot-project

# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template
cp .env.example .env
```

Edit `.env` with your DB credentials and (optionally) Kaggle keys.

***

## 7. Database Setup

1. **Create PostgreSQL user and database:**

```bash
sudo -u postgres psql

CREATE USER iot_user WITH ENCRYPTED PASSWORD 'your_password';
CREATE DATABASE climate_iot OWNER iot_user;
GRANT ALL PRIVILEGES ON DATABASE climate_iot TO iot_user;
\q
```

2. **Enable TimescaleDB** (if installed):

```bash
psql -U iot_user -d climate_iot -h localhost

CREATE EXTENSION IF NOT EXISTS timescaledb;
\dx   -- confirm it's listed
\q
```

3. **Initialize schema from code:**

```bash
source venv/bin/activate
python -c "from src.database import init_db; init_db()"
```

This creates the `weather_data` table and indexes.

***

## 8. Data Ingestion & Cleaning

### 8.1 Download Dataset

If you use Kaggle programmatically:

```bash
# Ensure kaggle.json is in ~/.kaggle and .env has KAGGLE_USERNAME / KAGGLE_KEY
source venv/bin/activate
python -c "from src.data_loader import download_kaggle_dataset; download_kaggle_dataset()"
```

Alternatively, manually download the CSV from the Kaggle page and put it in `data/raw/`.[2]

### 8.2 Inspect & Clean

```bash
source venv/bin/activate

# Quick data profile
python -c "from src.data_loader import load_weather_data; df = load_weather_data(); print(df.head()); print(df.info())"

# Clean and save to data/processed/
python src/data_processing.py
```

This step handles missing values, outliers, and basic normalization.

### 8.3 Load Into PostgreSQL

```bash
source venv/bin/activate
python scripts/load_data.py
```

Afterwards you can verify:

```bash
psql -U iot_user -d climate_iot -h localhost
SELECT COUNT(*) FROM weather_data;
SELECT city, MIN(date), MAX(date) FROM weather_data GROUP BY city LIMIT 5;
\q
```

***

## 9. Model Training

Train ARIMA/Prophet models on the cleaned dataset:

```bash
source venv/bin/activate
python scripts/train_models.py
```

This script:

- Splits train/test (e.g., 80/20 by time).
- Trains ARIMA.
- Trains Prophet.
- Logs MAE, RMSE, R² for each model.
- Saves model artefacts into `models/`.

You can adapt `scripts/train_models.py` to focus on specific cities or to tune hyperparameters.

***

## 10. Running the Dashboard Locally

```bash
source venv/bin/activate
streamlit run dashboards/app.py
```

Open in your browser:

- http://localhost:8501

Use the sidebar to choose cities, date ranges, metrics, and forecast horizons.

***

## 11. Deployment on AWS EC2 + DuckDNS

### 11.1 High‑Level Steps

Full details are in `docs/AWS_DEPLOYMENT.md`, but the main flow is:

1. Launch an Ubuntu t2.micro instance on AWS EC2 (Free Tier).[1]
2. SSH into the instance and install:
   - Python, virtualenv.
   - PostgreSQL (+ TimescaleDB if needed).
   - Git.
   - Nginx.
3. Clone this repo and repeat the virtualenv + `pip install -r requirements.txt`.[1]
4. Configure the same database schema and load a smaller or full dataset depending on RAM.  
5. Run Streamlit with `--server.address 0.0.0.0` and `--server.port 8501`.  
6. Configure Nginx as a reverse proxy for `weather-iot.duckdns.org` → `localhost:8501`.[12][1]
7. Configure your DuckDNS domain to point to the EC2 public IP and create a CNAME/AAAA record accordingly.[13]
8. Optionally enable HTTPS using Let’s Encrypt if you terminate TLS on Nginx.[14]

At the end, your app is reachable at:

- `https://weather-iot.duckdns.org/`

***

## 12. Example Use Cases

- Compare temperature trends between multiple cities over several years.[11]
- Estimate short‑term future temperatures for energy‑planning scenarios.[4]
- Explore extreme weather events and anomalies.[3]
- Use the platform as a template to later plug in **real ESP32 / sensor data** instead of only Kaggle data.

***

## 13. Future Work

- Integrate real IoT sensors (ESP32, Raspberry Pi) streaming to the same PostgreSQL/TimescaleDB instance via MQTT or HTTP.[3]
- Add more advanced models (LSTM/Temporal Fusion Transformers) for longer‑horizon forecasts.[15]
- Implement user authentication and role‑based dashboards.[10]
- Add alerting for threshold‑based events (e.g., heatwaves, cold waves).[16]

***

## 14. Acknowledgements

- **Dataset:** “The Weather Dataset / Global Daily Climate Data” on Kaggle.[2]
- **Inspiration:** Academic and industrial work on IoT‑based weather monitoring and prediction.[17][3]
- **Deployment references:** Community guides on hosting Streamlit with custom domains, Nginx, and DuckDNS.[14][12][1]

***
Screenshots from the dashboard:

