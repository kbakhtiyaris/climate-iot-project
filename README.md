# Global weather Data IoT Platform 🌍

## Project Overview
- **Course:** Internet of Things and Applied Data Science
- **Instructor:** Dr. Mehmet Ali Akyol
- **University:** Istanbul Gedik University
- **Student:** Khud bakhtiyar Iqbal Sofi, MAZEN IBRAHIM AWAD ABDELHAMID, Abdulrahman Bakouban

## Quick Start

### Local Setup (Ubuntu)
```bash

nano .env  # Add credentials
python scripts/train_models.py
streamlit run dashboards/app.py

## Steps to follow

---


**Checklist:**

```bash
# ✓ Clone fresh repository
git clone https://github.com/YOUR_USERNAME/climate-iot-project.git
cd climate-iot-project

# ✓ Run setup
bash setup.sh

# ✓ Check all files exist
ls src/
ls dashboards/
ls scripts/
ls models/

# ✓ Verify database works
python -c "from src.database import init_db; print('✓ Database OK')"

# ✓ Test models
python scripts/train_models.py

# ✓ Test dashboard
streamlit run dashboards/app.py

