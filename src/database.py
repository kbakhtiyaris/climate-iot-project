from sqlalchemy import create_engine, Column, String, Float, Date, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection string
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "climate_iot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class WeatherData(Base):
    """Climate data table with TimescaleDB hypertable"""
    __tablename__ = "weather_data"
    
    id = Column(String, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    city = Column(String, nullable=False, index=True)
    country = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    temp_avg = Column(Float, nullable=False)
    temp_min = Column(Float)
    temp_max = Column(Float)
    humidity = Column(Float)
    precipitation = Column(Float)
    wind_speed = Column(Float)
    pressure = Column(Float)
    
    __table_args__ = (
        Index('idx_city_date', 'city', 'date'),
        Index('idx_date', 'date'),
    )

def init_db():
    """Initialize database and create tables"""
    print("\n=== INITIALIZING DATABASE ===")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✓ Created tables")
        
        # Test connection
        with engine.connect() as conn:
            print("✓ Database connection successful")
            
        print("✓ Database initialized!\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        raise

def get_session():
    """Get database session"""
    return SessionLocal()

if __name__ == "__main__":
    init_db()
