import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# --- MLOps: Windows/Intel OpenMP Fix ---
# This prevents the 'libiomp5md.dll' crash on Windows machines
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Settings(BaseSettings):
    # --- Project Metadata ---
    PROJECT_NAME: str = "Stock-AI"
    VERSION: str = "1.0.0"
    ENV: str = "development" # Change to 'production' on GCP
    
    # --- API Keys (Must exist in your .env file) ---
    SUPABASE_URL: str
    SUPABASE_KEY: str
    GEMINI_API_KEY: str
    ALPHA_VANTAGE_KEY: str
    FINNHUB_KEY: str
    FMP_KEY: str
    
    # --- RL Hyperparameters (Centralized for MLOps) ---
    INITIAL_BALANCE: float = 10000.0
    TRANSACTION_FEE: float = 0.001  # 0.1% per trade
    TIMESTEPS: int = 50000
    
    # --- Database Settings ---
    MAX_DB_RETRIES: int = 3
    
    # Automatically look for a .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env")

# Initialize the global settings object
settings = Settings()

# Optional: Print a success message for debugging
if __name__ == "__main__":
    print(f"✅ Configuration for {settings.PROJECT_NAME} loaded successfully.")