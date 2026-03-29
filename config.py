import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Technical Fix for HP Omen/Windows: Prevents library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Settings(BaseSettings):
    # --- 🏗️ INFRASTRUCTURE ---
    PROJECT_NAME: str = "Stock-AI-Agent"
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # --- 🧠 AI BRAINS (LLM APIs) ---
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    HF_API_KEY: Optional[str] = None       # Hugging Face
    COHERE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: str

    # --- 📈 MARKET DATA (News & Price) ---
    ALPHA_VANTAGE_KEY: str
    FINNHUB_KEY: str
    FMP_KEY: str

    # --- ⚙️ BOT LOGIC SETTINGS ---
    # Max stocks to watch at once (to save API credits)
    MAX_WATCHLIST_SIZE: int = 10
    
    # Consensus Threshold (How much LLMs must agree)
    CONSENSUS_THRESHOLD: float = 0.3
    
    # Global Panic Threshold (0-10)
    PANIC_THRESHOLD: int = 7

    # This tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Initialize the settings object
settings = Settings()

if __name__ == "__main__":
    # Test print (hiding actual keys for safety)
    print(f"✅ Configuration Loaded for: {settings.PROJECT_NAME}")
    print(f"🔗 Supabase URL: {settings.SUPABASE_URL}")
    print(f"🤖 Primary LLM: OpenRouter (Nemotron)")
