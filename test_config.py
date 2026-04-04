from config import settings
from loguru import logger

def test_config_loading():
    print("\n--- 🕵️ Pydantic Configuration Test ---")
    
    try:
        # 1. Check if Supabase is loaded
        if settings.SUPABASE_URL and "supabase.co" in settings.SUPABASE_URL:
            logger.success(f"Supabase URL detected: {settings.SUPABASE_URL[:20]}...")
        else:
            logger.error("Supabase URL looks wrong or is missing.")

        # 2. Check a few Critical AI Keys (printing only first 5 chars for safety)
        keys_to_check = {
            "Gemini": settings.GEMINI_API_KEY,
            "Groq": settings.GROQ_API_KEY,
            "OpenRouter": settings.OPENROUTER_API_KEY,
            "Finnhub": settings.FINNHUB_KEY
        }

        for name, key in keys_to_check.items():
            if key and len(key) > 5:
                logger.success(f"{name} Key Loaded: {key[:5]}*****")
            else:
                logger.warning(f"{name} Key seems empty or too short!")

        # 3. Check optional keys
        if settings.HF_API_KEY:
            logger.info(f"Hugging Face Key found: {settings.HF_API_KEY[:5]}...")
        else:
            logger.info("Hugging Face Key is empty (Optional).")

        print("\n🎉 All critical configurations are VALID and LOADED!")

    except Exception as e:
        logger.critical(f"CONFIGURATION ERROR: {e}")
        print("\n💡 Tip: Check if your .env file is in the SAME folder as config.py.")

if __name__ == "__main__":
    test_config_loading()