from supabase import create_client, Client
from config import settings
from loguru import logger

# Initialize the Supabase Client
try:
    # We pull the URL and Key directly from our Settings object
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    logger.info("🚀 Supabase client initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase: {e}")
    supabase = None

def test_connection():
    """Run this to verify your .env keys are correct."""
    if not supabase:
        print("❌ Client not initialized. Check your .env file.")
        return

    try:
        # We try to fetch the table we planned earlier
        # Even if it's empty, a successful 'execute()' means the connection works.
        supabase.table("market_signals").select("*").limit(1).execute()
        print("✅ Connection Success: Stock-AI is talking to the Cloud!")
    except Exception as e:
        # If the table doesn't exist yet, it will throw an error, 
        # but the fact that it reached the DB is what matters.
        if "does not exist" in str(e):
            print("✅ Connection Success: The keys work, but we need to build the tables next!")
        else:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_connection()