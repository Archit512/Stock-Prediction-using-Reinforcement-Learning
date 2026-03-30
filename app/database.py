from supabase import create_client, Client
from config import settings
from loguru import logger
import pandas as pd

class DatabaseManager:
    def __init__(self):
        # Initializes the connection using your .env secrets
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # --- PORTFOLIO & ACCOUNT ---
    def get_account_status(self):
        """Fetches global balance for RL Agent observation."""
        res = self.supabase.table("user_account").select("*").eq("id", 1).single().execute()
        return res.data

    def update_account_status(self, balance, shares):
        """Syncs balance after a trade is executed."""
        data = {"current_balance": float(balance), "total_shares": float(shares)}
        self.supabase.table("user_account").update(data).eq("id", 1).execute()

    # --- MACRO & REGIME ---
    def update_macro_status(self, score, reason):
        """Logs the global panic level from MacroSentinel."""
        # Note: You need the macro_status table in your SQL for this to work
        data = {"panic_score": score, "reason": reason, "regime": "CRASH" if score > 7.5 else "NORMAL"}
        self.supabase.table("macro_status").upsert({"id": 1, **data}).execute()

    # --- WATCHLIST & DISCOVERY ---
    def get_active_watchlist(self):
        """Returns list of tickers the coordinator needs to analyze."""
        res = self.supabase.table("watchlist").select("ticker").execute()
        return [item['ticker'] for item in res.data]

    def add_to_watchlist(self, ticker):
        """Adds a newly discovered stock to the tracking list."""
        self.supabase.table("watchlist").upsert({"ticker": ticker}).execute()

    def get_portfolio_holdings(self):
        """Returns tickers that we currently own (where is_holding = True)."""
        res = self.supabase.table("watchlist").select("ticker").eq("is_holding", True).execute()
        return [item['ticker'] for item in res.data]

    # --- MLOPS & LOGGING ---
    def log_market_data(self, ticker, price, sentiment, reasoning, status="pending"):
        """The core logging function for the RL training pipeline."""
        data = {
            "ticker": ticker,
            "price_at_signal": float(price),
            "sentiment_score": float(sentiment),
            "headline": reasoning, 
            "status": status,
            "model_version": "v1-pavilion-ppo"
        }
        self.supabase.table("market_signals").insert(data).execute()

    def get_training_data(self):
        """Pulls all signals so Person A can train the model locally."""
        res = self.supabase.table("market_signals").select("*").order("created_at").execute()
        return pd.DataFrame(res.data)

# Create the singleton instance
db = DatabaseManager()