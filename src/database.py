from supabase import create_client, Client
from config import settings
from loguru import logger
import pandas as pd
from datetime import datetime, timezone

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
    def update_macro_status(self, panic_score, reason):
        """Logs the global panic level from MacroSentinel."""
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "panic_score": panic_score,
            "reason": reason,
            "regime": "CRASH" if panic_score > 7.5 else "NORMAL"
        }
        # Insert a new row each time so panic events are preserved
        self.supabase.table("macro_status").insert(data).execute()

    # --- WATCHLIST & DISCOVERY ---
    def get_active_watchlist(self):
        """Returns list of tickers the coordinator needs to analyze."""
        res = self.supabase.table("watchlist").select("ticker").execute()
        return [item['ticker'] for item in res.data]

    def add_to_watchlist(self, ticker):
        """Adds a newly discovered stock to the tracking list."""
        self.supabase.table("watchlist").upsert({"ticker": ticker}).execute()

    def remove_from_watchlist(self, ticker):
        """Removes a ticker from the watchlist."""
        self.supabase.table("watchlist").delete().eq("ticker", ticker).execute()

    def get_watchlist_snapshot(self):
        """Returns watchlist rows for cleanup and activity checks."""
        res = self.supabase.table("watchlist").select("ticker,is_holding,last_analyzed_at,added_at").execute()
        return res.data or []

    def mark_watchlist_analyzed(self, ticker):
        """Updates analysis timestamp for activity tracking."""
        self.supabase.table("watchlist").update(
            {"last_analyzed_at": datetime.now(timezone.utc).isoformat()}
        ).eq("ticker", ticker).execute()

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
        """Pulls all signals and transforms for RL training pipeline."""
        res = self.supabase.table("market_signals").select("*").order("created_at").execute()
        df = pd.DataFrame(res.data)
        
        if df.empty:
            logger.warning("⚠️ No training data in market_signals table yet.")
            return df
        
        # --- TRANSFORM: Map storage columns to RL environment columns ---
        # Rename price_at_signal to price for clarity
        if 'price_at_signal' in df.columns:
            df['price'] = df['price_at_signal']
        
        # Calculate price_change (simplified: use sentiment as proxy for now)
        # In production, compute from consecutive price rows
        if 'sentiment_score' in df.columns:
            df['price_change'] = df['sentiment_score'] / 10.0  # Normalized change
        else:
            df['price_change'] = 0.0
        
        # Add macro_panic_score column (default to 3 = neutral if missing)
        if 'macro_panic_score' not in df.columns:
            df['macro_panic_score'] = 3.0
        
        # Ensure sentiment_score exists (0.0 if partial data)
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0
        
        # Keep only columns RL environment expects
        required_cols = ['price', 'price_change', 'sentiment_score', 'macro_panic_score']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        logger.info(f"✅ Loaded {len(df)} training samples with columns: {list(df.columns)}")
        return df[required_cols]

# Create the singleton instance
db = DatabaseManager()