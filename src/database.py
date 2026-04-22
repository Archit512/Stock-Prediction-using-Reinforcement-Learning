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

    def update_account_status(self, balance, shares, equity_value):
        """Syncs balance and equity after a trade is executed."""
        data = {
            "current_balance": float(balance), 
            "total_shares": float(shares),
            "equity_value": float(equity_value) # 🔥 NEW: Syncs the true portfolio value
        }
        self.supabase.table("user_account").update(data).eq("id", 1).execute()

    def log_transaction(self, symbol, trade_type, quantity, price, fees=0.0, timestamp=None):
        """Stores each executed BUY or SELL in the transactions ledger."""
        data = {
            "symbol": symbol,
            "trade_type": trade_type,
            "quantity": float(quantity),
            "price": float(price),
            "fees": float(fees),
        }
        if timestamp is not None:
            data["timestamp"] = timestamp

        self.supabase.table("transactions").insert(data).execute()

    # --- MACRO & REGIME ---
    def update_macro_status(self, panic_score, reason):
        """Logs the global panic level from MacroSentinel."""
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "panic_score": panic_score,
            "reason": reason,
            "regime": "CRASH" if panic_score > 7.5 else "NORMAL"
        }
        # Insert a new row each time so panic events are preserved instead of overwriting.
        self.supabase.table("macro_status").insert(data).execute()

    # --- WATCHLIST & DISCOVERY ---
    def get_active_watchlist(self):
        """Returns list of tickers the coordinator needs to analyze."""
        res = self.supabase.table("watchlist").select("ticker").execute()
        return [item['ticker'] for item in res.data]

    def add_to_watchlist(self, ticker):
        """Adds a newly discovered stock to the tracking list."""
        now = datetime.now(timezone.utc).isoformat()
        self.supabase.table("watchlist").upsert({
            "ticker": ticker,
            "added_at": now,
            "last_analyzed_at": now,
        }).execute()

    def remove_from_watchlist(self, ticker):
        """Removes a ticker from the watchlist."""
        self.supabase.table("watchlist").delete().eq("ticker", ticker).execute()

    def get_watchlist_snapshot(self):
        """Returns watchlist rows for cleanup and activity checks."""
        res = self.supabase.table("watchlist").select("ticker,is_holding,last_analyzed_at,added_at").execute()
        return res.data or []

    def mark_watchlist_analyzed(self, ticker):
        """Updates analysis timestamp for activity tracking."""
        now = datetime.now(timezone.utc).isoformat()
        self.supabase.table("watchlist").update(
            {"last_analyzed_at": now}
        ).eq("ticker", ticker).execute()

    def get_portfolio_holdings(self):
        """Returns tickers that we currently own (where is_holding = True)."""
        res = self.supabase.table("watchlist").select("ticker").eq("is_holding", True).execute()
        return [item['ticker'] for item in res.data]

    # --- MLOPS & LOGGING ---
    def log_market_data(self, ticker, price, sentiment, log, headline=None, status="pending", panic_score=0.0):
        """Consolidated logging for AI signals, price, and headlines."""
        data = {
            "ticker": ticker,
            "price_at_signal": float(price),
            "sentiment_score": float(sentiment),
            "logs": log,  
            "headline": headline,    
            "status": status,
            "macro_panic_score": float(panic_score), 
            "model_version": "v1-pavilion-ppo"
        }
        self.supabase.table("market_signals").insert(data).execute()

    def log_hourly_snapshot(self, cash_balance, holdings_value, net_worth, profit_loss, panic_score=0.0, note=""):
        """Stores the hourly portfolio snapshot used to review profit/loss over time."""
        data = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cash_balance": float(cash_balance),
            "holdings_value": float(holdings_value),
            "net_worth": float(net_worth),
            "profit_loss": float(profit_loss),
            "panic_score": float(panic_score),
            "note": note,
        }
        self.supabase.table("portfolio_snapshots").insert(data).execute()

    def get_latest_hourly_snapshot(self):
        """Returns the most recent portfolio snapshot row, or None if table is empty."""
        res = (
            self.supabase.table("portfolio_snapshots")
            .select("*")
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]
        return None

    def get_transactions(self, symbol=None, limit=500):
        """Returns recent transaction rows for reporting or debugging."""
        query = self.supabase.table("transactions").select("*").order("timestamp", desc=True).limit(limit)
        if symbol:
            query = query.eq("symbol", symbol)
        res = query.execute()
        return res.data or []

    def get_training_data(self):
        """Pulls all signals and transforms for RL training pipeline."""
        res = self.supabase.table("market_signals").select("*").order("created_at").execute()
        df = pd.DataFrame(res.data)
        
        if df.empty:
            logger.warning("No training data in market_signals table yet.")
            return df
        
        # --- TRANSFORM: Map storage columns to RL environment columns ---
        if 'price_at_signal' in df.columns:
            # Convert to float to ensure mathematical operations work
            df['price'] = df['price_at_signal'].astype(float)
        else:
            df['price'] = 1.0 # Safe fallback
        
        # --- CRITICAL FIX: ACTUAL PRICE CHANGE ---
        # Calculate the real percentage change, grouped by ticker to prevent 
        # crossing data streams (e.g., subtracting AAPL price from TSLA price)
        if 'ticker' in df.columns:
            df['price_change'] = df.groupby('ticker')['price'].pct_change().fillna(0.0)
        else:
            df['price_change'] = df['price'].pct_change().fillna(0.0)
        
        # Ensure macro_panic_score exists (fallback to neutral 3.0 if older rows lack it)
        if 'macro_panic_score' not in df.columns:
            df['macro_panic_score'] = 3.0
        
        # Ensure sentiment_score exists (0.0 if partial data)
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0
            
        # Keep only columns RL environment expects
        required_cols = ['price', 'price_change', 'sentiment_score', 'macro_panic_score']
        
        # Final safety check for missing columns
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # --- THE FIX: Only apply fillna to our specific math columns! ---
        df[required_cols] = df[required_cols].fillna(0.0)
        
        logger.info(f"Loaded {len(df)} training samples with columns: {list(df.columns)}")
        return df[required_cols]
    
    def update_holding_status(self, ticker, is_holding: bool):
        """Updates the watchlist table when a buy/sell is executed."""
        self.supabase.table("watchlist").update({"is_holding": is_holding}).eq("ticker", ticker).execute()

    def sync_broker_account(self, cash, equity):
        """Syncs the Alpaca paper trading balance back to Supabase user_account."""
        data = {
            "current_balance": float(cash),
            "equity_value": float(equity)
        }
        self.supabase.table("user_account").update(data).eq("id", 1).execute()

# Create the singleton instance
db = DatabaseManager()