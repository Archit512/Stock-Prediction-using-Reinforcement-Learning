from supabase import create_client, Client
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class DatabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    def get_account_status(self):
        """Fetches current global balance and shares from Supabase."""
        res = self.supabase.table("user_account").select("*").eq("id", 1).single().execute()
        return res.data

    def update_account_status(self, balance, shares):
        """Persists the new balance after a trade."""
        data = {
            "current_balance": float(balance),
            "total_shares": float(shares),
            "last_updated": "now()"
        }
        self.supabase.table("user_account").update(data).eq("id", 1).execute()

    def get_training_data(self):
        """Downloads all logs for the RL agent to study on your HP Omen."""
        res = self.supabase.table("market_logs").select("*").order("created_at").execute()
        import pandas as pd
        return pd.DataFrame(res.data)

db = DatabaseManager()