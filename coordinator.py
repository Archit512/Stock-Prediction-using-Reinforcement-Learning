import time
import sys
import datetime # 🛠️ ADDED: To check real-world time
from loguru import logger
from config import settings
from src.agents.data_fetcher import DataFetcher
from src.agents.sentiment_agent import DualGroupAgent
from src.agents.macro_agent import MacroSentinel
from src.database import DatabaseManager
from src.inference import TradingBrain

# Configure logger 
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", colorize=True)

class TradingCoordinator:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = DualGroupAgent()
        self.macro = MacroSentinel()
        self.db = DatabaseManager()
        self.brain = TradingBrain()
        # 🗑️ REMOVED: self.cycle_count (No longer needed)

    def run_once(self):
        """Executes a single cycle and exits. Perfect for Google Cloud Scheduler."""
        
        # 🛠️ NEW LOGIC: Check the current minute to decide the cycle type
        # If Cloud Scheduler triggers at *:00, *:15, *:30, *:45
        # The *:00 run will have a minute close to 0 (e.g., 0, 1, or 2 due to slight delays)
        current_minute = datetime.datetime.now().minute
        
        # If the minute is less than 16, treat it as the top-of-the-hour heavy cycle
        is_hourly_cycle = current_minute < 16 
        
        cycle_type = "🧠 HEAVY (Full AI & Discovery)" if is_hourly_cycle else "⚡ LIGHT (Macro & Price Sync)"
        
        logger.info(f"--- 🏁 GOOGLE CLOUD EXECUTION START | Minute: {current_minute} | Mode: {cycle_type} ---")

        # --- 🛡️ 15-MINUTE ROUTINE: MACRO SAFETY CHECK ---
        logger.info("🌐 Fetching Global Macro Context...")
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        
        self.db.update_macro_status(panic_data['score'], panic_data['reason'])
        
        if panic_data['regime'] == "CRASH":
            logger.critical(f"🚨 GLOBAL CRASH DETECTED ({panic_data['score']}/10). HALTING BUYS.")
            self._process_holdings(hourly=is_hourly_cycle, emergency=True) 
            logger.info("🏁 EXECUTION COMPLETE. Exiting.")
            return

        # --- 15-MINUTE ROUTINE: PRICE SYNC ---
        if not is_hourly_cycle:
            logger.info("📋 Quick Sync: Updating prices for Holdings & Watchlist...")
            self._process_holdings(hourly=False)
            self._process_watchlist(hourly=False, panic_score=panic_data['score'])
            logger.info("🏁 LIGHT CYCLE COMPLETE. Exiting gracefully.")
            return 

        # --- 🧠 HOURLY ROUTINE: FULL AI PIPELINE ---
        if is_hourly_cycle:
            logger.info("📋 Auditing Holdings (Price & Sentiment)...")
            self._process_holdings(hourly=True)

            logger.info("👀 Evaluating Watchlist (AI & RL Brain)...")
            self._process_watchlist(hourly=True, panic_score=panic_data['score'])

            logger.info("🔭 Scanning Market for New Discoveries...")
            self._discover_new_stocks()
            logger.info("🏁 HEAVY CYCLE COMPLETE. Exiting gracefully.")

    # ... (Keep all your existing _process_holdings, _process_watchlist, and _discover_new_stocks functions exactly as they are) ...

# 🛠️ UPDATED EXECUTION BLOCK
if __name__ == "__main__":
    bot = TradingCoordinator()
    # 🗑️ REMOVED: bot.start() and the while loop
    
    # Just run it exactly once and let the script naturally end
    try:
        bot.run_once()
    except Exception as e:
        logger.error(f"🚨 Cloud Execution Error: {e}")