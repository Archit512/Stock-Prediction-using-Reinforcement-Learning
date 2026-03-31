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
        
        # If the minute is less than 15, treat it as the top-of-the-hour heavy cycle
        is_hourly_cycle = current_minute < 15 
        
        cycle_type = "🧠 HEAVY (Full AI & Discovery)" if is_hourly_cycle else "⚡ LIGHT (Macro & Price Sync)"
        
        logger.info(f"--- 🏁 GOOGLE CLOUD EXECUTION START | Minute: {current_minute} | Mode: {cycle_type} ---")

        # --- 🛡️ 15-MINUTE ROUTINE: MACRO SAFETY CHECK ---
        logger.info("🌐 Fetching Global Macro Context...")
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        
        self.db.update_macro_status(panic_data['panic_score'], panic_data['reason'])
        
        if panic_data['regime'] == "CRASH":
            logger.critical(f"🚨 GLOBAL CRASH DETECTED ({panic_data['panic_score']}/10). HALTING BUYS.")
            self._process_holdings(hourly=is_hourly_cycle, emergency=True) 
            logger.info("🏁 EXECUTION COMPLETE. Exiting.")
            return

        # --- 15-MINUTE ROUTINE: PRICE SYNC ---
        if not is_hourly_cycle:
            logger.info("📋 Quick Sync: Updating prices for Holdings & Watchlist...")
            self._process_holdings(hourly=False)
            self._process_watchlist(hourly=False, panic_score=panic_data['panic_score'])
            logger.info("🏁 LIGHT CYCLE COMPLETE. Exiting gracefully.")
            return 

        # --- 🧠 HOURLY ROUTINE: FULL AI PIPELINE ---
        if is_hourly_cycle:
            logger.info("📋 Auditing Holdings (Price & Sentiment)...")
            self._process_holdings(hourly=True)

            logger.info("👀 Evaluating Watchlist (AI & RL Brain)...")
            self._process_watchlist(hourly=True, panic_score=panic_data['panic_score'])

            logger.info("🔭 Scanning Market for New Discoveries...")
            self._discover_new_stocks()
            logger.info("🏁 HEAVY CYCLE COMPLETE. Exiting gracefully.")

    def _process_holdings(self, hourly=False, emergency=False):
        """Audit and rebalance current holdings."""
        try:
            holdings = self.db.get_portfolio_holdings()
            if not holdings:
                logger.info("📌 No current holdings to audit.")
                return
            
            for ticker in holdings:
                price = self.fetcher.get_price(ticker)
                if price:
                    logger.info(f"💵 {ticker}: ${price} (monitoring)")
                    if not emergency:
                        sentiment = self.analyzer.analyze(ticker, f"Brief sentiment for {ticker}")
                        if sentiment:
                            self.db.log_market_data(ticker, price, sentiment.get('score', 0), 
                                                    sentiment.get('reason', ''), 'monitoring')
        except Exception as e:
            logger.error(f"🚨 Error processing holdings: {e}")

    def _process_watchlist(self, hourly=False, panic_score=0):
        """Evaluate watchlist stocks for trading signals."""
        try:
            watchlist = self.db.get_active_watchlist()
            if not watchlist:
                logger.info("📋 Watchlist is empty.")
                return
            
            for ticker in watchlist[:5]:  # Limit to 5 to save API credits
                price = self.fetcher.get_price(ticker)
                
                # --- SAFETY LAYER: Save price even if news fails ---
                if not price:
                    logger.warning(f"⚠️ {ticker}: Price fetch failed, skipping.")
                    continue
                
                # Try to fetch news and sentiment
                news = self.fetcher.get_random_market_news(limit=1)
                
                if news:
                    sentiment = self.analyzer.analyze(ticker, news[0]['headline'])
                    if sentiment:
                        action, size = self.brain.get_action(0.01, sentiment.get('score', 0), panic_score)
                        status = ["HOLD", "BUY", "SELL"][action] if action in [0, 1, 2] else "HOLD"
                        self.db.log_market_data(ticker, price, sentiment.get('score', 0), 
                                                sentiment.get('reason', ''), status)
                        logger.info(f"📊 {ticker}: {status} signal logged")
                    else:
                        # News exists but sentiment analysis failed → save price anyway
                        logger.warning(f"⚠️ {ticker}: Sentiment analysis failed, saving price only.")
                        self.db.log_market_data(ticker, price, 0.0, 
                                                "News fetched but sentiment analysis unavailable", "pending")
                else:
                    # News fetch failed → save price with neutral sentiment
                    logger.warning(f"⚠️ {ticker}: News unavailable, saving price only.")
                    self.db.log_market_data(ticker, price, 0.0, 
                                            "Price check (news fetch failed)", "price_only")
                    
        except Exception as e:
            logger.error(f"🚨 Error processing watchlist: {e}")

    def _discover_new_stocks(self):
        """Scan market for new trading opportunities."""
        try:
            news_items = self.fetcher.get_random_market_news(limit=5)
            current_watchlist = set(self.db.get_active_watchlist())
            
            for item in news_items:
                ticker = item['ticker']
                if ticker not in current_watchlist and len(current_watchlist) < 10:
                    self.db.add_to_watchlist(ticker)
                    logger.success(f"🆕 Added {ticker} to watchlist via {item['source']}")
        except Exception as e:
            logger.error(f"🚨 Error discovering new stocks: {e}")

# 🛠️ UPDATED EXECUTION BLOCK
if __name__ == "__main__":
    bot = TradingCoordinator()
    # 🗑️ REMOVED: bot.start() and the while loop
    
    # Just run it exactly once and let the script naturally end
    try:
        bot.run_once()
    except Exception as e:
        logger.error(f"🚨 Cloud Execution Error: {e}")