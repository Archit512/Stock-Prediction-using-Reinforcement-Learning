import time
import sys
from loguru import logger
from config import settings
from agents.data_fetcher import DataFetcher
from agents.sentiment_agent import DualGroupAgent
from agents.macro_agent import MacroSentinel
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
        self.cycle_count = 0

    def run_cycle(self):
        """The 4-Step MLOps Pipeline: Macro -> Holdings -> Watchlist -> Discovery"""
        self.cycle_count += 1
        logger.info(f"--- 🏁 STARTING CYCLE #{self.cycle_count} ---")

        # --- STEP 1: GLOBAL MACRO SENTINEL ---
        # Fetch real-time global news and determine if the market is safe
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        
        # Save macro state to Supabase
        self.db.update_macro_status(panic_data['score'], panic_data['reason'])
        
        if panic_data['regime'] == "CRASH":
            logger.critical(f"🚨 GLOBAL CRASH DETECTED ({panic_data['score']}/10). HALTING ALL BUYS.")
            self._process_holdings(emergency=True) # Only check if we need to escape current positions
            return

        # --- STEP 2: HOLDINGS MAINTENANCE ---
        # Monitor stocks you already own (Babysitting)
        logger.info("📋 Auditing Current Holdings...")
        self._process_holdings()

        # --- STEP 3: WATCHLIST EVALUATION ---
        # Re-check stocks you previously liked to see if they hit a 'BUY' trigger
        logger.info("👀 Re-evaluating Watchlist Tickers...")
        self._process_watchlist()

        # --- STEP 4: DISCOVERY (NEW OPPORTUNITIES) ---
        # Find brand new stocks trending in the market
        logger.info("🔭 Scanning Market for New Opportunities...")
        self._discover_new_stocks()

    def _process_holdings(self, emergency=False):
        holdings = self.db.get_portfolio_holdings()
        for ticker in holdings:
            news = self.fetcher.get_ticker_news(ticker)
            price = self.fetcher.get_price(ticker)
            if not news or not price: continue

            result = self.analyzer.analyze(ticker, news)
            if result:
                self.db.log_market_data(ticker, price, result['score'], result['reason'], "HOLDING")
                
                # Logic for RL to eventually handle: -0.8 is an auto-sell, -0.3 is 'Chill'
                if result['score'] <= -0.8:
                    logger.critical(f"🔥 EMERGENCY SELL SIGNAL: {ticker} at {result['score']}")
                elif result['score'] < -0.4:
                    logger.warning(f"📉 WEAKNESS DETECTED: {ticker} sentiment is {result['score']}")

    def _process_watchlist(self):
        watchlist = self.db.get_active_watchlist()
        # 1. Get Global Panic Score once per cycle to save API calls
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        panic_score = panic_data['score']

        for ticker in watchlist:
            news = self.fetcher.get_ticker_news(ticker)
            price = self.fetcher.get_price(ticker)
            if not news or not price: continue

            # 2. Analyze Sentiment (Dual-Group Consensus)
            result = self.analyzer.analyze(ticker, news)
            if not result: continue

            # 3. ASK THE RL BRAIN (The Inference Step)
            # We calculate price_change (e.g., 0.02 for 2%) for the agent's observation
            # For discovery/watchlist, we can use 0 or recent 24h change
            price_change = 0.0 
            
            # Action: 0=Hold, 1=Buy, 2=Sell | Size: 0.0 to 1.0 (Kelly Limited)
            action_type, size = self.brain.get_action(
                price_change=price_change, 
                sentiment=result['score'], 
                panic=panic_score
            )

            # 4. EXECUTE BASED ON AGENT DECISION
            if action_type == 1 and size > 0:
                logger.success(f"💰 RL BUY SIGNAL: {ticker} | Size: {size*100:.1f}% | Confidence: {result['score']}")
                
                # Log the decision to Supabase for the RL 'Value Updation' tracking
                self.db.log_market_data(
                    ticker=ticker, 
                    price=price, 
                    sentiment=result['score'], 
                    reasoning=f"RL Action {action_type} with size {size}. Reason: {result['reason']}", 
                    status="RL_EXECUTED"
                )
                
                # TODO: Link your actual Broker API or Paper Trading Execute call here
            else:
                logger.info(f"⏳ RL Agent chose to HOLD {ticker} (Action: {action_type})")

    def _discover_new_stocks(self):
        trending = self.fetcher.get_random_market_news(limit=5)
        watchlist_set = set(self.db.get_active_watchlist())
        holdings_set = set(self.db.get_portfolio_holdings())

        for item in trending:
            ticker = item['ticker']
            if ticker in watchlist_set or ticker in holdings_set: continue

            price = self.fetcher.get_price(ticker)
            result = self.analyzer.analyze(ticker, item['headline'])
            
            if result and result['score'] > 0.5:
                logger.info(f"✨ New Discovery: {ticker} added to Watchlist.")
                self.db.add_to_watchlist(ticker)
                self.db.log_market_data(ticker, price, result['score'], result['reason'], "DISCOVERY")

    def start(self, interval_minutes=15):
        """Infinite loop for 24/7 market monitoring."""
        logger.info(f"💎 Bot Active. Monitoring every {interval_minutes} minutes.")
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"🚨 Coordinator Error: {e}")
            
            logger.info(f"💤 Cycle #{self.cycle_count} complete. Sleeping...")
            time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    bot = TradingCoordinator()
    bot.start(interval_minutes=15)