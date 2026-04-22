import sys
import datetime
from datetime import datetime as dt
from loguru import logger

# Import existing modules
from config import settings
from src.agents.data_fetcher import DataFetcher
from src.agents.sentiment_agent import DualGroupAgent
from src.agents.macro_agent import MacroSentinel
from src.database import DatabaseManager
from src.inference import TradingBrain

# 🛠️ NEW: Import the Alpaca Broker
from src.broker import PaperTrader

# Configure logger 
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", colorize=True)

class TradingCoordinator:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = DualGroupAgent()
        self.macro = MacroSentinel()
        self.db = DatabaseManager()
        self.initial_prices = {}
        
        self.hourly_opening_net_worth = None
        # 🧠 Initialize the AI and Execution Layers
        self.brain = TradingBrain()
        self.broker = PaperTrader()

    def run_once(self):
        """Executes a single cycle and exits. Perfect for AWS Lambda / EventBridge."""
        current_minute = datetime.datetime.now().minute
        
        
        # Heavy cycle runs only in the first 15 mins of the hour to save API credits.
        is_hourly_cycle = current_minute < 15
        
        cycle_type = "🧠 HEAVY (Full AI & Execution)" if is_hourly_cycle else "⚡ LIGHT (Macro & Price Sync)"
        logger.info(f"--- 🏁 CLOUD EXECUTION START | Minute: {current_minute} | Mode: {cycle_type} ---")

        # --- 🛡️ MACRO SAFETY CHECK ---
        logger.info("🌐 Fetching Global Macro Context...")
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        
        self.db.update_macro_status(panic_data['panic_score'], panic_data['reason'])
        
        if panic_data['regime'] == "CRASH":
            logger.critical(f"🚨 GLOBAL CRASH DETECTED ({panic_data['panic_score']}/10). HALTING BUYS.")
            self._process_holdings(hourly=is_hourly_cycle, emergency=True) 
            logger.info("🏁 EXECUTION COMPLETE. Exiting.")
            return

        # --- ⚡ LIGHT CYCLE: PRICE SYNC ONLY ---
        if not is_hourly_cycle:
            logger.info("📋 Quick Sync: Updating prices for Holdings & Watchlist...")
            self._process_holdings(hourly=False)
            self._process_watchlist(hourly=False, panic_score=panic_data['panic_score'])
            logger.info("🏁 LIGHT CYCLE COMPLETE. Exiting gracefully.")
            return 

        # --- 🧠 HEAVY CYCLE: FULL AI PIPELINE ---
        if is_hourly_cycle:
            logger.info("📋 Auditing Holdings (Price, Sentiment & Broker Sync)...")
            self._process_holdings(hourly=True)

            hourly_news_items = self.fetcher.get_random_market_news(limit=10)

            logger.info("👀 Evaluating Watchlist (AI, RL Brain & Execution)...")
            self._process_watchlist(hourly=True, panic_score=panic_data['panic_score'], hourly_news_items=hourly_news_items)

            logger.info("🔭 Scanning Market for New Discoveries...")
            self._discover_new_stocks(news_items=hourly_news_items)

            self._record_hourly_performance_snapshot(panic_score=panic_data['panic_score'])
            logger.info("🏁 HEAVY CYCLE COMPLETE. Exiting gracefully.")

    def _process_holdings(self, hourly=False, emergency=False):
        """Audit current holdings, manage emergency liquidation, and sync balances."""
        try:
            holdings = self.db.get_portfolio_holdings()
            if not holdings:
                logger.info("📌 No current holdings to audit.")
                return
            
            for ticker in holdings:
                price = self.fetcher.get_price(ticker)
                if price:
                    logger.info(f"💵 {ticker}: ${price} (monitoring holding)")
                    
                    # 🚨 EMERGENCY LIQUIDATION TO CASH
                    if emergency:
                        logger.critical(f"🚨 EMERGENCY LIQUIDATION: {ticker}")
                        # Action 2 is SELL, Size 1.0 is 100% of position
                        quantity = 0.0
                        try:
                            quantity = float(self.broker.client.get_open_position(ticker).qty)
                        except Exception:
                            quantity = 0.0
                        success = self.broker.execute_trade(ticker, action_type=2, size=1.0, current_price=float(price))
                        if success:
                            self.db.update_holding_status(ticker, False)
                            if quantity > 0:
                                self.db.log_transaction(ticker, "SELL", quantity, price)
                            # 🔥 FIXED: Using named arguments to prevent headline-column-shift
                            self.db.log_market_data(
                                ticker=ticker, 
                                price=price, 
                                sentiment=0.0, 
                                log="Emergency Macro Liquidation", 
                                status="SELL"
                            )
                        continue

                    # Regular hourly audit
                    if not emergency and hourly:
                        sentiment = self.analyzer.analyze(ticker, f"Brief sentiment for {ticker}")
                        if sentiment:
                            self.db.log_market_data(
                                ticker=ticker, 
                                price=price, 
                                sentiment=sentiment.get('score', 0), 
                                log=sentiment.get('reason', ''), 
                                status='monitoring'
                            )
            
            # 💼 Sync Alpaca balance to Supabase user_account
            acc_data = self.broker.get_account_sync_data()
            if acc_data:
                self.db.sync_broker_account(acc_data['cash'], acc_data['equity'])
                
        except Exception as e:
            logger.error(f"🚨 Error processing holdings: {e}")

    def _process_watchlist(self, hourly=False, panic_score=0):
        """Evaluate watchlist, fetch RL predictions, and execute trades via Broker."""
        try:
            watchlist = self.db.get_active_watchlist()
            if not watchlist:
                logger.info("📋 Watchlist is empty.")
                return

            news_by_ticker = {}
            if hourly:
                hourly_news_items = self.fetcher.get_random_market_news(limit=10)
                for item in hourly_news_items:
                    ticker = item.get("ticker")
                    headline = item.get("headline")
                    if ticker and headline and ticker not in news_by_ticker:
                        news_by_ticker[ticker] = headline
            
            for ticker in watchlist[:5]:  # Limit to 5 to save API credits
                price = self.fetcher.get_price(ticker)
                
                if not price:
                    logger.warning(f"⚠️ {ticker}: Price fetch failed, skipping.")
                    self.db.mark_watchlist_analyzed(ticker)
                    continue

                if not hourly:
                    # 🔥 FIXED: Using named arguments
                    self.db.log_market_data(ticker=ticker, price=price, sentiment=0.0, log="Price sync only", status="price_only")
                    self.db.mark_watchlist_analyzed(ticker)
                    logger.info(f"💵 {ticker}: Price synced (light cycle)")
                    continue
                
                # Try to fetch news and sentiment
                headline = None
                if hourly:
                    headline = news_by_ticker.get(ticker)
                else:
                    news = self.fetcher.get_random_market_news(limit=1)
                    if news:
                        headline = news[0]['headline']

                if headline:
                    sentiment = self.analyzer.analyze(ticker, headline)
                    if sentiment:
                        if hasattr(self, "brain") and self.brain is not None:
                            
                            # Feature Engineering: Price Change
                            initial_price = self.initial_prices.get(ticker)
                            if initial_price is None:
                                self.initial_prices[ticker] = float(price)
                                initial_price = float(price)

                            price_change = 0.0
                            if initial_price:
                                price_change = (float(price) - float(initial_price)) / float(initial_price)

                            # 🧠 1. Get Action & Size from RL Brain
                            action, size = self.brain.get_action(price_change, sentiment.get('score', 0), panic_score)
                            status = ["HOLD", "BUY", "SELL"][action] if action in [0, 1, 2] else "HOLD"

                            trade_quantity = 0.0
                            if status == "BUY":
                                account = self.broker.get_account_sync_data()
                                available_cash = float(account["cash"]) if account else 0.0
                                trade_quantity = int((available_cash * float(size)) // float(price)) if price else 0.0
                            elif status == "SELL":
                                try:
                                    trade_quantity = float(self.broker.client.get_open_position(ticker).qty)
                                except Exception:
                                    trade_quantity = 0.0
                            
                            # 🛑 NEW: Weekend Check (0 = Monday, 5 = Saturday, 6 = Sunday)
                            is_weekend = datetime.datetime.now().weekday() >= 5

                            # 💼 2. Execute Trade via Alpaca
                            if status in ["BUY", "SELL"]:
                                if is_weekend:
                                    logger.info(f"⏸️ Weekend active! Skipping broker execution for {ticker}. Intended: {status}.")
                                    status = f"weekend_{status.lower()}" # Modifies status so it logs intention without acting
                                else:
                                    logger.info(f"⚡ Brain requested {status} for {ticker}. Sending to Broker...")
                                    trade_success = self.broker.execute_trade(ticker, action, size, float(price))
                                    
                                    # 🗄️ 3. Update Database State on Success
                                    if trade_success:
                                        is_holding = True if status == "BUY" else False
                                        self.db.update_holding_status(ticker, is_holding)
                                        if trade_quantity > 0:
                                            self.db.log_transaction(ticker, status, trade_quantity, price)
                                    else:
                                        status = "HOLD" # Revert status if trade failed (market closed/insufficient funds)

                        else:
                            status = "pending"
                            logger.warning("⚠️ TradingBrain missing; logging pending signal only.")
                        
                        # 🔥 FIXED: Comprehensive logging with named arguments to ensure correct columns
                        self.db.log_market_data(
                            ticker=ticker, 
                            price=price, 
                            sentiment=sentiment.get('score', 0), 
                            log=sentiment.get('reason', ''), 
                            headline=headline,
                            status=status,
                            panic_score=panic_score
                        )
                        logger.info(f"📊 {ticker}: {status} signal logged")
                    else:
                        logger.warning(f"⚠️ {ticker}: Sentiment analysis failed, saving price only.")
                        self.db.log_market_data(ticker=ticker, price=price, sentiment=0.0, log="Sentiment analysis unavailable", status="pending")
                else:
                    logger.warning(f"⚠️ {ticker}: News unavailable, saving price only.")
                    self.db.log_market_data(ticker=ticker, price=price, sentiment=0.0, log="News fetch failed", status="price_only")

                self.db.mark_watchlist_analyzed(ticker)
            
            # 💼 Sync Alpaca balance to Supabase after watchlist executions
            acc_data = self.broker.get_account_sync_data()
            if acc_data:
                self.db.sync_broker_account(acc_data['cash'], acc_data['equity'])
                    
        except Exception as e:
            logger.error(f"🚨 Error processing watchlist: {e}")

    def _record_hourly_performance_snapshot(self, panic_score=0):
        """Stores a once-per-heavy-cycle portfolio snapshot for later P/L review."""
        try:
            acc_data = self.broker.get_account_sync_data()
            if acc_data:
                cash_balance = float(acc_data.get("cash", 0.0))
                net_worth = float(acc_data.get("equity", cash_balance))
            else:
                account = self.db.get_account_status()
                cash_balance = float(account.get("current_balance", 0.0))
                net_worth = float(account.get("equity_value", cash_balance))

            holdings_value = max(0.0, net_worth - cash_balance)

            previous_snapshot = self.db.get_latest_hourly_snapshot()
            if previous_snapshot:
                baseline_worth = float(previous_snapshot.get("net_worth", net_worth))
            else:
                baseline_worth = self.hourly_opening_net_worth if self.hourly_opening_net_worth is not None else net_worth
                if self.hourly_opening_net_worth is None:
                    self.hourly_opening_net_worth = net_worth

            profit_loss = net_worth - float(baseline_worth)

            self.db.log_hourly_snapshot(
                cash_balance=cash_balance,
                holdings_value=holdings_value,
                net_worth=net_worth,
                profit_loss=profit_loss,
                panic_score=panic_score,
                note="heavy_cycle_snapshot",
            )
            logger.info(f"📈 Hourly snapshot saved | Net worth: ${net_worth:.2f} | P/L: ${profit_loss:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ Unable to save hourly performance snapshot: {e}")

    def _cleanup_watchlist_if_oversized(self, max_size=10, min_days_to_keep=7):
        """
        Remove older non-holding tickers when watchlist size exceeds max_size.
        Enforces a strict minimum tracking window (default 7 days) to ensure
        high-quality continuous data for RL training.
        """
        snapshot = self.db.get_watchlist_snapshot()
        current_size = len(snapshot)

        if current_size <= max_size:
            return

        removable = [row for row in snapshot if not row.get("is_holding", False)]
        if not removable:
            return

        def _to_dt(value):
            if not value:
                return dt.min
            try:
                return dt.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                return dt.min

        removable.sort(key=lambda row: (_to_dt(row.get("last_analyzed_at")), _to_dt(row.get("added_at"))))

        remove_count = current_size - max_size
        for row in removable[:remove_count]:
            ticker = row.get("ticker")
            if ticker:
                self.db.remove_from_watchlist(ticker)
                logger.info(f"🧹 Removed inactive ticker from watchlist: {ticker}")

    def _discover_new_stocks(self, news_items=None):
        """Scan market for new trading opportunities."""
        try:
            self._cleanup_watchlist_if_oversized(max_size=10)
            if news_items is None:
                news_items = self.fetcher.get_random_market_news(limit=5)
            current_watchlist = set(self.db.get_active_watchlist())
            
            for item in news_items:
                ticker = item['ticker']
                if ticker not in current_watchlist and len(current_watchlist) < 10:
                    self.db.add_to_watchlist(ticker)
                    current_watchlist.add(ticker)
                    logger.success(f"🆕 Added {ticker} to watchlist via {item['source']}")
        except Exception as e:
            logger.error(f"🚨 Error discovering new stocks: {e}")

if __name__ == "__main__":
    bot = TradingCoordinator()
    try:
        bot.run_once()
    except Exception as e:
        logger.error(f"🚨 Cloud Execution Error: {e}")