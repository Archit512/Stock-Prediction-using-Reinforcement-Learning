import time
import sys
import datetime 
from datetime import datetime as dt
from loguru import logger
from config import settings
from src.agents.data_fetcher import DataFetcher
from src.agents.sentiment_agent import DualGroupAgent
from src.agents.macro_agent import MacroSentinel
from src.database import DatabaseManager
from src.inference import TradingBrain

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", colorize=True)

class TradingCoordinator:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = DualGroupAgent()
        self.macro = MacroSentinel()
        self.db = DatabaseManager()
        self.initial_prices = {}
        self.brain = TradingBrain()

    def run_once(self):
        current_minute = datetime.datetime.now().minute
        is_hourly_cycle = current_minute < 15
        
        cycle_type = "HEAVY (Full AI & Discovery)" if is_hourly_cycle else "LIGHT (Macro & Price Sync)"
        logger.info(f"--- AWS CLOUD EXECUTION START | Minute: {current_minute} | Mode: {cycle_type} ---")

        logger.info("Fetching Global Macro Context...")
        macro_news = self.fetcher.get_global_macro_news()
        panic_data = self.macro.get_panic_status(macro_news)
        
        self.db.update_macro_status(panic_data['panic_score'], panic_data['reason'])
        
        if panic_data['regime'] == "CRASH":
            logger.critical(f"GLOBAL CRASH DETECTED ({panic_data['panic_score']}/10). HALTING BUYS.")
            self._process_holdings(hourly=is_hourly_cycle, emergency=True, panic_score=panic_data['panic_score']) 
            logger.info("EXECUTION COMPLETE. Exiting.")
            return

        if not is_hourly_cycle:
            logger.info("Quick Sync: Updating prices for Holdings & Watchlist...")
            self._process_holdings(hourly=False, panic_score=panic_data['panic_score'])
            self._process_watchlist(hourly=False, panic_score=panic_data['panic_score'])
            logger.info("LIGHT CYCLE COMPLETE. Exiting gracefully.")
            return 

        if is_hourly_cycle:
            logger.info("Auditing Holdings (Price & Sentiment)...")
            self._process_holdings(hourly=True, panic_score=panic_data['panic_score'])

            hourly_news_items = self.fetcher.get_random_market_news(limit=10)

            logger.info("Evaluating Watchlist (AI & RL Brain)...")
            self._process_watchlist(hourly=True, panic_score=panic_data['panic_score'], hourly_news_items=hourly_news_items)

            logger.info("Scanning Market for New Discoveries...")
            self._discover_new_stocks(news_items=hourly_news_items)
            logger.info("HEAVY CYCLE COMPLETE. Exiting gracefully.")

    def _process_holdings(self, hourly=False, emergency=False, panic_score=0):
        try:
            holdings = self.db.get_portfolio_holdings()
            if not holdings:
                logger.info("No current holdings to audit.")
                return
            
            for ticker in holdings:
                price = self.fetcher.get_price(ticker)
                if price:
                    logger.info(f"{ticker}: ${price} (monitoring)")
                    if not emergency:
                        sentiment = self.analyzer.analyze(ticker, f"Brief sentiment for {ticker}")
                        if sentiment:
                            self.db.log_market_data(ticker, price, sentiment.get('score', 0), 
                                                    sentiment.get('reason', ''), status='monitoring', panic_score=panic_score)
        except Exception as e:
            logger.error(f"Error processing holdings: {e}")

    def _process_watchlist(self, hourly=False, panic_score=0, hourly_news_items=None):
        try:
            watchlist = self.db.get_active_watchlist()
            if not watchlist:
                logger.info("Watchlist is empty.")
                return

            news_by_ticker = {}
            if hourly and hourly_news_items:
                for item in hourly_news_items:
                    ticker = item.get("ticker")
                    headline = item.get("headline")
                    if ticker and headline and ticker not in news_by_ticker:
                        news_by_ticker[ticker] = headline
            
            for ticker in watchlist[:10]: 
                price = self.fetcher.get_price(ticker)
                
                if not price:
                    logger.warning(f"{ticker}: Price fetch failed, skipping.")
                    self.db.mark_watchlist_analyzed(ticker)
                    continue

                if not hourly:
                    self.db.log_market_data(
                        ticker, price, 0.0, "Price sync only (non-hourly cycle)", 
                        status="price_only", panic_score=panic_score
                    )
                    self.db.mark_watchlist_analyzed(ticker)
                    logger.info(f"{ticker}: Price synced (light cycle)")
                    continue
                
                headline = news_by_ticker.get(ticker)

                # 🛠️ FIX: Targeted fetch fallback if it wasn't in the global batch
                if not headline and hourly:
                    logger.info(f"{ticker} not in batch news. Running targeted fetch...")
                    headline = self.fetcher.get_ticker_news(ticker)

                if headline:
                    sentiment = self.analyzer.analyze(ticker, headline)
                    if sentiment:
                        if hasattr(self, "brain") and self.brain is not None:
                            initial_price = self.initial_prices.get(ticker)
                            if initial_price is None:
                                self.initial_prices[ticker] = float(price)
                                initial_price = float(price)

                            price_change = 0.0
                            if initial_price:
                                price_change = (float(price) - float(initial_price)) / float(initial_price)

                            action, size = self.brain.get_action(price_change, sentiment.get('score', 0), panic_score)
                            status = ["HOLD", "BUY", "SELL"][action] if action in [0, 1, 2] else "HOLD"
                        else:
                            status = "pending"
                            logger.warning("TradingBrain not initialized; logging pending signal only.")
                            
                        self.db.log_market_data(ticker, price, sentiment.get('score', 0), 
                                                sentiment.get('reason', ''), headline=headline, status=status, panic_score=panic_score)
                        logger.info(f"{ticker}: {status} signal logged")
                    else:
                        logger.warning(f"{ticker}: Sentiment analysis failed, saving price only.")
                        self.db.log_market_data(ticker, price, 0.0, 
                                                "News fetched but sentiment analysis unavailable", headline=headline, status="pending", panic_score=panic_score)
                else:
                    logger.warning(f"{ticker}: News unavailable, saving price only.")
                    self.db.log_market_data(ticker, price, 0.0, 
                                            "Price check (news fetch failed)", headline=headline, status="price_only", panic_score=panic_score)

                self.db.mark_watchlist_analyzed(ticker)
                    
        except Exception as e:
            logger.error(f"Error processing watchlist: {e}")

    def _cleanup_watchlist_if_oversized(self, max_size=10, min_days_to_keep=7):
        snapshot = self.db.get_watchlist_snapshot()
        current_size = len(snapshot)

        if current_size <= max_size:
            return

        now = dt.now(datetime.timezone.utc)
        removable_candidates = []

        for row in snapshot:
            if row.get("is_holding", False):
                continue
                
            added_at_str = str(row.get("added_at")).replace("Z", "+00:00")
            try:
                added_at = dt.fromisoformat(added_at_str)
                if added_at.tzinfo is None:
                    added_at = added_at.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                added_at = dt.min.replace(tzinfo=datetime.timezone.utc)

            days_tracked = (now - added_at).days
            if days_tracked >= min_days_to_keep:
                removable_candidates.append(row)

        if not removable_candidates:
            logger.info(f"Watchlist is at {current_size}, but all non-holdings are protected by the {min_days_to_keep}-day immunity. Skipping cleanup.")
            return

        def _to_dt(value):
            if not value: return dt.min.replace(tzinfo=datetime.timezone.utc)
            try: 
                d = dt.fromisoformat(str(value).replace("Z", "+00:00"))
                if d.tzinfo is None: d = d.replace(tzinfo=datetime.timezone.utc)
                return d
            except Exception: return dt.min.replace(tzinfo=datetime.timezone.utc)

        removable_candidates.sort(key=lambda row: _to_dt(row.get("last_analyzed_at")))

        remove_count = current_size - max_size
        for row in removable_candidates[:remove_count]:
            ticker = row.get("ticker")
            if ticker:
                self.db.remove_from_watchlist(ticker)
                logger.info(f"Evicted {ticker} from watchlist (Tracked for {min_days_to_keep}+ days)")

    def _evict_one_inactive_non_holding(self, min_days_to_keep=7):
        snapshot = self.db.get_watchlist_snapshot()
        now = dt.now(datetime.timezone.utc)
        candidates = []

        for row in snapshot:
            if row.get("is_holding", False):
                continue

            added_at_str = str(row.get("added_at")).replace("Z", "+00:00")
            try:
                added_at = dt.fromisoformat(added_at_str)
                if added_at.tzinfo is None:
                    added_at = added_at.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                added_at = dt.min.replace(tzinfo=datetime.timezone.utc)

            days_tracked = (now - added_at).days
            if days_tracked >= min_days_to_keep:
                candidates.append(row)

        if not candidates:
            return None

        def _to_dt(value):
            if not value:
                return dt.min.replace(tzinfo=datetime.timezone.utc)
            try:
                parsed = dt.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                return parsed
            except Exception:
                return dt.min.replace(tzinfo=datetime.timezone.utc)

        victim = min(candidates, key=lambda row: _to_dt(row.get("last_analyzed_at")))
        victim_ticker = victim.get("ticker")
        if victim_ticker:
            self.db.remove_from_watchlist(victim_ticker)
            logger.info(
                f"Evicted {victim_ticker} (inactive, non-holding, tracked {min_days_to_keep}+ days)"
            )
        return victim_ticker

    def _discover_new_stocks(self, news_items=None):
        try:
            if news_items is None:
                news_items = self.fetcher.get_random_market_news(limit=5)

            current_watchlist = set(self.db.get_active_watchlist())
            max_watchlist_size = 10

            for item in news_items:
                ticker = item.get("ticker")
                source = item.get("source", "unknown source")
                if not ticker:
                    continue

                if ticker in current_watchlist:
                    continue

                if len(current_watchlist) < max_watchlist_size:
                    self.db.add_to_watchlist(ticker)
                    current_watchlist.add(ticker)
                    logger.success(f"Added {ticker} to watchlist via {source}")
                    continue

                evicted_ticker = self._evict_one_inactive_non_holding(min_days_to_keep=7)
                if evicted_ticker:
                    current_watchlist.discard(evicted_ticker)
                    self.db.add_to_watchlist(ticker)
                    current_watchlist.add(ticker)
                    logger.success(f"Added {ticker} to watchlist via {source} after evicting {evicted_ticker}")
                else:
                    logger.info(
                        f"Skipped {ticker}: watchlist full and no eligible inactive non-holding (7+ days) to evict"
                    )
        except Exception as e:
            logger.error(f"Error discovering new stocks: {e}")

if __name__ == "__main__":
    bot = TradingCoordinator()
    try:
        bot.run_once()
    except Exception as e:
        logger.error(f"Cloud Execution Error: {e}")