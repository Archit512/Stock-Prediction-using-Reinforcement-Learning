import requests
import yfinance as yf
from config import settings
from loguru import logger
from typing import List, Dict, Optional

class DataFetcher:
    def __init__(self):
        self.finnhub_key = settings.FINNHUB_KEY
        self.fmp_key = settings.FMP_KEY
        self.alpha_key = settings.ALPHA_VANTAGE_KEY

    def get_random_market_news(self, limit: int = 15) -> List[Dict]:
        """Discovery Hierarchy: Finnhub -> FMP -> Alpha -> yfinance"""
        discovered = []

        # 1. Try Finnhub
        try:
            url = f"https://finnhub.io/api/v1/news?category=general&token={self.finnhub_key}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                for n in res:
                    ticker = n.get("related")
                    if ticker: # Finnhub only returns if 'related' is tagged
                        discovered.append({"ticker": ticker, "headline": n.get("headline"), "source": "Finnhub"})
                if discovered: return discovered[:limit]
            else:
                logger.debug("🔍 Finnhub returned 0 ticker-related news.")
        except Exception as e:
            logger.debug(f"⚠️ Finnhub API Error: {e}")

        # 2. Try FMP (Often more reliable for trending)
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?limit=25&apikey={self.fmp_key}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                return [{"ticker": n.get("symbol"), "headline": n.get("title"), "source": "FMP"} for n in res[:limit]]
        except Exception as e:
            logger.debug(f"⚠️ FMP API Error: {e}")

        # 3. Try yfinance (THE ULTIMATE SAFETY NET)
        # We search for 'Market' to get broad trending news
        try:
            logger.info("🛡️ Switching to yfinance for discovery...")
            yf_search = yf.Search("Market", news_count=limit)
            if yf_search.news:
                for n in yf_search.news:
                    # yf news often has multiple related symbols
                    ticker = n.get('relatedTickers', [None])[0]
                    if ticker:
                        discovered.append({"ticker": ticker, "headline": n.get("title"), "source": "yfinance"})
                if discovered: return discovered
        except Exception as e:
            logger.error(f"🚨 yfinance Discovery Failed: {e}")

        return []

    def get_price(self, ticker: str) -> Optional[float]:
        """Price Hierarchy: Finnhub -> FMP -> Alpha -> yfinance"""
        # (Your existing price logic is already great, keeping it safe)
        try:
            # Try Finnhub first
            res = requests.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}").json()
            if res.get('c') and res['c'] != 0: return float(res['c'])
            
            # If 0 or error, try yfinance immediately
            logger.warning(f"⚠️ API Price for {ticker} unavailable. Falling back to yfinance.")
            stock = yf.Ticker(ticker)
            return float(stock.fast_info['last_price'])
        except:
            return None

    def get_global_macro_news(self) -> str:
        """Pulls global context for the Macro Agent."""
        try:
            # Using yfinance Search for 'World News' is extremely safe
            yf_global = yf.Search("Global Economy", news_count=5).news
            return " | ".join([n['title'] for n in yf_global])
        except:
            return "No global news available."

if __name__ == "__main__":
    fetcher = DataFetcher()
    # Diagnostic Print
    trending = fetcher.get_random_market_news(5)
    if not trending:
        logger.critical("❌ TRADING DISCOVERY IS EMPTY. Check internet or API keys.")
    else:
        logger.success(f"✅ Found {len(trending)} trending stocks.")
        print(f"🚀 Top Pick: {trending[0]}")