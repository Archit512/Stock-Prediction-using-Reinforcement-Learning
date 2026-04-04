import requests
import yfinance as yf
import random
from config import settings
from loguru import logger
from typing import List, Dict, Optional

class DataFetcher:
    def __init__(self):
        self.finnhub_key = settings.FINNHUB_KEY
        self.fmp_key = settings.FMP_KEY
        self.alpha_key = settings.ALPHA_VANTAGE_KEY
        
        # Session with a browser-like header.
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        })

    def get_random_market_news(self, limit: int = 15) -> List[Dict]:
        """Discovery Hierarchy: Alpha Vantage -> FMP -> yfinance"""
        discovered = []

        # --- 1. Alpha Vantage News Sentiment ---
        try:
            logger.debug("Attempting Alpha Vantage for discovery...")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&sort=LATEST&limit={limit + 10}&apikey={self.alpha_key}"
            res = self.session.get(url, timeout=15).json()
            
            if "feed" in res:
                for item in res["feed"]:
                    if item.get("ticker_sentiment") and len(item["ticker_sentiment"]) > 0:
                        ticker = item["ticker_sentiment"][0].get("ticker")
                        headline = item.get("title", "")
                        
                        if ticker and headline and len(ticker) <= 5:
                            discovered.append({
                                "ticker": ticker, 
                                "headline": headline, 
                                "source": "AlphaVantage"
                            })
                
                if discovered:
                    logger.success(f"Alpha Vantage: Found {len(discovered)} news items")
                    return discovered[:limit]
            else:
                 logger.debug("Alpha Vantage returned empty or hit rate limit.")
        except Exception as e:
            logger.debug(f"Alpha Vantage API Error: {e}")

        # --- 2. Try FMP ---
        try:
            logger.debug("Attempting FMP for discovery...")
            url = f"https://financialmodelingprep.com/api/v3/stock_news?limit={limit + 5}&apikey={self.fmp_key}"
            res = self.session.get(url, timeout=15).json()
            
            if isinstance(res, list) and len(res) > 0:
                for n in res:
                    if n.get('symbol') and n.get('title'):
                        discovered.append({
                            "ticker": n.get("symbol"), 
                            "headline": n.get("title"), 
                            "source": "FMP"
                        })
                
                if discovered:
                    logger.success(f"FMP: Found {len(discovered)} news items")
                    return discovered[:limit]
        except Exception as e:
            logger.debug(f"FMP API Error: {e}")

        # --- 3. yfinance Safety Net ---
        try:
            logger.info("Switching to yfinance Active Ticker discovery...")
            active_market_movers = [
                "NVDA", "TSLA", "AAPL", "AMD", "META", 
                "MSFT", "AMZN", "GOOGL", "PLTR", "COIN"
            ]
            random.shuffle(active_market_movers)
            
            for ticker in active_market_movers:
                if len(discovered) >= limit:
                    break
                    
                search_query = f"{ticker} stock news"
                search_results = yf.Search(search_query, news_count=1).news
                
                if search_results and isinstance(search_results, list) and len(search_results) > 0:
                    n = search_results[0]
                    headline = n.get('title') or n.get('headline', '')
                    
                    if headline:
                        discovered.append({
                            "ticker": ticker, 
                            "headline": headline, 
                            "source": "yfinance"
                        })
            
            if discovered:
                logger.success(f"yfinance: Found {len(discovered)} valid ticker-headline pairs")
                return discovered[:limit]
                
        except Exception as e:
            logger.error(f"yfinance Discovery Failed: {e}")

        return []

    # 🛠️ NEW METHOD: Targeted fetch fallback
    def get_ticker_news(self, ticker: str) -> Optional[str]:
        """Fetches the latest news headline for a specific targeted ticker."""
        try:
            logger.debug(f"Attempting targeted news fetch for {ticker}...")
            search_query = f"{ticker} stock news"
            search_results = yf.Search(search_query, news_count=1).news
            
            if search_results and isinstance(search_results, list) and len(search_results) > 0:
                n = search_results[0]
                headline = n.get('title') or n.get('headline', '')
                if headline:
                    return headline
        except Exception as e:
            logger.debug(f"Targeted yfinance news fetch failed for {ticker}: {e}")
            
        return None

    def get_price(self, ticker: str) -> Optional[float]:
        try:
            res = self.session.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}", timeout=10).json()
            if res.get('c') and res['c'] != 0: 
                return float(res['c'])
            
            stock = yf.Ticker(ticker)
            price = stock.info.get('currentPrice') or stock.fast_info.get('last_price')
            if price:
                return float(price)
            return None
        except Exception as e:
            logger.debug(f"Price fetch failed for {ticker}: {e}")
            return None

    def get_global_macro_news(self) -> str:
        try:
            logger.info("Fetching global macro news from yfinance...")
            yf_global = yf.Search("Global Economy", news_count=5).news
            
            headlines = []
            if yf_global:
                for n in yf_global:
                    if not n: continue
                    headline = n.get('title') or n.get('headline', '')
                    if headline:
                        headlines.append(headline)
            
            if headlines:
                result = " | ".join(headlines)
                logger.success(f"Got {len(headlines)} macro news headlines")
                return result
            else:
                return "No global news available."
                
        except Exception as e:
            logger.error(f"Global macro news error: {e}")
            return "No global news available."