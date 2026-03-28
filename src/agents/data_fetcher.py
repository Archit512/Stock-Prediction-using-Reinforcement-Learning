import requests
import time
from config import settings
from loguru import logger

class DataFetcher:
    def __init__(self):
        self.alpha_key = settings.ALPHA_VANTAGE_KEY
        self.finnhub_key = settings.FINNHUB_KEY
        self.fmp_key = settings.FMP_KEY

    def get_current_price(self, ticker: str):
        """Failover logic: Alpha -> Finnhub -> FMP."""
        
        # 1. Try Alpha Vantage (Strict limits)
        price = self._finnhub_price(ticker)
        if price: return price
        
        # 2. Try Finnhub (60 calls/min)
        logger.warning(f"🔄 Alpha limited. Trying Finnhub for {ticker}...")
        price = self._alpha_price(ticker) 
        if price: return price
        
        # 3. Try FMP (250 calls/day)
        logger.warning(f"🔄 Finnhub failed. Trying FMP for {ticker}...")
        price = self._fmp_price(ticker)
        if price: return price

        logger.error(f"❌ All Price APIs failed for {ticker}")
        return None

    def _alpha_price(self, ticker: str):
        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": self.alpha_key}
        try:
            data = requests.get(url, params=params).json()
            return float(data["Global Quote"]["05. price"])
        except: return None

    def _finnhub_price(self, ticker: str):
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
        try:
            data = requests.get(url).json()
            return float(data["c"]) # 'c' is current price
        except: return None

    def _fmp_price(self, ticker: str):
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={self.fmp_key}"
        try:
            data = requests.get(url).json()
            return float(data[0]["price"])
        except: return None

    def get_stock_news(self, ticker: str):
        """Fetches news via Alpha Vantage (Primary News Source)."""
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": self.alpha_key}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            return data.get("feed", [])
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return []

if __name__ == "__main__":
    fetcher = DataFetcher()
    ticker = "MSFT"
    print(f"💰 Fetching {ticker} price via Failover System...")
    print(f"Final Price Result: {fetcher.get_current_price(ticker)}")

    headlines = fetcher.get_stock_news(ticker)
    
    print(f"\n📰 LATEST NEWS HEADLINES:")
    if headlines:
        for i, news in enumerate(headlines, 1):
            print(f"{i}. {news['title']}")
    else:
        print("❌ No news found or API limit reached.")
        
    print("\n--- TEST COMPLETE ---")