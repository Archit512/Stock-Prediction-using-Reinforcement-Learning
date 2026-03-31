import sys
import os

# Adds the project root (one level up from 'agents') to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.data_fetcher import DataFetcher
from src.agents.sentiment_agent import DualGroupAgent
from src.agents.macro_agent import MacroSentinel
from loguru import logger
import sys

# Customizing logger for clear test output
logger.remove()
logger.add(sys.stderr, format="<white>{time:HH:mm:ss}</white> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

def test_coordinator_logic():
    logger.info("🚀 Starting Stock-AI Logic Test (No DB/RL)")

    # 1. Initialize Components
    try:
        fetcher = DataFetcher()
        analyzer = DualGroupAgent() # This is your Gemini/Groq LLM logic
        macro = MacroSentinel()
        logger.success("✅ Components Initialized")
    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}")
        return

    # 2. Test Macro Logic (Global News + LLM)
    logger.info("Step 1: Testing Macro Sentinel...")
    macro_news = fetcher.get_global_macro_news()
    if macro_news:
        logger.info(f"Fetched Macro News: {macro_news[:100]}...")
        panic_status = macro.get_panic_status(macro_news)
        logger.info(f"📊 Macro Panic Score: {panic_status['panic_score']}/10")
    else:
        logger.warning("⚠️ No Macro News fetched.")

    # 3. Test Discovery & Sentiment (Ticker News + LLM)
    logger.info("Step 2: Testing Discovery & Sentiment Analysis...")
    trending = fetcher.get_random_market_news(limit=2) # Just test 2 to save tokens
    
    if not trending:
        logger.error("❌ Failed to fetch trending news. Check API keys/Proxy.")
        return

    for item in trending:
        ticker = item['ticker']
        headline = item['headline']
        logger.info(f"🔍 Analyzing {ticker}: {headline}")

        # Test the LLM Analysis
        result = analyzer.analyze(ticker, headline)
        
        if result:
            score = result.get('score', 0)
            reason = result.get('reason', 'No reason provided')
            color = "green" if score > 0.5 else "red"
            logger.info(f"<{color}>Score: {score}</{color}> | Reason: {reason}")
        else:
            logger.error(f"❌ LLM failed to analyze {ticker}")

    logger.success("✨ Logic Test Completed!")

if __name__ == "__main__":
    test_coordinator_logic()