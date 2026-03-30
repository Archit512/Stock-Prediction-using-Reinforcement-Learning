from config import settings
from loguru import logger
from agents.sentiment_agent import DualGroupAgent

class MacroSentinel:
    def __init__(self):
        # We reuse the Dual-Group Consensus logic
        self.analyzer = DualGroupAgent()
        
    def get_panic_status(self, global_news_string: str):
        """
        Analyzes the ACTUAL market news fetched by DataFetcher.
        """
        if not global_news_string or len(global_news_string) < 30:
            logger.warning("⚠️ Macro news string too short. Defaulting to NORMAL.")
            return {"score": 0, "regime": "NORMAL", "reason": "Insufficient news data."}

        # Instead of a dummy, we pass the REAL headlines
        macro_prompt = (
            f"As an AI Macro Economist, analyze these headlines for systemic financial risk: {global_news_string}\n"
            "Assess if there is a 'Black Swan' event (War, Pandemic, Global Recession).\n"
            "Return JSON: {'sentiment_score': float, 'reasoning': 'str'}"
        )
        
        # We use 'GLOBAL' as a category tag for Supabase logs
        result = self.analyzer.analyze("GLOBAL", macro_prompt)
        
        if not result:
            return {"score": 0, "regime": "NORMAL", "reason": "AI Consensus failed."}
            
        # Convert -1.0 to 1.0 (Sentiment) into 0-10 (Panic)
        # Score of -1.0 (Total Panic) becomes 10.0
        panic_score = round((1 - result['score']) * 5, 1)
        
        regime = "NORMAL"
        if panic_score > 7.5: 
            regime = "CRASH"
        elif panic_score > 4.5: 
            regime = "CAUTION"
            
        return {
            "score": panic_score,
            "regime": regime,
            "reason": result['reason'],
            "status": result['status']
        }