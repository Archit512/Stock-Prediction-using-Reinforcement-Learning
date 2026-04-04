from config import settings
from loguru import logger
# REMOVE: from src.agents.sentiment_agent import DualGroupAgent

class MacroSentinel:
    def __init__(self):
        # We no longer initialize self.analyzer here at the top level
        self.analyzer = None 
        
    def get_panic_status(self, global_news_string: str):
        # --- FIXED: LOCAL IMPORT TO BREAK CIRCULAR LOOP ---
        from src.agents.sentiment_agent import DualGroupAgent
        if self.analyzer is None:
            self.analyzer = DualGroupAgent()

        if not global_news_string or len(global_news_string) < 30:
            logger.warning("Macro news string too short. Defaulting to NORMAL.")
            return {"panic_score": 0, "regime": "NORMAL", "reason": "Insufficient news data."}

        macro_prompt = (
            f"Analyze these headlines for systemic financial risk: {global_news_string}\n"
            "Return JSON: {'score': float, 'reason': 'str'}"
        )
        
        # Use the locally imported analyzer
        result = self.analyzer.analyze("GLOBAL", macro_prompt)
        
        # FIX: Handle None result when all AI groups fail
        if result is None:
            logger.warning("AI analysis failed. Defaulting to NORMAL.")
            return {"panic_score": 0, "regime": "NORMAL", "reason": "AI analysis unavailable."}
        
        # Ensure we use the right keys to avoid errors
        sentiment_score = result.get('score', 0) 
        panic_score = round((1 - sentiment_score) * 5, 1)
        
        regime = "NORMAL"
        if panic_score > 7.5: regime = "CRASH"
        elif panic_score > 4.5: regime = "CAUTION"
            
        return {
            "panic_score": panic_score, # Match the key used in your test file
            "regime": regime,
            "reason": result.get('reason', 'AI Consensus failed.')
        }