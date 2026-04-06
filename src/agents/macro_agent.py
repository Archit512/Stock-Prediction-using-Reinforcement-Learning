from config import settings
from loguru import logger

class MacroSentinel:
    def __init__(self):
        # We no longer initialize self.analyzer here at the top level
        self.analyzer = None 
        
    def get_panic_status(self, global_news_string: str):
        # 🛡️ GLOBAL SAFETY NET: Prevents ANY error from bubbling up to coordinator.py
        try:
            from src.agents.sentiment_agent import DualGroupAgent
            if self.analyzer is None:
                self.analyzer = DualGroupAgent()

            if not global_news_string or len(global_news_string) < 30:
                logger.warning("⚠️ Macro news string too short. Defaulting to NORMAL.")
                return {"panic_score": 0.0, "regime": "NORMAL", "reason": "Insufficient news data."}

            macro_prompt = (
                f"Analyze these headlines for systemic financial risk: {global_news_string}\n"
                "Return strict JSON format: {'score': float (between -1.0 and 1.0), 'reason': 'str'}"
            )
            
            result = self.analyzer.analyze("GLOBAL", macro_prompt)
            
            # 🛡️ TYPE SAFETY: Ensure the AI actually returned a dictionary
            if not isinstance(result, dict):
                logger.warning(f"⚠️ AI returned invalid format type ({type(result)}). Defaulting to NORMAL.")
                return {"panic_score": 0.0, "regime": "NORMAL", "reason": "AI returned invalid format."}
            
            # 🛡️ BULLETPROOF EXTRACTION: Check multiple possible keys the AI might hallucinate
            raw_score = result.get('score', result.get('sentiment_score', result.get('panic_score', 0.0)))
            
            # 🛡️ FLOAT CASTING: Ensure the AI didn't return a string like "0.5" or "neutral"
            try:
                sentiment_score = float(raw_score)
            except (ValueError, TypeError):
                logger.warning(f"⚠️ AI returned non-numeric score '{raw_score}'. Defaulting to 0.0.")
                sentiment_score = 0.0
            
            # Clamp sentiment between -1.0 and 1.0 just in case the AI goes rogue with numbers
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Calculate panic: 1.0 (Great) -> 0 Panic. -1.0 (Terrible) -> 10 Panic.
            panic_score = round((1 - sentiment_score) * 5, 1)
            
            regime = "NORMAL"
            if panic_score > 7.5: 
                regime = "CRASH"
            elif panic_score > 4.5: 
                regime = "CAUTION"
                
            return {
                "panic_score": panic_score,
                "regime": regime,
                "reason": str(result.get('reason', 'AI Consensus failed.'))
            }
            
        except Exception as e:
            # 🛡️ ULTIMATE FALLBACK: If absolutely anything fails, keep trading safely.
            logger.error(f"🚨 MacroSentinel Critical Failure: {e}. Defaulting to safe NORMAL regime.")
            return {
                "panic_score": 0.0, 
                "regime": "NORMAL", 
                "reason": f"System error caught: {str(e)}"
            }