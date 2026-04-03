from config import settings
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional

# ------------------ OUTPUT SCHEMA ------------------ #
class SentimentResult(BaseModel):
    sentiment_score: float = Field(description="Score from -1.0 to 1.0")
    reasoning: str = Field(description="Financial logic")

# ------------------ MAIN AGENT ------------------ #
class DualGroupAgent:
    def __init__(self):
        self.parser = JsonOutputParser(pydantic_object=SentimentResult)
        self.prompt = ChatPromptTemplate.from_template(
            "Analyze {ticker} news: {headline}. Return ONLY JSON.\n{format_instructions}"
        )

    def _call_llm(self, llm, ticker, headline) -> Optional[dict]:
        try:
            chain = self.prompt | llm | self.parser
            result = chain.invoke({
                "ticker": ticker,
                "headline": headline,
                "format_instructions": self.parser.get_format_instructions()
            })
            # Convert Pydantic model to dict if needed
            if hasattr(result, 'dict'):
                return result.dict()
            return result
        except Exception as e:
            logger.debug(f"LLM Error: {e}")
            return None

    

    def _get_group1_opinion(self, ticker, headline):
        # --- 1. TRY HUGGING FACE (DEEPSEEK-R1) ---
        if settings.HF_API_KEY:
            try:
                llm = HuggingFaceEndpoint(
                    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    task="text-generation",
                    max_new_tokens=1024,
                    huggingfacehub_api_token=settings.HF_API_KEY,
                    temperature=0.6 # Recommended for R1 Distill
                )
                chat_model = ChatHuggingFace(llm=llm)
                
                # We handle the reasoning tag issue here
                res = self._call_llm(chat_model, ticker, headline)
                if res: 
                    return res
                logger.warning("Group 1: Hugging Face failed to return valid JSON.")
            except Exception as e:
                logger.debug(f"Hugging Face Error: {e}")

        # --- 2. FALLBACK TO OPENROUTER (LLAMA 3.3 FREE) ---
        if settings.OPENROUTER_API_KEY:
            logger.info("Group 1: Falling back to OpenRouter Llama...")
            res = self._call_llm(
                ChatOpenAI(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    api_key=settings.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=15
                ),
                ticker, headline
            )
            if res: 
                return res

        # --- 3. LAST RESORT (NEMOTRON FREE) ---
        if settings.OPENROUTER_API_KEY:
            logger.warning("Group 1: Llama failed. Trying Nemotron fallback...")
            return self._call_llm(
                ChatOpenAI(
                    model="nvidia/nemotron-3-super-120b-a12b:free",
                    api_key=settings.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=15
                ),
                ticker, headline
            )

        return None
        

    def _get_group2_opinion(self, ticker, headline):
        # 1. Cohere (FIXED: Using 08-2024 active model name)
        if settings.COHERE_API_KEY:
            res = self._call_llm(
                ChatCohere(
                    model="command-r-plus-08-2024", # Updated to fix 404
                    cohere_api_key=settings.COHERE_API_KEY,
                    timeout=15
                ),
                ticker, headline
            )
            if res: return res

        # 2. Groq Fallback
        if settings.GROQ_API_KEY:
            logger.warning("Group 2: Cohere failed. Trying Groq...")
            res = self._call_llm(
                ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=settings.GROQ_API_KEY,
                    timeout=15
                ),
                ticker, headline
            )
            if res: return res

        # 3. Gemini Fallback
        if settings.GEMINI_API_KEY:
            logger.warning("Group 2: Groq failed. Trying Gemini...")
            return self._call_llm(
                ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=settings.GEMINI_API_KEY,
                    timeout=15
                ),
                ticker, headline
            )
        return None

    def analyze(self, ticker, headline):
        # --- BREAK CIRCULAR IMPORT HERE ---
        # If you need to check macro state within analysis, import it locally
        # from src.agents.macro_agent import MacroSentinel 
        
        logger.info(f"🧠 Analyzing {ticker}...")
        op1 = self._get_group1_opinion(ticker, headline)
        op2 = self._get_group2_opinion(ticker, headline)

        if op1 and op2:
            a = op1['sentiment_score']
            b = op2['sentiment_score']
            avg = round((op1['sentiment_score'] + op2['sentiment_score']) / 2, 2)

            if((a > 0 and b > 0) or (a < 0 and b < 0)) :
                    return {"score": avg, "reason": op1['reasoning'], "status": "VERIFIED"}
           
            if((a < 0 and b > 0) or (a > 0 and b < 0)) :
                diff = abs(op1['sentiment_score'] - op2['sentiment_score'])
                
                if diff <= settings.CONSENSUS_THRESHOLD:
                        logger.success(f"✅ Consensus: {avg}")
                        return {"score": avg, "reason": op1['reasoning'], "status": "CONFLICT"}
                else:
                        logger.error(f"❌ Hallucination Detected (Diff: {diff:.2f}).")
                        # FIXED: Return neutral response instead of None
                        return {"score": avg, "reason": "Conflicting AI opinions detected", "status": "CONFLICT"}

        survivor = op1 or op2
        if survivor:
            logger.warning(f"📡 Only one AI group responded. Score: {survivor['sentiment_score']}")
            return {"score": survivor['sentiment_score'], "reason": survivor['reasoning'], "status": "UNVERIFIED"}

        logger.critical(f"🚨 ALL AI GROUPS FAILED for {ticker}.")
        # FIXED: Return neutral response instead of None
        return {"score": 0.0, "reason": "All AI groups failed - neutral default", "status": "FAILED"}
    