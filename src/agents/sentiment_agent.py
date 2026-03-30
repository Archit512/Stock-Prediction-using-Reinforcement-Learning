from config import settings
from loguru import logger
from langchain_community.chat_models import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional

class SentimentResult(BaseModel):
    sentiment_score: float = Field(description="Score from -1.0 to 1.0")
    reasoning: str = Field(description="Financial logic")

class DualGroupAgent:
    def __init__(self):
        self.parser = JsonOutputParser(pydantic_object=SentimentResult)
        self.prompt = ChatPromptTemplate.from_template(
            "Analyze {ticker} news: {headline}. Return ONLY JSON.\n{format_instructions}"
        )

    def _call_llm(self, llm, ticker, headline) -> Optional[dict]:
        """Generic invoker to handle errors for any model."""
        try:
            chain = self.prompt | llm | self.parser
            # Setting a 15-second timeout to prevent the bot from hanging
            return chain.invoke({
                "ticker": ticker, 
                "headline": headline, 
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            logger.debug(f"LLM Error: {e}")
            return None

    # --- 🔵 GROUP 1: OPENROUTER HIERARCHY ---
    def _get_group1_opinion(self, ticker, headline):
        # 1. DeepSeek (Reasoning King)
        res = self._call_llm(ChatOpenRouter(model="deepseek/deepseek-chat:free", openai_api_key=settings.OPENROUTER_API_KEY), ticker, headline)
        if res: return res
        
        # 2. NVIDIA Nemotron (Heavyweight Fallback)
        logger.warning("Group 1: DeepSeek failed. Trying Nemotron...")
        res = self._call_llm(ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", openai_api_key=settings.OPENROUTER_API_KEY), ticker, headline)
        return res

    # --- 🔴 GROUP 2: MULTI-CLOUD HIERARCHY ---
    def _get_group2_opinion(self, ticker, headline):
        # 1. Cohere (Finance Specialist)
        if settings.COHERE_API_KEY:
            res = self._call_llm(ChatCohere(model="command-r", cohere_api_key=settings.COHERE_API_KEY), ticker, headline)
            if res: return res
        
        # 2. Groq (Speed King)
        logger.warning("Group 2: Cohere failed/missing. Trying Groq...")
        res = self._call_llm(ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=settings.GROQ_API_KEY), ticker, headline)
        if res: return res

        # 3. Gemini (The Ultimate Safety Net)
        logger.warning("Group 2: Groq failed. Trying Gemini...")
        res = self._call_llm(ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GEMINI_API_KEY), ticker, headline)
        return res

    # --- ⚔️ THE RESILIENT DUEL ---
    def analyze(self, ticker, headline):
        logger.info(f"🧠 Analyzing {ticker}...")
        
        op1 = self._get_group1_opinion(ticker, headline)
        op2 = self._get_group2_opinion(ticker, headline)

        # CASE 1: Full Success (Consensus)
        if op1 and op2:
            diff = abs(op1['sentiment_score'] - op2['sentiment_score'])
            if diff <= settings.CONSENSUS_THRESHOLD:
                avg = round((op1['sentiment_score'] + op2['sentiment_score']) / 2, 2)
                logger.success(f"✅ Consensus: {avg}")
                return {"score": avg, "reason": op1['reasoning'], "status": "VERIFIED"}
            else:
                logger.error(f"❌ Hallucination Detected (Diff: {diff:.2f}). Skipping.")
                return None

        # CASE 2: Partial Success (Single Group Alive)
        survivor = op1 or op2
        if survivor:
            logger.warning(f"📡 Only one AI group responded. Using unverified score: {survivor['sentiment_score']}")
            return {"score": survivor['sentiment_score'], "reason": survivor['reasoning'], "status": "UNVERIFIED"}

        # CASE 3: Total Failure (Skip)
        logger.critical(f"🚨 ALL AI GROUPS FAILED for {ticker}. News skipped for this cycle.")
        return None