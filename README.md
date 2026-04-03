# Stock Prediction using Reinforcement Learning

An AI-driven stock intelligence and trading research system that combines:
- Multi-source market data ingestion
- Multi-LLM sentiment consensus
- Macro risk regime detection
- Reinforcement learning (PPO) for action selection
- Supabase-backed persistent state and training logs

This repository is designed for scheduled cloud execution cycles and iterative model training.

## What This Project Does

The coordinator runs in two cycle modes:
- Light cycle (every 15 minutes): updates macro status and price sync for watchlist and holdings
- Heavy cycle (hourly): full AI pipeline including sentiment analysis, watchlist evaluation, and new ticker discovery

Core workflow:
1. Fetch macro and market news
2. Score risk regime using MacroSentinel
3. Process holdings and watchlist
4. Log signals to Supabase
5. Train PPO model on historical logged signals
6. Use trained model for inference (when enabled)

## Repository Structure

```text
Stock-Prediction-using-Reinforcement-Learning/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── coordinator.py
├── train_ppo.py
├── test_config.py
├── app/
│   └── main.py
└── src/
	├── __init__.py
	├── database.py
	├── inference.py
	├── env/
	├── rl_env/
	│   └── cloud_env.py
	└── agents/
		├── __init__.py
		├── data_fetcher.py
		├── sentiment_agent.py
		├── macro_agent.py
		└── test_llm_datafetcher.py
```

- [coordinator.py](coordinator.py): Main orchestration loop and cycle logic
- [train_ppo.py](train_ppo.py): RL training entry point using PPO
- [config.py](config.py): Environment-based settings via Pydantic
- [requirements.txt](requirements.txt): Python dependencies
- [test_config.py](test_config.py): Configuration sanity test
- [src/database.py](src/database.py): Supabase read/write layer
- [src/inference.py](src/inference.py): TradingBrain PPO inference wrapper
- [src/rl_env/cloud_env.py](src/rl_env/cloud_env.py): Custom Gymnasium environment for persistent cloud training
- [src/agents/data_fetcher.py](src/agents/data_fetcher.py): Price + news data collection (Alpha Vantage, FMP, yfinance)
- [src/agents/sentiment_agent.py](src/agents/sentiment_agent.py): Multi-provider LLM sentiment consensus
- [src/agents/macro_agent.py](src/agents/macro_agent.py): Macro panic/regime detection
- [src/agents/test_llm_datafetcher.py](src/agents/test_llm_datafetcher.py): Agent logic smoke test
- [app/main.py](app/main.py): Placeholder for API service entrypoint

## Tech Stack

- Python
- Stable-Baselines3 (PPO)
- Gymnasium
- Supabase
- LangChain + multiple LLM providers
- yfinance, Finnhub, Alpha Vantage, FMP
- Loguru

## Setup

### 1. Clone and enter project

```bash
git clone <your-repo-url>
cd Stock-Prediction-using-Reinforcement-Learning
```

### 2. Create virtual environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the repository root with the following keys:

```env
SUPABASE_URL=
SUPABASE_KEY=

GEMINI_API_KEY=
GROQ_API_KEY=
HF_API_KEY=
COHERE_API_KEY=
OPENROUTER_API_KEY=

ALPHA_VANTAGE_KEY=
FINNHUB_KEY=
FMP_KEY=

MAX_WATCHLIST_SIZE=10
CONSENSUS_THRESHOLD=0.35
PANIC_THRESHOLD=7
```

Notes:
- HF_API_KEY and COHERE_API_KEY are optional in code, but recommended if you want full fallback coverage.
- Do not commit .env.

## Database Expectations (Supabase)

The code expects these tables (at minimum):
- user_account
- macro_status
- watchlist
- market_signals

Important assumptions in current code:
- user_account has a row with id=1
- watchlist includes ticker and is_holding fields
- market_signals stores price, sentiment, and logs used for RL training transformation

## How to Run

### Configuration check

```bash
python test_config.py
```

### Run one coordinator cycle

```bash
python coordinator.py
```

### Run LLM/data fetcher smoke test

```bash
python src/agents/test_llm_datafetcher.py
```

### Train PPO model

```bash
python train_ppo.py
```

Training output model name:
- ppo_trading_model_v1.zip

## Inference

Inference wrapper is implemented in [src/inference.py](src/inference.py).
Current coordinator code has TradingBrain initialization commented out; if you want live RL actioning, wire model loading in [coordinator.py](coordinator.py).

## Deployment Notes

The coordinator is designed for scheduled cloud execution (for example, every 15 minutes with a heavier hourly pass). Keep API and key rate limits in mind.

## Current Status and Known Gaps

- [app/main.py](app/main.py) is currently a placeholder.
- Inference path exists but is not fully enabled in coordinator by default.
- Data quality and schema consistency in Supabase are critical for training stability.

## Safety Disclaimer

This project is for research and educational purposes. It is not financial advice. Trading involves risk, including potential loss of capital.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and smoke checks
5. Open a pull request

## License

Add your preferred license file (for example, MIT) and update this section accordingly.
