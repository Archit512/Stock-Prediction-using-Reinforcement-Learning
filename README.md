# Stock-Prediction-using-Reinforcement-Learning
Real Time Stock Prediction based on Global News Sentiment Analysis.

## Required API Keys

Create a `.env` file in the project root (you can copy from `.env.example`) with the following keys:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GEMINI_API_KEY`
- `ALPHA_VANTAGE_KEY`
- `FINNHUB_KEY`
- `FMP_KEY`

### Where to get each key

- Supabase: create a project at https://supabase.com and copy URL + key from **Project Settings > API**.
- Gemini: generate a key at https://aistudio.google.com/app/apikey.
- Alpha Vantage: request free API key at https://www.alphavantage.co/support/#api-key.
- Finnhub: create account/key at https://finnhub.io/dashboard.
- Financial Modeling Prep (FMP): generate key at https://site.financialmodelingprep.com/developer/docs.

## Quick Setup

1. Copy `.env.example` to `.env`.
2. Paste your real keys in `.env`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify Supabase key pair:

```bash
python app/database.py
```

5. Verify market data APIs:

```bash
python src/agents/data_fetcher.py
```
