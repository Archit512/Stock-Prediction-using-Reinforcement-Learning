import numpy as np
from stable_baselines3 import PPO
from loguru import logger
from src.database import db

class TradingBrain:
    def __init__(self, model_path="models/ppo_trading_model_v1.zip"):
        try:
            # 1. Load the trained PPO 'Brain'
            self.model = PPO.load(model_path)
            logger.success(f"RL Brain loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            self.model = None

    def get_action(self, price_change, sentiment, panic):
        """
        Input: Real-time features
        Output: (ActionType, FinalSize)
        """
        if self.model is None:
            logger.warning("No model loaded. Defaulting to HOLD.")
            return 0, 0.0 # Default to HOLD if model is missing

        try:
            # 2. Sync Global Balance for the 'Balance_Ratio' feature
            # Added error handling so a split-second database timeout doesn't crash the bot
            account = db.get_account_status()
            balance_ratio = float(account.get('current_balance', 200000.0)) / 10000.0
        except Exception as e:
            logger.error(f"Failed to fetch account status during live tick: {e}")
            balance_ratio = 1.0 # Assume starting capital if DB drops momentarily

        try:
            # 3. Create the 'Observation' (The Agent's Eyes)
            obs = np.array([price_change, sentiment, panic, balance_ratio], dtype=np.float32)

            # 4. Predict the Action
            # deterministic=True: Best decision only, no exploration in live trading!
            action, _states = self.model.predict(obs, deterministic=True)

            # 5. Extract results (Action 0-2, Size 0-1)
            # CRITICAL FIX: Round the float BEFORE converting to integer
            action_type = int(np.clip(round(float(action[0])), 0, 2))
            
            # Failsafe bounds check to ensure it strictly stays 0, 1, or 2
            action_type = max(0, min(2, action_type))
            
            requested_size = float(action[1])

            # --- FINAL SAFETY: RE-APPLY KELLY GOVERNOR ---
            p = (sentiment + 1) / 2
            kelly_size = max(0.0, float((2 * p - 1)))
            
            # 🔥 NEW: Minimum Allocation Floor for Positive Sentiment
            # If the news is positive, ensure the Kelly governor allows at least a 5% trade
            if sentiment > 0.0:
                kelly_size = max(0.05, kelly_size)
            
            final_size = min(requested_size, kelly_size)
            
            # Failsafe bounds check for allocation size
            final_size = max(0.0, min(1.0, final_size))

            return action_type, final_size
            
        except Exception as e:
            logger.error(f"Inference failed to calculate action: {e}")
            return 0, 0.0 # Safe default: Do nothing (HOLD) if math fails