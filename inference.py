import numpy as np
from stable_baselines3 import PPO
from loguru import logger
from src.database import db

class TradingBrain:
    def __init__(self, model_path="models/ppo_trading_model_v1.zip"):
        try:
            # 1. Load the trained PPO 'Brain'
            self.model = PPO.load(model_path)
            logger.success(f"🧠 RL Brain loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"🚨 Failed to load RL model: {e}")
            self.model = None

    def get_action(self, price_change, sentiment, panic):
        """
        Input: Real-time features
        Output: (ActionType, FinalSize)
        """
        if self.model is None:
            return 0, 0.0 # Default to HOLD if model is missing

        # 2. Sync Global Balance for the 'Balance_Ratio' feature
        account = db.get_account_status()
        balance_ratio = float(account['current_balance']) / 10000.0

        # 3. Create the 'Observation' (The Agent's Eyes)
        obs = np.array([price_change, sentiment, panic, balance_ratio], dtype=np.float32)

        # 4. Predict the Action
        # deterministic=True: We want the best decision, no more exploration in live trading!
        action, _states = self.model.predict(obs, deterministic=True)

        # 5. Extract results (Action 0-2, Size 0-1)
        action_type = int(action[0])
        requested_size = action[1]

        # --- FINAL SAFETY: RE-APPLY KELLY GOVERNOR ---
        # Even in inference, we never trust the agent's size blindly
        p = (sentiment + 1) / 2
        kelly_size = max(0, (2 * p - 1))
        final_size = min(requested_size, kelly_size)

        return action_type, final_size