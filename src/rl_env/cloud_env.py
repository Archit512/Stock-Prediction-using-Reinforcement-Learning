import gymnasium as gym
import numpy as np
from src.database import db
from loguru import logger

class CloudPersistentEnv(gym.Env):
    def __init__(self, df, is_training=True):
        super(CloudPersistentEnv, self).__init__()
        self.df = df
        self.is_training = is_training # <--- CRITICAL ADDITION
        
        # Observation: [Price_Change, Sentiment, Macro_Panic, Balance_Ratio]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Action: [Type (0:Hold, 1:Buy, 2:Sell), Size (0.0 to 1.0)]
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float32)

    def _get_obs(self):
        try:
            row = self.df.iloc[self.current_step]
            balance_ratio = self.balance / 200000.0 
            
            # Use .get() to avoid KeyErrors if columns mismatch slightly
            return np.array([
                row.get('price_change', 0.0), 
                row.get('sentiment_score', 0.0), 
                row.get('macro_panic_score', 0.0), 
                balance_ratio
            ], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting observation at step {self.current_step}: {e}")
            return np.zeros(4, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if self.is_training:
            # ISOLATED SANDBOX: Do not hit Supabase
            self.balance = 200000.0
            self.shares_held = 0.0
        else:
            # LIVE CLOUD INFERENCE: Fetch real state
            try:
                account = db.get_account_status()
                self.balance = float(account['current_balance'])
                self.shares_held = float(account['total_shares'])
            except Exception as e:
                logger.error(f"Failed to fetch account status, defaulting to 10k: {e}")
                self.balance = 10000.0
                self.shares_held = 0.0

        # Handle 'price' vs 'price_at_signal' from your DB
        initial_price = self.df.iloc[0].get('price', self.df.iloc[0].get('price_at_signal', 1.0))
        self.net_worth = self.balance + (self.shares_held * initial_price)
        
        return self._get_obs(), {}

    def step(self, action):
        try:
            # PPO outputs floats. We MUST round the action[0] to get exactly 0, 1, or 2
            action_type = int(round(action[0])) 
            requested_size = action[1]
            
            row = self.df.iloc[self.current_step]
            current_price = row.get('price', row.get('price_at_signal', 1.0))
            sentiment = row.get('sentiment_score', 0.0)

            # Kelly Governor
            p = (sentiment + 1) / 2
            kelly_size = max(0, (2 * p - 1)) 
            
            # 🔥 NEW: Minimum Allocation Floor (Must match inference.py!)
            if sentiment > 0.0:
                kelly_size = max(0.05, kelly_size)
                
            final_size = min(requested_size, kelly_size)

            # Execute trade
            if action_type == 1: # BUY
                cost = self.balance * final_size
                if current_price > 0:
                    self.shares_held += cost / current_price
                self.balance -= cost
            elif action_type == 2: # SELL
                self.balance += self.shares_held * current_price
                self.shares_held = 0

            # CRITICAL: Only update Supabase if we are deployed live
            if not self.is_training:
                db.update_account_status(self.balance, self.shares_held)

            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            
            # Value Updation (Reward)
            if done:
                new_price = current_price
            else:
                new_price = self.df.iloc[self.current_step].get('price', self.df.iloc[self.current_step].get('price_at_signal', 1.0))
                
            new_net_worth = self.balance + (self.shares_held * new_price)
            
            # Prevent math domain errors (log of 0 or negative)
            if new_net_worth <= 0 or self.net_worth <= 0:
                reward = -1.0 
            else:
                reward = np.log(new_net_worth / self.net_worth)
                
            self.net_worth = new_net_worth

            return self._get_obs(), float(reward), done, False, {}
            
        except Exception as e:
            logger.error(f"Step failed at index {self.current_step}: {e}")
            return self._get_obs(), -0.1, True, False, {} # End episode safely on crash